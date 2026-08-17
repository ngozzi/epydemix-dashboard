# engine/calibration.py
#
# Compare the pertussis partial-immunity model against an observed case dataset
# (cases by age band and vaccination status) and support three workflows:
#   1. Overlay / compare  -> observed_targets() + modeled_case_summary() + fit_error()
#   2. Seed & project      -> seed_ic_from_observed()
#   3. Auto-calibrate      -> calibrate_to_observed()
#
# Track proxy: "No" (not up to date) -> naive track (I / E_to_I),
#              "Yes" (up to date)    -> partial track (Ip / Ep_to_Ip).

from __future__ import annotations
import numpy as np

from constants import DEFAULT_AGE_GROUPS
from data.observed_datasets import aggregate_to_bands


# ----------------------------------------------------------------------------
# Observed targets
# ----------------------------------------------------------------------------
def observed_targets(raw: dict) -> dict:
    """Summarise an observed dataset into comparison targets."""
    bands = aggregate_to_bands(raw)
    counts = np.array([bands[a]["total"] for a in DEFAULT_AGE_GROUPS], dtype=float)
    naive = np.array([bands[a]["naive"] for a in DEFAULT_AGE_GROUPS], dtype=float)
    partial = np.array([bands[a]["partial"] for a in DEFAULT_AGE_GROUPS], dtype=float)

    total = counts.sum()
    known = naive + partial
    with np.errstate(invalid="ignore", divide="ignore"):
        share_by_band = np.where(known > 0, partial / known, np.nan)

    known_total = float((naive + partial).sum())
    return {
        "age_counts": counts,
        "age_dist": counts / total if total > 0 else counts,
        "naive_by_band": naive,
        "partial_by_band": partial,
        "partial_share_by_band": share_by_band,
        "overall_partial_share": (float(partial.sum()) / known_total) if known_total > 0 else np.nan,
        "total_cases": total,
    }


# ----------------------------------------------------------------------------
# Modeled case summary (from a completed run)
# ----------------------------------------------------------------------------
def modeled_case_summary(df_trans) -> dict:
    """
    Cumulative new infections per age band and track from a run's transitions.
    Naive new cases   = sum_t E_to_I_<band>
    Partial new cases = sum_t Ep_to_Ip_<band>
    """
    def col_sum(col):
        return float(df_trans[col].sum()) if col in df_trans.columns else 0.0

    naive = np.array([col_sum(f"E_to_I_{a}") for a in DEFAULT_AGE_GROUPS], dtype=float)
    partial = np.array([col_sum(f"Ep_to_Ip_{a}") for a in DEFAULT_AGE_GROUPS], dtype=float)
    total = naive + partial
    tsum = total.sum()

    with np.errstate(invalid="ignore", divide="ignore"):
        share_by_band = np.where(total > 0, partial / total, np.nan)

    return {
        "age_counts": total,
        "age_dist": total / tsum if tsum > 0 else total,
        "naive_by_band": naive,
        "partial_by_band": partial,
        "partial_share_by_band": share_by_band,
        "overall_partial_share": (float(partial.sum()) / tsum) if tsum > 0 else np.nan,
        "total_cases": float(tsum),
    }


# ----------------------------------------------------------------------------
# Fit error between observed and modeled
# ----------------------------------------------------------------------------
def fit_error(obs: dict, mod: dict) -> dict:
    """
    Scalar fit diagnostics comparing shape (age distribution) and the
    vaccinated (partial) share of cases. Both are scale-free so they compare
    a case count (observed) against a modeled incidence (different magnitude).
    """
    age_rmse = float(np.sqrt(np.nanmean((obs["age_dist"] - mod["age_dist"]) ** 2)))

    o_s = obs["overall_partial_share"]
    m_s = mod["overall_partial_share"]
    share_err = float(abs(o_s - m_s)) if np.isfinite(o_s) and np.isfinite(m_s) else np.nan

    # combined score: age-distribution shape + vaccinated-share match
    score = age_rmse + (share_err if np.isfinite(share_err) else 1.0)
    return {"age_dist_rmse": age_rmse, "partial_share_err": share_err, "score": score}


# ----------------------------------------------------------------------------
# Seed initial conditions directly from observed counts
# ----------------------------------------------------------------------------
def seed_ic_from_observed(Nk, raw: dict, immune_pct: float,
                          partial_immune_pct: float = 71.0,
                          immune_pct_by_age=None) -> dict:
    """
    Build an 8-compartment initial-conditions dict where the seeded infections
    reproduce the observed age x vaccination distribution:
      No  -> naive track  (split E/I)
      Yes -> partial track (split Ep/Ip)
      Unknown -> allocated within band by the known naive:partial ratio.
    Background immunity fills Sp/Rp/R from the immune_pct slider.
    """
    Nk = np.asarray(Nk, dtype=float)
    bands = aggregate_to_bands(raw)
    ic = {c: np.zeros(len(Nk)) for c in ["S", "E", "I", "R", "Sp", "Ep", "Ip", "Rp"]}

    for i, a in enumerate(DEFAULT_AGE_GROUPS):
        naive = float(bands[a]["naive"])
        partial = float(bands[a]["partial"])
        unk = float(bands[a]["Unknown"])
        known = naive + partial
        if known > 0:
            naive += unk * naive / known
            partial += unk * partial / known
        else:
            naive += unk  # no known status -> assume naive

        ic["E"][i] = naive / 2.0
        ic["I"][i] = naive / 2.0
        ic["Ep"][i] = partial / 2.0
        ic["Ip"][i] = partial / 2.0

    # background immunity pool -> Sp / Rp / R. A per-age immunity vector (e.g. real
    # vaccination coverage by band) overrides the uniform immune_pct when supplied.
    if immune_pct_by_age is not None:
        immune = Nk * (np.asarray(immune_pct_by_age, dtype=float) / 100.0)
    else:
        immune = Nk * (immune_pct / 100.0)
    sp = partial_immune_pct / 100.0
    rem = max(0.0, 1.0 - sp)
    ic["Sp"] = immune * sp
    ic["Rp"] = immune * rem * (0.24 / 0.29)
    ic["R"] = immune * rem * (0.05 / 0.29)

    # Floor every non-S compartment to integers, then let S absorb the remainder
    # so the compartments sum exactly to Nk (per age group).
    for c in ["E", "I", "Ep", "Ip", "Sp", "Rp", "R"]:
        ic[c] = np.maximum(0, ic[c]).astype(int)
    seeded = ic["E"] + ic["I"] + ic["Ep"] + ic["Ip"] + ic["Sp"] + ic["Rp"] + ic["R"]
    ic["S"] = np.maximum(0, Nk - seeded)

    return ic


# ----------------------------------------------------------------------------
# Coarse auto-calibration
# ----------------------------------------------------------------------------
def calibrate_to_observed(base_scenario: dict, raw: dict, run_fn,
                          r0_grid=None, partial_immune_grid=None,
                          sigma_grid=None) -> dict:
    """
    Coarse grid search over the most identifiable knobs for this data
    (R0, initial Sp share of the immune pool, and sigma) minimising fit_error.

    `run_fn(scenario) -> (df_comp, df_trans)` runs one model evaluation.
    `base_scenario` is a full scenario config; this function copies it and
    overrides model_params / initial_conditions per grid point.

    Returns best params, best error, and the full evaluation trace.
    """
    import copy

    obs = observed_targets(raw)
    r0_grid = r0_grid if r0_grid is not None else [8.0, 12.0, 16.0]
    partial_immune_grid = partial_immune_grid if partial_immune_grid is not None else [50.0, 71.0, 90.0]
    sigma_grid = sigma_grid if sigma_grid is not None else [0.1, 0.2, 0.3]

    trace = []
    best = None
    for r0 in r0_grid:
        for pi in partial_immune_grid:
            for sig in sigma_grid:
                sc = copy.deepcopy(base_scenario)
                sc["model_params"]["R0"] = float(r0)
                sc["model_params"]["rel_infectiousness_partial"] = float(sig)
                sc["initial_conditions"]["partial_immune_pct"] = float(pi)
                df_comp, df_trans = run_fn(sc)
                mod = modeled_case_summary(df_trans)
                err = fit_error(obs, mod)
                point = {"R0": r0, "partial_immune_pct": pi,
                         "rel_infectiousness_partial": sig, **err,
                         "modeled_partial_share": mod["overall_partial_share"]}
                trace.append(point)
                if best is None or err["score"] < best["score"]:
                    best = point

    return {"best": best, "trace": trace, "observed": {
        "overall_partial_share": obs["overall_partial_share"],
        "age_dist": obs["age_dist"].tolist(),
    }}
