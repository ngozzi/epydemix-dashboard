"""Sensitivity sweep for omega3 (vaccine-derived waning, Sp -> S).

Before omega3 existed, Sp had exactly one exit -- infection -- so a vaccinated
individual who avoided infection stayed at reduced susceptibility delta forever.
This sweep quantifies what adding the transition does, and in particular whether
it moves the modelled age distribution toward what Oregon actually observed.

Observed target: OHA 2023 Selected Reportable Communicable Disease Summary,
'Cases by Sex and Age Group', pertussis, aggregated to the model's five bands
(n=40, so treat as directional -- 2024's 1,105 cases would be a far better
target but OHA has not published the 2024 summary).

Run:  ./venv/Scripts/python.exe scripts/sensitivity_waning.py [geography] [years]
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import DEFAULT_AGE_GROUPS                      # noqa: E402
from schemas import MODEL_PARAM_SCHEMAS, INITIAL_CONDITION_DEFAULTS  # noqa: E402
import engine.run as R                                        # noqa: E402
from helpers import daily_doses_by_age                        # noqa: E402
from epydemix.population import load_epydemix_population      # noqa: E402

GEOGRAPHY = sys.argv[1] if len(sys.argv) > 1 else "United_States"
YEARS = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
SIM_LENGTH = int(YEARS * 365)
N_SIM = 5                     # sweep needs speed, not tight quantiles
MODEL = "SEIRS (Pertussis)"

# Oregon 2023 pertussis cases mapped onto DEFAULT_AGE_GROUPS.
# 0-4: 10 | 5-19: 22 | 20-49: 1 | 50-64: 1 | 65+: 6   (total 40)
# The 60-69 OHA band straddles 50-64/65+; it held 0 cases in 2023 so the
# split rule is immaterial here, but it will matter for other years.
OBSERVED_OR_2023 = np.array([10, 22, 1, 1, 6], dtype=float)
OBSERVED_SHARE = OBSERVED_OR_2023 / OBSERVED_OR_2023.sum()

SWEEP = [50.0, 20.0, 10.0, 5.0, 3.0]   # years; 50 ~= the old no-waning model


def base_scenario(population):
    mp = {p["key"]: p["default"] for p in MODEL_PARAM_SCHEMAS[MODEL]}
    ic = dict(INITIAL_CONDITION_DEFAULTS.get(MODEL,
                                             INITIAL_CONDITION_DEFAULTS))
    if not isinstance(ic, dict) or "infected_pct" not in ic:
        ic = {"infected_pct": 0.1, "immune_pct": 80.0,
              "partial_immune_pct": 71.0, "partial_infection_pct": 33.0}
    doses = daily_doses_by_age(campaigns=[], Nk=population.Nk,
                               sim_length=SIM_LENGTH,
                               age_groups=DEFAULT_AGE_GROUPS, dt=0.2)
    return {
        "model": MODEL, "population": population,
        "sim_length": SIM_LENGTH, "time_step": 0.2,
        "model_params": mp, "initial_conditions": ic,
        "contact_interventions": [],
        "vaccination_settings": {"target_compartments": ["S"]},
        "daily_doses_by_age": doses,
    }


def age_case_counts(df_trans):
    """New infections per band, both tracks."""
    def s(c):
        return float(df_trans[c].sum()) if c in df_trans.columns else 0.0
    naive = np.array([s(f"E_to_I_{a}") for a in DEFAULT_AGE_GROUPS])
    partial = np.array([s(f"Ep_to_Ip_{a}") for a in DEFAULT_AGE_GROUPS])
    return naive, partial


def main():
    print(f"geography={GEOGRAPHY}  horizon={YEARS}y ({SIM_LENGTH}d)  "
          f"Nsim={N_SIM}")
    pop = load_epydemix_population(GEOGRAPHY)
    R.N_SIM = N_SIM
    sc = base_scenario(pop)
    print(f"population={pop.Nk.sum():,.0f}  bands={DEFAULT_AGE_GROUPS}\n")

    rows = []
    for w in SWEEP:
        sc["model_params"]["waning_vaccine_to_susceptible_years"] = w
        _, df_trans = R.run_pertussis_stub(sc)
        naive, partial = age_case_counts(df_trans)
        total = naive + partial
        share = total / total.sum() if total.sum() else total
        l1 = float(np.abs(share - OBSERVED_SHARE).sum())
        rows.append({
            "omega3_years": w,
            "total_cases": total.sum(),
            "pct_partial_track": 100 * partial.sum() / total.sum()
            if total.sum() else np.nan,
            **{f"share_{a}": share[i] for i, a in enumerate(DEFAULT_AGE_GROUPS)},
            "L1_vs_OR2023": l1,
        })
        print(f"  omega3={w:>5.1f}y  cases={total.sum():>12,.0f}  "
              f"5-19 share={share[1]:.3f}  L1 vs OR2023={l1:.3f}")

    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "sensitivity_waning_results.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    print("\nObserved Oregon 2023 share: " +
          "  ".join(f"{a}={OBSERVED_SHARE[i]:.3f}"
                    for i, a in enumerate(DEFAULT_AGE_GROUPS)))
    print("\nModelled share by omega3:")
    print(df[["omega3_years"] + [f"share_{a}" for a in DEFAULT_AGE_GROUPS]
             + ["L1_vs_OR2023"]].round(3).to_string(index=False))
    best = df.loc[df.L1_vs_OR2023.idxmin()]
    print(f"\nclosest to observed: omega3={best.omega3_years}y  "
          f"(L1={best.L1_vs_OR2023:.3f})")


if __name__ == "__main__":
    main()
