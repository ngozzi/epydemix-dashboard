# Scoping: age-cohort + demography pertussis model

**Goal.** Model the *actual* CDC DTaP/Tdap dose calendar — doses delivered at
specific ages (2, 4, 6, 15–18 months; 4–6 y; 11 y; then every 10 y) with the
required minimum spacing — rather than the current program-level approximation
(coverage of a coarse age band over a time window).

**Status quo.** The dashboard pertussis model is a closed population
(`run_pertussis_stub`), age-structured only through the 5 contact-matrix bands
(`0-4, 5-19, 20-49, 50-64, 65+`). Vaccination is a per-day dose flow S→Sₚ driven
by campaigns. Three structural facts block a faithful dose calendar:

1. **No aging.** Individuals never move between age bands, so "reaches 2 months"
   or "reaches 11 years" is not an event that can trigger a dose.
2. **No births/deaths.** There is no inflow of new susceptible infants — the
   population that the infant series targets never renews.
3. **Coarse infancy resolution.** Doses 1–5 (2 mo → 6 y) all live in the single
   `0-4` band; spacing between doses cannot be represented at 5-year resolution.

---

## What a faithful version requires

### 1. Finer age resolution
Replace the 5 bands with single-cohort resolution where the schedule is dense:
- **Monthly** cohorts for 0–24 months (captures 2/4/6/15–18-month doses).
- **Yearly** cohorts for 2–18 years (captures the 4–6 y and 11 y doses).
- Coarser bands (e.g. 5–10 y) for adults where only decennial boosters apply.

This multiplies the compartment count: 8 epi states × ~40 age cohorts ≈ **320
compartments** (vs 8 today).

### 2. Aging process
Add spontaneous transitions moving each cohort to the next at rate `1 /
(cohort width in days)` for **every** epi state (S, E, I, R, Sₚ, Eₚ, Iₚ, Rₚ).
Aging must preserve immune status (an aging Sₚ stays Sₚ).

### 3. Demography (births + deaths)
- **Births**: inflow into the youngest cohort's S at the crude birth rate ×
  population (new fully-susceptible infants). Optionally a maternal-immunity
  fraction seeded into Sₚ/Rₚ.
- **Deaths**: outflow μ from every compartment (age-specific mortality ideally).
  Needed so the population and age pyramid stay stable over multi-year runs.

### 4. Age-triggered vaccination
Redefine a "campaign" as a **dose event on a cohort**: when a cohort occupies the
target age, apply coverage × efficacy as an S→Sₚ (or Sₚ→Rₚ booster) pulse, with
per-dose coverage and the minimum-spacing constraint enforced between doses.
Booster doses (Tdap 11 y, decennial) top up waning immunity (Sₚ→Rₚ or Rₚ refresh).

---

## Implementation options

| Option | Approach | Fidelity | Effort | Risk |
|---|---|---|---|---|
| **A. In epydemix** | Age-replicate all 8 states, add aging + births/deaths as spontaneous transitions, drive vaccination by cohort age | High | High | epydemix has no native aging/vital-dynamics or age-triggered dosing; ~320 compartments may strain the stochastic engine and the dashboard UI/viz |
| **B. Standalone cohort module** | Custom deterministic (ODE) or stochastic age-cohort model outside epydemix; import contact matrices for mixing; surface results in the dashboard as a new engine | High | High | Diverges from epydemix; duplicate machinery for interventions/plots; more code to maintain |
| **C. Hybrid approximation** | Keep epydemix transmission on current bands; add a light demographic layer + time-varying vaccination rates that *mimic* cohort dosing (e.g. constant infant-cohort inflow vaccinated at fixed rate) | Medium | Medium | Still not a true per-age dose calendar; a refinement of what exists, not the real schedule |

---

## Recommendation

If the objective is **decision support for Lane County booster/coverage policy**,
Option **C** likely gives most of the value for a fraction of the effort: add
births (infant renewal) + deaths so multi-year dynamics are stable, and express
DTaP/Tdap as sustained age-band coverage (already done) plus a booster refresh
Sₚ→Rₚ. That captures "coverage and boosters vs. resurgence" without a 320-state
model.

If the objective is **dose-timing/spacing fidelity** (e.g. evaluating a delayed
or compressed schedule), Option **A** or **B** is required. Recommend **A** as a
prototype first (stay in epydemix, reuse plotting/interventions), validated on a
small cohort set (monthly 0–24 mo + yearly to 18 y + 3 adult bands ≈ 30 cohorts),
before deciding whether the engine scales.

## Data needed
- Single-year (and monthly, 0–2 y) age distribution for the target geography.
- Crude birth rate and age-specific mortality.
- Per-dose coverage (not just ≥3-dose) and real-world per-dose efficacy.
- Contact matrix at the finer age resolution (or a mapping from the 5-band matrix).

## Suggested phases
1. **Demography prototype** (Option C core): births/deaths on current 8-state
   model; verify a stable multi-year endemic equilibrium.
2. **Cohort prototype** (Option A, ~30 cohorts): aging + age-triggered single
   dose; validate one cohort flows S→Sₚ at the right age.
3. **Full schedule**: all 5 DTaP doses + Tdap + decennial with spacing; booster
   refresh of waned immunity.
4. **Dashboard integration**: cohort-aware viz, schedule editor, performance pass.
