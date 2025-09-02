import numpy as np
import pandas as pd

# small helpers
def _col_exists(trj, comp, age):
    return f"{comp}_{age}" in trj

def _series(trj, comp, age):
    # returns ndarray shape (Nsim, T)
    return np.asarray(trj[f"{comp}_{age}"])

def _age_list(population):
    return list(population.Nk_names) + ["total"]

def _denom(population, age):
    if age == "total":
        return population.Nk.sum()
    idx = list(population.Nk_names).index(age)
    return population.Nk[idx]


def compute_attack_rate(trj, population): 
    attack_rows = []
    for age in _age_list(population):
        key_ok = _col_exists(trj, "Recovered", age)
        if not key_ok:  # skip gracefully if some model variant doesn't produce it
            continue
        arr = _series(trj, "Recovered", age)          # (Nsim, T)
        final = arr[:, -1]                             # value at end of sim
        denom = _denom(population, age)
        ar = final / denom * 100.0                     # %
        attack_rows.append({
            "Age group": age,
            "Median (%)": np.median(ar),
            "95% CI low (%)": np.percentile(ar, 2.5),
            "95% CI high (%)": np.percentile(ar, 97.5)
        })
    return pd.DataFrame(attack_rows)


def compute_peak_size(trj, population):
    peak_rows = []
    for age in _age_list(population):
        if not _col_exists(trj, "Infected", age):
            continue
        arr = _series(trj, "Infected", age)           # (Nsim, T)
        peak_vals = arr.max(axis=1)                   # per-sim max
        peak_rows.append({
            "Age group": age,
            "Median peak": np.median(peak_vals),
            "95% CI low": np.percentile(peak_vals, 2.5),
            "95% CI high": np.percentile(peak_vals, 97.5)
        })
    return pd.DataFrame(peak_rows)


def compute_peak_time(trj, population):
    peaktime_rows = []
    for age in _age_list(population):
        if not _col_exists(trj, "Infected", age):
            continue
        arr = _series(trj, "Infected", age)           # (Nsim, T)
        argmax = arr.argmax(axis=1)                   # day index of peak per sim
        # summarize as median + IQR of day index; also show corresponding median date
        day_median = int(np.median(argmax))
        day_q1 = int(np.percentile(argmax, 2.5))
        day_q3 = int(np.percentile(argmax, 97.5))
        peaktime_rows.append({
            "Age group": age,
            "Median day": day_median,
            "95% CI low (day)": day_q1,
            "95% CI high (day)": day_q3,
        })
    return pd.DataFrame(peaktime_rows)