import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import base64
from epydemix.population import load_epydemix_population, Population
from epydemix.utils import compute_simulation_dates
from collections import Counter
from typing import Sequence, Any
from constants import START_DATE
from datetime import timedelta


@st.cache_data
def load_population(geography: str) -> Population:
    return load_epydemix_population(geography)


@st.cache_data
def load_locations():
    return pd.read_csv(
        "https://raw.githubusercontent.com/epistorm/epydemix-data/refs/heads/main/locations.csv"
    ).location.unique()


def contact_matrix_df(layer: str, matrices: dict, groups: list[str]) -> pd.DataFrame:
    """Return the matrix being visualized (layer or overall) as a labeled DataFrame."""
    if layer == "overall":
        M = np.sum(np.stack(list(matrices.values()), axis=0), axis=0)
    else:
        M = np.asarray(matrices[layer])
    return pd.DataFrame(M, index=groups, columns=groups)


def build_compartment_timeseries_df(trj, comp=None, age=None):

    # If comp and age are None, return all compartments and age groups
    if comp is None and age is None:
        df = pd.DataFrame()
        for key in trj.keys():
            series = np.asarray(trj[key])  # shape (Nsim, T)
            Nsim, T = series.shape
            # Get median
            df_temp = pd.DataFrame({
                "Day": np.arange(T),
                key + "_median": np.median(series, axis=0),
            })
            # Add each run as a separate column
            for i in range(Nsim):
                df_temp[f"{key}_run_{i+1}"] = series[i]
            if df.empty:
                df = df_temp
            else:
                df = pd.merge(df, df_temp, on="Day", how="outer")
        return df
    else:
        # If comp and age are not None, return the specified compartment and age group
        key = f"{comp}_{age}"
        if key not in trj:
            return None

        series = np.asarray(trj[key])  # shape (Nsim, T)
        Nsim, T = series.shape

        df = pd.DataFrame({
            "Day": np.arange(T),
            "Median": np.median(series, axis=0),
        })
        # add each run as a separate column
        for i in range(Nsim):
            df[f"Run_{i+1}"] = series[i]

        return df


def data_uri(path):
    p = Path(path)
    mime = "image/svg+xml" if p.suffix.lower()==".svg" else "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def daily_doses_by_age(
    campaigns: list[dict[str, Any]],
    Nk: Sequence[float] | np.ndarray,
    sim_length: int,
    age_groups: list[str],
    dt: float,
) -> pd.DataFrame:
    """
    Compute daily vaccine doses by age group, summed across all campaigns.

    Parameters
    ----------
    campaigns:
        List of campaign dicts. Expected fields (MVP):
          - start_day (int), end_day (int)
          - target_age_groups (list[str])
          - coverage (float in [0,1])  # target cumulative coverage over campaign window
          - rollout: {"shape": "flat"|"ramp", "ramp_up_days": int}
          - ve_sus (float in [0,1])  # vaccine efficacy against susceptibility
        Additional fields are ignored.

    Nk:
        Population sizes per age group, ordered consistently with `age_groups`.

    sim_length:
        Simulation length in days. Output will include t = 0..sim_length-1.

    age_groups:
        Age-group labels in the same order as Nk. Used to map campaign targets.

    dt:
        Time step for the simulation.

    Returns
    -------
    pd.DataFrame:
        Columns: ["t"] + age_groups
        Values: daily doses (float) allocated to each age group.
    """
    if sim_length < 1:
        raise ValueError("sim_length must be >= 1")
    if len(age_groups) == 0:
        raise ValueError("age_groups must be a non-empty list")
    Nk_arr = np.asarray(Nk, dtype=float)
    if Nk_arr.shape[0] != len(age_groups):
        print(Nk_arr.shape[0], len(age_groups))
        raise ValueError("Nk length must match age_groups length")

    # Compute simulation dates and count steps per day
    end_date = START_DATE + timedelta(days=sim_length)
    simulation_dates = compute_simulation_dates(START_DATE, end_date, dt=dt)
    day_counts = Counter(pd.Timestamp(d).date() for d in simulation_dates)
    repeat_counts = [day_counts.get((START_DATE + timedelta(days=i)).date(), 1) for i in range(sim_length + 1)]

    # Output array: (sim_length, n_age)
    doses = np.zeros((sim_length + 1, len(age_groups)), dtype=float)
    age_to_idx = {ag: i for i, ag in enumerate(age_groups)}

    for camp in campaigns or []:
        # --- Read required fields with reasonable defaults
        start = int(camp.get("start_day", 0))
        end = int(camp.get("end_day", -1))
        if end < start:
            continue  # skip invalid windows silently (or raise, if you prefer)

        cov = float(camp.get("coverage", 0.0))
        cov = max(0.0, min(1.0, cov))

        ve = float(camp.get("ve_sus", 1.0))
        ve = max(0.0, min(1.0, ve))

        target_ages = camp.get("target_age_groups", []) or []
        target_indices = [age_to_idx[a] for a in target_ages if a in age_to_idx]
        if not target_indices or cov <= 0.0:
            continue

        rollout = camp.get("rollout", {}) or {}
        shape = str(rollout.get("shape", "flat")).lower()
        ramp_up_days = int(rollout.get("ramp_up_days", 0) or 0)

        # --- Clip to simulation horizon
        start_clip = max(0, start)
        end_clip = min(sim_length - 1, end)
        if end_clip < start_clip:
            continue

        # Campaign duration in days (inclusive)
        D = (end_clip - start_clip) + 1
        if D <= 0:
            continue

        # --- Build time weights over the clipped window
        if shape == "ramp":
            r = max(1, min(ramp_up_days, D))
            # weights: days 1..r ramp linearly, then steady at 1
            w = np.ones(D, dtype=float)
            w[:r] = (np.arange(r, dtype=float) + 1.0) / float(r)
        else:
            # default: flat
            w = np.ones(D, dtype=float)

        w_sum = float(w.sum())
        if w_sum <= 0.0:
            continue

        # --- Allocate per age group
        # Total doses in each targeted age group over the campaign window:
        #   total_doses_age = coverage * Nk_age
        for j in target_indices:
            total = cov * Nk_arr[j] * ve
            if total <= 0.0:
                continue
            daily = (total / w_sum) * w  # length D
            doses[start_clip : end_clip + 1, j] += daily

    df = pd.DataFrame(doses, columns=age_groups)
    df[age_groups] = df[age_groups].astype(int)
    df["total"] = df[age_groups].sum(axis=1)
    df.insert(0, "t", np.arange(sim_length + 1, dtype=int))

    # Expand rows based on simulation dates (duplicate each day's row by its step count)
    df = df.loc[df.index.repeat(repeat_counts)].reset_index(drop=True)

    return df
