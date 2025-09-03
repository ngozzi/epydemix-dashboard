import streamlit as st
import pandas as pd
import numpy as np

def invalidate_results():
    st.session_state["trajectories"] = None

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


def build_compartment_timeseries_df(trj, comp, age):
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