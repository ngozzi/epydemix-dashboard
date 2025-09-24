import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import base64

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

def reset_all_state():
    """Full reset of the dashboard (clear *all* session state)."""
    st.session_state.clear()
    st.toast("🧼 Full reset: all settings restored to defaults.", icon="♻️")
    st.rerun()

def data_uri(path):
    p = Path(path)
    mime = "image/svg+xml" if p.suffix.lower()==".svg" else "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"