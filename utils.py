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