import streamlit as st
import pandas as pd


def invalidate_results():
    st.session_state["trajectories"] = None

@st.cache_data
def load_locations():
    return pd.read_csv(
        "https://raw.githubusercontent.com/epistorm/epydemix-data/refs/heads/main/locations.csv"
    ).location.unique()