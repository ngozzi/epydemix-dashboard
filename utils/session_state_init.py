import streamlit as st
from utils.helpers import load_locations

def init_simulation_settings():
    if st.session_state.get("_initialized"):
        return

    st.session_state["model_choice"] = "SIR"
    st.session_state["country_name"] = load_locations()[0]
    st.session_state["model_loaded"] = False
    st.session_state["simulation_output"] = None
    st.session_state["model_config"] = None
    st.session_state["_initialized"] = True
    