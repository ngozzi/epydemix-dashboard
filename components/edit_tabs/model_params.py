import streamlit as st
from utils.config_engine import render_config_params

def render_model_params_tab():
    st.subheader("🧩 Model Parameters")

    # Render parameters dynamically from config
    param_values = render_config_params(st.session_state.model_config)
    st.session_state.param_values = param_values

    