import streamlit as st

def render_simulation_settings_tab():
    st.subheader("🧩 Simulation Settings")

    # Number of simulations + length
    n_v = st.number_input("Number of simulations", 1, 100, 10, key="n_sims")
    simulation_days_v = st.number_input("Simulation Days", 1, 730, 250, key="sim_days")

    # Store results back in session state
    st.session_state.n_v = n_v
    st.session_state.simulation_days_v = simulation_days_v
