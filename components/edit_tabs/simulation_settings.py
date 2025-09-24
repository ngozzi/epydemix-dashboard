import streamlit as st

def render_simulation_settings_tab():
    st.subheader("Simulation Settings")
    with st.expander("Define the simulation settings.", expanded=True):
        
        days_ui = st.number_input(
                "Simulation Days",
                min_value=1,
                value=st.session_state.get("sim_days", 120),
                key="sim_days_input",
            )
        st.session_state["sim_days"] = days_ui

        n_sims_ui = st.number_input(
                "Number of simulations",
                min_value=1,
                value=st.session_state.get("n_sims", 10),
                key="n_sims_ui",
            )
        st.session_state["n_sims"] = n_sims_ui