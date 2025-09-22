import streamlit as st

def render_simulation_settings_tab():
    st.subheader("🧩 Simulation Settings")
    #with st.form("sim_settings_form"):
    days = st.number_input(
            "Simulation Days",
            min_value=1,
            value=st.session_state["sim_days"],
            key="sim_days_input",
        )
    st.session_state["sim_days"] = days

    n_sims = st.number_input(
            "Number of simulations",
            min_value=1,
            value=st.session_state["n_sims"],
            key="n_sims_input",
        )
    st.session_state["n_sims"] = n_sims
        #st.number_input(label="Number of simulations", min_value=1, max_value=100, key="n_sims", value=st.session_state["n_sims"])
        #st.number_input(label="Simulation Days", min_value=1, max_value=730, key="sim_days", value=st.session_state["sim_days"])

        #submitted = st.form_submit_button("Save settings")
        #if submitted:
        #    st.session_state["sim_days"] = days
        #    st.session_state["n_sims"] = n_sims
        #    st.success("Settings saved")