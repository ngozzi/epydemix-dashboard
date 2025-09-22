import streamlit as st

def init_simulation_settings():
    if st.session_state.get("_initialized"):
        return

    st.session_state["n_sims"] = 10
    st.session_state["sim_days"] = 250
    st.session_state["_initialized"] = True
    

def init_session_state():

    # Ensure defaults in session state
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "model_config" not in st.session_state:
        st.session_state.model_config = None
    if "country_name" not in st.session_state:
        st.session_state.country_name = None
    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = True
    if "simulation_results" not in st.session_state: 
        st.session_state.simulation_results = None
    if "population" not in st.session_state:
        st.session_state.population = None
    if "model" not in st.session_state:
        st.session_state.model = None
    if "simulation_dates" not in st.session_state:
        st.session_state.simulation_dates = None
        
    # Number of simulations + length
    #if "n_sims" not in st.session_state:
    #    st.session_state["n_sims"] = 10
    #if "sim_days" not in st.session_state:
    #    st.session_state["sim_days"] = 250

    ## UPDATE VALUES FROM SESSION STATE
    # Parameter values
    if "param_values" in st.session_state:
        for pname, val in st.session_state.param_values.items():
            st.session_state.setdefault(f"param_{pname}", val)

