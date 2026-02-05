# ui/setup_panel.py

import streamlit as st
import numpy as np
from ui.initial_conditions import render_initial_conditions
from ui.model_params import render_model_params
from ui.interventions import render_contact_interventions
from ui.scenarios import render_save_run_controls, render_saved_scenarios_list
from ui.vaccinations import render_vaccination_campaigns
from schemas import INITIAL_CONDITION_DEFAULTS, MODEL_PARAM_SCHEMAS
from state import reset_model_params_to_defaults, reset_initial_conditions_to_defaults, reset_workspace


def _on_model_change():
    m = st.session_state["selected_model"]
    reset_model_params_to_defaults(m, MODEL_PARAM_SCHEMAS)
    reset_initial_conditions_to_defaults(m, INITIAL_CONDITION_DEFAULTS)


def render_setup_panel(load_locations_fn, model_param_schemas):
    st.subheader("Setup")

    # Initialize workspace early if scenarios/results exist but workspace doesn't
    scenarios_exist = bool(st.session_state.get("scenarios"))
    results_exist = bool(st.session_state.get("results"))
    
    # Get current selections early (before any UI elements)
    st.session_state.setdefault("selected_model", "SEIR (Measles)")
    st.session_state.setdefault("selected_geography", load_locations_fn()[0])
    
    # Initialize workspace if needed BEFORE checking workspace_active
    if st.session_state.get("workspace") is None and (scenarios_exist or results_exist):
        st.session_state["workspace"] = {
            "model": st.session_state["selected_model"],
            "geography": st.session_state["selected_geography"]
        }
    
    # Now get the workspace state
    workspace = st.session_state.get("workspace")
    workspace_active = (workspace is not None)

    # Show workspace info if active
    if workspace_active:
        with st.container(border=True):
            st.markdown("**Workspace**")
            st.caption(
                f"{workspace['model']} · {workspace['geography']} — scenario comparison is restricted to this context. Start a new session to change the model or geography."
            )

            if st.button("Start new session", type="primary", use_container_width=True):
                reset_workspace()
                st.rerun()

    # Model & geography
    c1, c2 = st.columns(2)
    with c1:
        model = st.selectbox(
            "Model",
            options=["SEIR (Measles)", "SEIRS (Influenza)", "SEIHR (COVID-19)"],
            key="selected_model",
            on_change=_on_model_change,
            disabled=workspace_active,
            help="Select the model to use for the simulation.",
        )
    
    with c2:
        # load locations
        locations = load_locations_fn()
        geography = st.selectbox(
            "Geography",
            options=locations,
            key="selected_geography",
            help="Type to search within the list.",
            disabled=workspace_active,
        )

    # Simulation length and time step
    c1, c2 = st.columns(2)
    with c1:
        st.number_input(
            "Simulation length (days)",
            min_value=1,
            max_value=5000,
            value=250,
            step=10,
            help="Total duration of the simulation.",
            key="sim_length",
        )
    
    with c2:
        st.number_input(
            "$\Delta t$ (days)",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Time step for the simulation. Smaller values result in more accurate simulations but require more computational resources.",
            key="time_step",
        )

    # Scenario controls (use model/geography)
    render_save_run_controls(model, geography)

    st.caption("Change the settings below to create custom scenarios.")

    # Expanders
    with st.expander("Initial conditions", expanded=False):
        render_initial_conditions(model, INITIAL_CONDITION_DEFAULTS)

    with st.expander("Model parameters", expanded=False):
        render_model_params(model, model_param_schemas)

    with st.expander("Contact interventions", expanded=False):
        render_contact_interventions()

    with st.expander("Vaccination campaigns", expanded=False):
        render_vaccination_campaigns(model)

    render_saved_scenarios_list()

    return model, geography