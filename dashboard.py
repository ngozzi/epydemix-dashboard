import streamlit as st
from epydemix.model import load_predefined_model
from epydemix.population import load_epydemix_population
from epydemix.utils import compute_simulation_dates
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from visualizations.plots import plot_contact_intensity, plot_population, plot_compartments_traj, plot_contact_matrix
from utils.helpers import invalidate_results, load_locations, contact_matrix_df, build_compartment_timeseries_df, reset_all_state
from utils.stats import compute_attack_rate, compute_peak_size, compute_peak_time, compute_endemic_state
from utils.config_engine import (
    load_model_config_from_file, load_model_config_from_json_bytes,
    compute_spectral_radius, eval_derived,
    build_epimodel_from_config, compute_override_value
)
from utils.simulation import run_simulation
from components.welcome_card import show_welcome_card
from components.layout import show_sidebar_logo, show_fixed_logo
from streamlit_js_eval import streamlit_js_eval
from components.sidebar import render_sidebar
from components.edit_tabs.model_params import render_model_params_tab
from components.edit_tabs.contact_interventions import render_contact_interventions_tab
from components.edit_tabs.parameter_overrides import render_parameter_overrides_tab
from components.edit_tabs.initial_conditions import render_initial_conditions_tab
from components.edit_tabs.simulation_settings import render_simulation_settings_tab
from components.viz_tabs.summary import render_summary_tab
from components.viz_tabs.population import render_population_tab
from components.viz_tabs.contacts import render_contacts_tab
from components.viz_tabs.compartments import render_compartments_tab


# ---------- LAYOUT ----------
show_sidebar_logo()
show_fixed_logo()

# ---------- CONSTANTS ----------
start_date = datetime(2024, 1, 1)
LAYER_NAMES = ["home", "school", "work", "community"]
facecolor="#0c1019"
BUILTINS = {"SEIR": "models/seir.json", "SIR": "models/sir.json", "SIS": "models/sis.json"}

# ---------- SIDEBAR ----------
render_sidebar()

# ---------- MAIN ----------
st.title("Epydemix Simulation Dashboard")

# Show welcome card only if model not yet loaded
if not st.session_state.model_loaded:
    ## TODO: update welcome card
    show_welcome_card()

else:
    # Make sure page scrolls back to top after rerun
    st.markdown(
        """
        <script>
        window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True,
    )

    # --- Run Simulations button ---
    if st.button("🚀 Run Simulations"):
        st.session_state.edit_mode = False
        run_simulation(start_date)
    
    # --- Editing Mode toggle ---
    edit_mode_toggle = st.toggle("✏️ Editing Mode", key="edit_mode", value=True, help="Switch to editing mode to modify the model and the simulation settings.")

    if edit_mode_toggle:
        st.write("Navigate through the tabs to edit the model and the simulation settings. In case of changes, click the **Run Simulations** button to run the model with the new settings.")
        # --- Editing mode ---
        tabs = st.tabs(["Model Parameters", "Initial Conditions", "Simulation Settings", "Contact Interventions", "Parameter Overrides"])
        
        # --- Tab 1: Model Parameters ---
        with tabs[0]:
            render_model_params_tab()

        # --- Tab 2: Initial Conditions ---
        with tabs[1]:
            render_initial_conditions_tab()
        
        # --- Tab 3: Simulation Settings ---
        with tabs[2]:
            render_simulation_settings_tab()

        # --- Tab 3: Contact Interventions ---
        with tabs[3]:
            render_contact_interventions_tab()

        # --- Tab 4: Parameter Overrides ---
        with tabs[4]:
            render_parameter_overrides_tab(st.session_state.model_config)

    else:

        # ---- Visualization mode ----
        # Tab selection with radio buttons (fixed state handling)
        tab_options = ["Summary", "Compartments", "Population", "Contact Matrices", "Interventions"]
        selected_tab = st.radio(
            "Visualization Tabs",
            options=tab_options,
            key="tab_radio_buttons",
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # Render content based on selected tab
        if selected_tab == "Summary":
            render_summary_tab()
        elif selected_tab == "Compartments":
            render_compartments_tab()
        elif selected_tab == "Population":
            render_population_tab()
        elif selected_tab == "Contact Matrices":
            render_contacts_tab()
        elif selected_tab == "Interventions":
            st.subheader("🤝 Interventions")
            st.info("Intervention plots will appear here.")
