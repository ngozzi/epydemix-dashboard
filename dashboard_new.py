import streamlit as st
from components.welcome_card import show_welcome_card, show_logos
from components.sidebar import render_sidebar
from components.header import show_dashboard_header
from utils.session_state_init import init_simulation_settings
from components.edit_tabs.model_location import render_model_location_tab
from components.edit_tabs.model_params import render_model_params_tab
from components.edit_tabs.initial_conditions import render_initial_conditions_tab
from components.edit_tabs.contact_interventions import render_contact_interventions_tab
from components.edit_tabs.parameter_overrides import render_parameter_overrides_tab
from components.edit_tabs.run_tab import render_run_tab
from components.edit_tabs.simulation_settings import render_simulation_settings_tab
from components.viz_tabs.compartments import render_compartments_tab
from components.viz_tabs.demographic_and_contacts import render_demographic_and_contacts_tab
#from components.viz_tabs.interventions import render_interventions_tab
from utils.simulation import build_run_config
    
# Set page config
st.set_page_config(page_title="Epydemix Dashboard", layout="wide", initial_sidebar_state="collapsed")

def main():

    # Initialize session state
    init_simulation_settings()
    
    # Render dashboard header
    show_dashboard_header()

    # Render sidebar
    with st.sidebar:
        render_sidebar()
    
    choice = st.segmented_control("Navigate", ["About", "Dashboard"], key="nav", default="About", label_visibility="collapsed")
    # Render welcome card
    if choice == "About":
        show_welcome_card()
        show_logos()

    # Render dashboard
    else:
        tabs = st.tabs(["Model & Geography", "Model and Simulation Settings", "Contact Interventions", "Parameter Overrides", "Run", "Results"])

        # Render model & location tab
        with tabs[0]:
            render_model_location_tab()

        # Render parameters tab
        with tabs[1]:
            if st.session_state.model_config is None:
                st.info("Please select a model and geography first")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    render_model_params_tab()
                with col2:
                    render_initial_conditions_tab()
                with col3:
                    render_simulation_settings_tab()
    

        with tabs[2]:
            if st.session_state.model_config is None:
                st.info("Please select a model and geography first")
            else:
                render_contact_interventions_tab()

        with tabs[3]:
            if st.session_state.model_config is None:
                st.info("Please select a model and geography first")
            else:
                render_parameter_overrides_tab(st.session_state.model_config)

        with tabs[4]:
            if st.session_state.model_config is None:
                st.info("Please select a model and geography first")
            else:
                render_run_tab()

        with tabs[5]:
            if st.session_state.simulation_output is None:
                st.info("Please run a simulation first")
            else:
                tab_options = ["Compartments", "Demographic and Contacts", "Interventions"]
                selected_tab = st.radio(
                    "Visualization Tabs",
                    options=tab_options,
                    key="tab_radio_buttons",
                    horizontal=True,
                    label_visibility="collapsed"
                )

                # Render content based on selected tab
                if selected_tab == "Compartments":
                    render_compartments_tab(
                        st.session_state["simulation_output"]["simulation_results"], 
                        st.session_state["simulation_output"]["population"], 
                        st.session_state.model_config, 
                        st.session_state.country_name,
                        )

                elif selected_tab == "Demographic and Contacts":
                    render_demographic_and_contacts_tab(
                        st.session_state["simulation_output"]["population"], 
                        st.session_state["simulation_output"]["population"].contact_matrices,
                        st.session_state.country_name,     
                        )

                elif selected_tab == "Interventions":
                    st.subheader("Interventions")
                    st.info("Intervention plots will appear here.")
                
                
if __name__ == "__main__":
    main()


