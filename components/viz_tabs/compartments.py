import streamlit as st
import pandas as pd
import numpy as np
from visualizations.plots import plot_compartments_traj
from utils.helpers import build_compartment_timeseries_df


def render_compartments_tab(simulation_results, population, model_config, country_name):
    """Render the compartments visualization tab."""

    # Get available compartments and age groups
    compartments = model_config.compartments

    # TODO: make this dynamic based on the model config
    age_groups = ["total", "0-4", "5-19", "20-49", "50-64", "65+"] 
    
    # Filter age groups based on available data
    available_age_groups = []
    for age in age_groups:
        if any(f"{comp}_{age}" in simulation_results.get_stacked_compartments() for comp in compartments):
            available_age_groups.append(age)
    
    if not available_age_groups:
        st.warning("No compartment data available.")
        return
    
    # Get trajectories
    trajectories = simulation_results.get_stacked_compartments()

    # Two columns: left (header and selection) | right (plot)
    col1, sep, col2 = st.columns([2.3, 0.02, 5])

    with col1:
        # Header
        st.subheader(f"Compartment Trajectories") 
        st.markdown(f'<span style="color:#94a3b8;">'
                    f'Simulation results for <strong><span style="color:#60f0d8;">{country_name}</span></strong> with <strong><span style="color:#60f0d8;">{model_config.name}</span></strong> model'
                    f'</span>', unsafe_allow_html=True)  
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # Compartment selection
        compartment = st.selectbox(
            "Select Compartment:",
            options=compartments,
            index=compartments.index("Infected") if "Infected" in compartments else 0,
            key="compartment_selector",
            help="Choose which compartment to visualize"
        )
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # Age group selection
        age_group = st.selectbox(
            "Select Age Group:",
            options=available_age_groups,
            index=0,
            key="age_group_selector",
            help="Choose which age group to visualize"
        )
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # Download button section
        export_df = build_compartment_timeseries_df(trajectories)
        if export_df is not None:
            st.download_button(
                label="Download Trajectory Data (CSV)",
                data=export_df.to_csv(index=False),
                file_name=f"{country_name}_{model_config.name}_trajectories.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download trajectory data with median and all simulation runs"
            )
        else:
            st.info("No data available for the selected series.")

    # Plot
    with col2:
        # Show median toggle
        show_median = st.toggle(
            "Show median trajectory",
            value=True,
            key="show_median_toggle",
            help="Toggle to show/hide the median trajectory line"
        )

        # Compartment trajectory plot
        plot_compartments_traj(
            trj=trajectories,
            comp=compartment,
            age=age_group,
            show_median=show_median,
            facecolor="#0c1019",
            linecolor="#50f0d8"
        )
    
    
    
