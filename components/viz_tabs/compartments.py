import streamlit as st
import pandas as pd
import numpy as np
from visualizations.plots import plot_compartments_traj
from utils.helpers import build_compartment_timeseries_df
from utils.stats import compute_attack_rate, compute_peak_size, compute_peak_time, compute_endemic_state

def render_compartments_tab():
    """Render the compartments visualization tab."""
    
    if st.session_state.simulation_results is None:
        st.info("Run a simulation first to see compartment trajectories.")
        return
    
    st.subheader("🧩 Compartment Trajectories")
    
    # Get simulation data
    results = st.session_state["simulation_results"]
    population = st.session_state["population"]
    model_config = st.session_state.model_config
    
    # Display simulation info
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Country", st.session_state.country_name)
    
    with col2:
        st.metric("Model Type", model_config.name)
    
    # Get available compartments and age groups
    compartments = model_config.compartments
    age_groups = ["total", "0-4", "5-19", "20-49", "50-64", "65+"]
    
    # Filter age groups based on available data
    available_age_groups = []
    for age in age_groups:
        if any(f"{comp}_{age}" in results.get_stacked_compartments() for comp in compartments):
            available_age_groups.append(age)
    
    if not available_age_groups:
        st.warning("No compartment data available.")
        return
    
    # Compartment and age group selection
    col1, col2 = st.columns(2)
    
    with col1:
        compartment = st.selectbox(
            "Select Compartment:",
            options=compartments,
            index=compartments.index("Infected") if "Infected" in compartments else 0,
            key="compartment_selector",
            help="Choose which compartment to visualize"
        )
    
    with col2:
        age_group = st.selectbox(
            "Select Age Group:",
            options=available_age_groups,
            index=0,
            key="age_group_selector",
            help="Choose which age group to visualize"
        )
    
    # Show median toggle
    show_median = st.toggle(
        "Show median trajectory",
        value=True,
        key="show_median_toggle",
        help="Toggle to show/hide the median trajectory line"
    )
    
    # Create the compartment trajectory plot
    trajectories = results.get_stacked_compartments()
    plot_compartments_traj(
        trj=trajectories,
        comp=compartment,
        age=age_group,
        show_median=show_median,
        facecolor="#0c1019",
        linecolor="#50f0d8"
    )
    
    # Download button section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Create trajectory data for CSV
        export_df = build_compartment_timeseries_df(trajectories, compartment, age_group)
        
        if export_df is not None:
            st.download_button(
                label="📊 Download Trajectory Data (CSV)",
                data=export_df.to_csv(index=False),
                file_name=f"{st.session_state.country_name}_{compartment}_{age_group}_trajectories.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download trajectory data with median and all simulation runs"
            )
        else:
            st.info("No data available for the selected series.")
    