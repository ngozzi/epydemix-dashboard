import streamlit as st
import pandas as pd
from visualizations.plots import plot_population

def render_population_tab():
    """Render the population visualization tab."""
    
    if st.session_state.simulation_results is None:
        st.info("Run a simulation first to see population visualization.")
        return
    
    st.subheader("👥 Population Distribution")
    
    # Get population data
    population = st.session_state["population"]
    
    # Display population info
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Country", st.session_state.country_name)
    
    with col2:
        st.metric("Total Population", f"{population.Nk.sum():,}")
    
    # Create population data for CSV (needed for download)
    population_data = pd.DataFrame({
        "Age Group": population.Nk_names,
        "Count": population.Nk,
        "Percentage": 100 * population.Nk / population.Nk.sum()
    })

    # Toggle between counts and percentages
    show_percent = st.toggle(
        "Show as percentages", 
        value=False,
        help="Toggle between absolute counts and percentages of total population"
    )
    
    # Create the population plot
    plot_population(
        population=population,
        show_percent=show_percent,
        facecolor="#0c1019",
        linecolor="#50f0d8"
    )
    
    # Download button section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📊 Download Population Data (CSV)",
            data=population_data.to_csv(index=False),
            file_name=f"{st.session_state.country_name}_population.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download population data with age groups, counts, and percentages"
        )
