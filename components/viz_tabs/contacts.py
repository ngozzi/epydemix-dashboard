import streamlit as st
import pandas as pd
import numpy as np
from visualizations.plots import plot_contact_matrix
from utils.helpers import contact_matrix_df

# Contact layer selection
## TODO: make this dynamic based on the model config
LAYER_NAMES = ["home", "school", "work", "community"]
contact_options = ["overall"] + LAYER_NAMES

def render_contacts_tab():
    """Render the contacts visualization tab."""
    
    if st.session_state.simulation_results is None:
        st.info("Run a simulation first to see contact matrices.")
        return
    
    st.subheader("🔗 Contact Matrices")
    
    # Get population data
    population = st.session_state["population"]
    
    # Display contact matrix info
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Country", st.session_state.country_name)
    
    with col2:
        selected_contact = st.selectbox(
            "Select Contact Layer:",
            options=contact_options,
            index=0,
            key="contact_layer_selector",
            help="Choose which contact layer to visualize"
        )   
        
    # Calculate some basic statistics
    if selected_contact == "overall":
        matrix = sum(population.contact_matrices.values())
    else:
        matrix = population.contact_matrices[selected_contact]
    
    # Convert to numpy array for calculations
    matrix = np.array(matrix)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # find the most active age group (highest sum of row)
        most_active_age_group = matrix.sum(axis=1).argmax()
        st.metric("Most Active Age Group", f"{population.Nk_names[most_active_age_group]}")
    
    with col2:
        # find the least active age group (lowest sum of row)
        least_active_age_group = matrix.sum(axis=1).argmin()
        st.metric("Least Active Age Group", f"{population.Nk_names[least_active_age_group]}")

    with col3:
        # find the mean contact rate (mean of row sums)
        st.metric("Mean Contact Rate", f"{matrix.sum(axis=1).mean():.2f}")
    
    with col4:
        # find the spectral radius (max of eigenvalues)
        st.metric("Spectral Radius", f"{np.linalg.eigvals(matrix).max().real:.2f}")

    # Create the contact matrix plot
    plot_contact_matrix(
        layer=selected_contact,
        matrices=population.contact_matrices,
        groups=population.Nk_names,
        facecolor="#0c1019",
        cmap="teals"
    )

    # Download button section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Create contact matrix data for CSV
        contact_df = contact_matrix_df(selected_contact, population.contact_matrices, population.Nk_names)
        
        st.download_button(
            label="📊 Download Contact Matrix (CSV)",
            data=contact_df.round(3).to_csv(index=True),
            file_name=f"{st.session_state.country_name}_contacts_{selected_contact}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download contact matrix data with age groups and contact rates"
        )
    