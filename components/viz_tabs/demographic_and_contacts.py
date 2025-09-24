import streamlit as st
import pandas as pd
import numpy as np
from visualizations.plots import plot_contact_matrix, plot_population
from utils.helpers import contact_matrix_df

def render_demographic_and_contacts_tab(
    population, 
    contact_matrices, 
    country_name, 
    sep_width=0.02
    ):
    """Render the population visualization tab."""

    # Two main columns: Population Donut | Contact Matrix
    left, sep, right = st.columns([1, sep_width, 1])

    # Population Panel
    with left:
        st.subheader("Population Distribution")
        st.markdown(f'<span style="color:#94a3b8;">'
                    f'<strong><span style="color:#60f0d8;">{country_name}</span></strong> population by age group'
                    f'</span>', unsafe_allow_html=True)

        # Population Data Download
        population_data = pd.DataFrame({
            "Age Group": population.Nk_names,
            "Count": population.Nk,
            "Percentage": 100 * population.Nk / population.Nk.sum()
        })
        st.download_button(
            label="Download Population Data (CSV)",
            data=population_data.to_csv(index=False),
            file_name=f"{country_name}_population.csv",
            mime="text/csv",
            use_container_width=False,
            help="Download population by age group (counts and percentages)"
        )

        # Population Metrics
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Population", f"{population.Nk.sum():,}")
        with col2: st.metric("Most Populous Age Group", f"{population.Nk_names[population.Nk.argmax()]}")
        with col3: st.metric("Least Populous Age Group", f"{population.Nk_names[population.Nk.argmin()]}")

        # Population Donut Plot
        plot_population(
            population=population,
            facecolor="#0c1019",
        )

    # Contact Matrix Panel
    with right:        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Contact Matrix")
            st.markdown(f'<span style="color:#94a3b8;">'
                        f'<strong><span style="color:#60f0d8;">{country_name}</span></strong> contact matrices'
                        f'</span>', unsafe_allow_html=True)      
        with col2:
            st.markdown("<div style='height:0px'></div>", unsafe_allow_html=True)
            layer = st.selectbox("Contact layer", 
                                options=["overall", "home", "school", "work", "community"], 
                                index=0, 
                                help="Choose which contact layer to visualize")

        # Contact Matrix Data Download
        contact_df = contact_matrix_df(layer, population.contact_matrices, population.Nk_names)
        st.download_button(
                label="Download Contact Matrix (CSV)",
                data=contact_df.round(3).to_csv(index=True),
                file_name=f"{country_name}_contacts_{layer}.csv",
                mime="text/csv",
                use_container_width=False,
                help="Download contact matrix data with age groups and contact rates"
            )

        # Contact Matrix Metrics
        if layer == "overall":
            matrix = sum(population.contact_matrices.values())
        else:
            matrix = population.contact_matrices[layer]
        matrix = np.array(matrix)

        col1, col2, col3, col4 = st.columns(4)    
        with col1: st.metric("Most Active Age Group", f"{population.Nk_names[matrix.sum(axis=1).argmax()]}")
        with col2: st.metric("Least Active Age Group", f"{population.Nk_names[matrix.sum(axis=1).argmin()]}")
        with col3: st.metric("Mean Contact Rate", f"{matrix.sum(axis=1).mean():.2f}")
        with col4: st.metric("Spectral Radius", f"{np.linalg.eigvals(matrix).max().real:.2f}")
        
        # Contact Matrix Plot
        plot_contact_matrix(
            layer=layer,
            matrices=contact_matrices,  
            groups=population.Nk_names,  
            facecolor="#0c1019",
            cmap="teals",
        )
