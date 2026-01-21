import streamlit as st
from helpers import data_uri

def show_dashboard_header():
    epyscenario = data_uri("assets/epyscenario.svg")
    epydemix = data_uri("assets/epydemix-icon.svg")
    st.markdown(f"""
                <style>
                .title-row {{ display:flex; align-items:center; gap:8px; }}
                .title-row img {{ height:115px; }}
                </style>
                <div class="title-row">
                <img src="{epyscenario}" alt="EpyScenario Dashboard">
                </div>
                """, unsafe_allow_html=True)
    st.caption("Build Scenarios. Run Simulations. Explore Insights.")