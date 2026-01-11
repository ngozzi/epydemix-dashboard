import streamlit as st
from helpers import data_uri

def show_dashboard_header():
    epydemix = data_uri("assets/epydemix-icon.svg")
    st.markdown(f"""
                <style>
                .title-row {{ display:flex; align-items:center; gap:8px; }}
                .title-row img {{ height:58px; }}
                .title-row h1 {{ margin:0; font-weight:700; }}
                </style>
                <div class="title-row">
                <img src="{epydemix}" alt="Epydemix">
                <h1><span style="color:#FFAE42;">EpyScenario</span> Dashboard</h1>
                </div>
                """, unsafe_allow_html=True)
    st.caption("Build Scenarios. Run Simulations. Explore Insights.")