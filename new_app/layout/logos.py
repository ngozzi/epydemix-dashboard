import streamlit as st
from helpers import data_uri


def show_logos():

    isi = data_uri("assets/isi-logo-white.svg")
    neu = data_uri("assets/neu-logo-white.svg")
    epistorm = data_uri("assets/epistorm-logo.png")

    st.markdown(f"""
            <style>
            .logo-row {{ display:flex; gap:40px; align-items:center; justify-content:flex-start; margin-top:16px; }}
            .logo-row img {{ height:36px; }}
            </style>
            <div class="logo-row">
            <a href="https://www.isi.it" target="_blank" rel="noopener"><img src="{isi}" alt="ISI"></a>
            <a href="https://www.northeastern.edu/" target="_blank" rel="noopener"><img src="{neu}" alt="NEU"></a>
            <a href="https://www.epistorm.org/" target="_blank" rel="noopener"><img src="{epistorm}" alt="Epistorm"></a>
            </div>
            """, unsafe_allow_html=True)