import streamlit as st
from helpers import data_uri


def show_logos():

    isi = data_uri("assets/isi-logo-white.svg")
    neu = data_uri("assets/neu-logo-white.svg")
    epistorm = data_uri("assets/epistorm-logo.png")
    irs = data_uri("assets/irs-logo.jpg")

    st.markdown(f"""
            <style>
            .logo-row {{ display:flex; gap:40px; align-items:center; justify-content:flex-start; margin-top:16px; flex-wrap:wrap; }}
            .logo-row img {{ height:36px; }}
            /* IRS logo is dark-on-white; put it on a white chip so it reads on the dark theme */
            .logo-row .irs-chip {{ background:#ffffff; border-radius:8px; padding:6px 12px; display:inline-flex; align-items:center; }}
            .logo-row .irs-chip img {{ height:30px; }}
            </style>
            <div class="logo-row">
            <a href="https://www.isi.it" target="_blank" rel="noopener"><img src="{isi}" alt="ISI"></a>
            <a href="https://www.northeastern.edu/" target="_blank" rel="noopener"><img src="{neu}" alt="NEU"></a>
            <a href="https://www.epistorm.org/" target="_blank" rel="noopener"><img src="{epistorm}" alt="Epistorm"></a>
            <a href="https://www.internationalrespondersystems.com" target="_blank" rel="noopener" class="irs-chip"><img src="{irs}" alt="International Responder Systems"></a>
            </div>
            """, unsafe_allow_html=True)