import streamlit as st
from utils.helpers import data_uri

def show_welcome_card():
    st.markdown(
        """
        <style>
        .welcome-card{
            background: linear-gradient(180deg, rgba(80,240,216,0.08), rgba(80,240,216,0.03));
            border: 1px solid rgba(80,240,216,0.35);
            border-radius: 14px;
            padding: 18px 18px 10px 18px;
            margin: 8px 0 16px 0;
        }
        .pill{
            display:inline-block; padding:2px 8px; margin-right:6px;
            border-radius:999px; font-size:12px; background:#152031; color:#dffdfa;
            border:1px solid rgba(80,240,216,0.35);
        }
        .muted{color:#cbd5e1; font-size:14px;}
        .steps li{margin-bottom:4px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="welcome-card">
            <h3>👋 Welcome to the <strong>Epydemix Simulation Dashboard</strong></h3>
            <p class="muted">
            Run stochastic SIR/SEIR/SIS simulations using country-specific population and contact matrices.
            Configure model & interventions in the sidebar, then visualize epidemic trajectories and summary stats.
            </p>
            <div>
            <span class="pill">SIR / SEIR / SIS</span>
            <span class="pill">Contact interventions</span>
            <span class="pill">Parameter overrides</span>
            <span class="pill">CSV exports</span>
            </div>
            <br/>
            <ul class="steps">
            <li><strong>Set parameters</strong> in the sidebar (model, epidemiological parameters, initial conditions).</li>
            <li><strong>Add interventions</strong> by layer (home/school/work/community) with start/end contact reduction.</li>
            <li><strong>Optionally override</strong> epidemiological parameters between two days.</li>
            <li>Click <strong>Run Simulation</strong>, then explore tabs for plots & tables.</li>
            </ul>
            <p class="muted">
            Need help? See <a href="https://epydemix.org" target="_blank">epydemix.org</a>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_logos():

    isi = data_uri("assets/isi-logo-white.svg")
    neu = data_uri("assets/neu-logo-white.svg")

    st.markdown(f"""
            <style>
            .logo-row {{ display:flex; gap:40px; align-items:center; justify-content:flex-start; margin-top:16px; }}
            .logo-row img {{ height:36px; }}
            </style>
            <div class="logo-row">
            <a href="https://www.isi.it" target="_blank" rel="noopener"><img src="{isi}" alt="ISI"></a>
            <a href="https://www.northeastern.edu/" target="_blank" rel="noopener"><img src="{neu}" alt="NEU"></a>
            </div>
            """, unsafe_allow_html=True)