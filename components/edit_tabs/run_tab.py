import streamlit as st
from utils.simulation import build_run_config
import pandas as pd
from streamlit.components.v1 import html
from utils.simulation import run_simulation
from datetime import datetime
import json

start_date = datetime(2024, 1, 1)


def switch(tab):
        return f"""
    var tabGroup = window.parent.document.getElementsByClassName("stTabs")[0]
    var tab = tabGroup.getElementsByTagName("button")
    tab[{tab}].click()
    """

def render_run_tab():
    st.subheader("Review & Run")

    # Retrieve run configuration
    run_cfg = build_run_config()
    payload = json.dumps(run_cfg, indent=2, default=str)
    
    # Buttons
    c1, c2, _ = st.columns([0.2, 0.25, 0.6])
    with c1: 
        run = st.button("Run Simulations", use_container_width=True)
        if run:
            results = run_simulation(run_cfg, start_date=start_date,)
            html(f"<script>{switch(5)}</script>", height=0)
            st.session_state["simulation_output"] = results
            
    with c2: 
        download_config = st.download_button(
            label="Download Configuration (JSON)", 
            data=payload,
            file_name="run_config.json",
            mime="application/json",
            use_container_width=True,
        )
    
    # Display run configuration summary
    display_run_config_summary(run_cfg)
    

def display_run_config_summary(run_cfg):
    # Display run configuration
    ## Basic settings
    base_rows = [
        ("Model", str(run_cfg["model_config"].name)),
        ("Country", str(run_cfg["country_name"])),
        ("Simulation days", str(run_cfg["sim_days"])),
        ("Simulations (N)", str(run_cfg["n_sims"])),
    ]

    ## Add model parameters
    for param_name, param_spec in run_cfg["model_config"].parameters.items():
        display_name = run_cfg["model_config"].param_display_names.get(param_name, param_name)
        value = run_cfg["param_values"].get(param_name, param_spec["default"])
        base_rows.append((display_name, f"{value:.2f}"))

    col1, sep, col2, sep, col3 = st.columns([1, 0.02, 1, 0.02, 1])
    with col1:
        # Display basic configuration table
        st.markdown("##### Model and Simulation Summary")
        st.dataframe(pd.DataFrame(base_rows, columns=["Setting", "Value"]), use_container_width=True, hide_index=True)

    with col2:
        # Contact interventions table
        st.markdown("##### Contact Interventions Summary")
        if run_cfg["interventions"]:
            df_iv = pd.DataFrame([
                {
                    "Layer": str(layer),
                    "Start day": str(v["start"]),
                    "End day": str(v["end"]),
                    "Reduction (%)": str(int(round(100 * v["reduction"])))
                }
                for layer, v in run_cfg["interventions"].items()
            ]).sort_values(["Start day", "Layer"])
            st.dataframe(df_iv, use_container_width=True, hide_index=True)
        else:
            st.info("No contact interventions enabled.")

    with col3:
        # Vaccinations table
        st.markdown("##### Vaccinations Summary")
        vaccinations = st.session_state.get("vaccinations", [])
        if vaccinations:
            rows = []
            for it in vaccinations:
                rows.append({
                    "Start day": str(it.get("start", "")),
                    "End day": str(it.get("end", "")),
                    "Coverage (%)": str(int(round(100 * float(it.get("coverage", 0.0))))),
                    "Effectiveness (%)": str(int(round(100 * float(it.get("effectiveness", 0.0)))))
                })
            df_vx = pd.DataFrame(rows).sort_values(["Start day"])
            st.dataframe(df_vx, use_container_width=True, hide_index=True)
        else:
            st.info("No vaccination campaigns configured.")