import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def render_summary_tab():
    """Render the simulation summary tab with configuration and results tables."""
    
    if "simulation_results" not in st.session_state:
        st.info("Run a simulation first to see the summary.")
        return
    
    st.subheader("📊 Model andSimulation Summary")
    
    # Get simulation data
    model_config = st.session_state.model_config
    country_name = st.session_state.country_name
    n_v = st.session_state.n_v
    simulation_days_v = st.session_state.simulation_days_v
    param_values = st.session_state.param_values
    interventions = st.session_state.get("interventions", {})
    parameter_overrides = st.session_state.get("parameter_overrides", {})
    
    # Basic settings
    base_rows = [
        ("Model", str(model_config.name)),
        ("Country", str(country_name)),
        ("Simulation days", str(simulation_days_v)),
        ("Simulations (N)", str(n_v)),
    ]
    
    # Add model parameters
    for param_name, param_spec in model_config.parameters.items():
        display_name = model_config.param_display_names.get(param_name, param_name)
        value = param_values.get(param_name, param_spec["default"])
        base_rows.append((display_name, f"{value:.2f}"))
    
    # Display basic configuration table
    st.dataframe(pd.DataFrame(base_rows, columns=["Setting", "Value"]), use_container_width=True)
    
    # Contact interventions table
    st.subheader("🤝 Contact Interventions Summary")
    if interventions:
        df_iv = pd.DataFrame([
            {
                "Layer": str(layer),
                "Start day": str(v["start"]),
                "End day": str(v["end"]),
                "Reduction (%)": str(int(round(100 * v["reduction"])))
            }
            for layer, v in interventions.items()
        ]).sort_values(["Start day", "Layer"])
        st.dataframe(df_iv, use_container_width=True)
    else:
        st.info("No contact interventions enabled.")
    
    # Parameter overrides table
    st.subheader("🦠 Parameter Overrides Summary")
    if parameter_overrides:
        rows = []
        for pname, spec in parameter_overrides.items():
            display_name = model_config.param_display_names[pname]
            rows.append({
                "Parameter": str(display_name),
                "Start day": str(spec["start_day"]),
                "End day": str(spec["end_day"]),
                "Override value": f"{spec['param']:.2f}"
            })
        
        df_ovr = pd.DataFrame(rows).sort_values(["Start day", "Parameter"])
        st.dataframe(df_ovr, use_container_width=True)
    else:
        st.info("No parameter overrides enabled.")
    
    # Notes
    st.caption(
        "Notes: Interventions scale contacts by layer in their active window. "
        "Parameter overrides adjust the value of the model parameters (e.g. R₀ or infectious period) only within their day range."
    )
