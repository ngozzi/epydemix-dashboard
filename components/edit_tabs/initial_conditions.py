import streamlit as st

def render_initial_conditions_tab(): 
    st.subheader("🧩 Initial Conditions")

    # Initial conditions
    ## TODO: add possibility to switch between percentages and absolute numbers
    st.markdown("Define the starting distribution of the population across compartments (as percentages).")
    compartments = st.session_state.model_config.compartments

    total_pct = 0
    initial_conditions = {}
    init_defaults = st.session_state.model_config.initial_conditions
    for comp in compartments:
        val = st.number_input(
            f"Initial {comp} (%)",
            min_value=0.0, 
            max_value=100.0, 
            step=0.1,
            value=init_defaults.get(comp, 0.0)
        )
        initial_conditions[comp] = val
        total_pct += val

    # Validation
    if abs(total_pct - 100) > 1e-6:
        st.error(f"Percentages must sum to 100%. Current total: {total_pct:.2f}%")
    else:
        st.success("✅ Initial conditions valid.")
    st.session_state.initial_conditions = initial_conditions
