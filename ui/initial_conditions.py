# ui/initial_conditions.py

import streamlit as st
from state import ensure_initial_conditions_defaults


def render_initial_conditions(model: str, ic_defaults: dict) -> None:
    ensure_initial_conditions_defaults(model, ic_defaults)

    st.caption("Specify initial infected and background immunity (percent of population).")

    with st.container(border=True):
        c1, c2 = st.columns(2, gap="small")

        with c1:
            infected_pct = st.number_input(
                "Initial infected (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state["initial_conditions"]["infected_pct"]),
                step=0.1,
            )

        with c2:
            immune_pct = st.number_input(
                "Background immunity (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state["initial_conditions"]["immune_pct"]),
                step=0.1,
            )

        if infected_pct + immune_pct > 100.0:
            st.error("Initial infected + background immunity must be ≤ 100%.")
            return

        st.session_state["initial_conditions"]["infected_pct"] = float(infected_pct)
        st.session_state["initial_conditions"]["immune_pct"] = float(immune_pct)
