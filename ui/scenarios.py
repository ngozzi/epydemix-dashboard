# ui/scenarios.py

import streamlit as st
from state import ensure_results_defaults, ensure_scenario_state_defaults, save_current_scenario, run_current_scenario
from engine.run import run_scenario

def render_save_run_controls(model: str, geography: str) -> None:
    """
    Renders:
    - Scenario name input
    - Save / Run buttons
    """
    ensure_scenario_state_defaults()

    st.text_input("Scenario name", key="scenario_name")
    if st.button("Run", type="primary", use_container_width=True):
        if st.session_state.get("workspace") is None:
            st.session_state["workspace"] = {"model": model, "geography": geography}
        sid = run_current_scenario(model, geography)
        st.rerun()


def render_saved_scenarios_list() -> None:
    """
    Snapshot list with explicit actions (no implied edit).
    """
    ensure_scenario_state_defaults()

    scenarios = st.session_state.get("scenarios", {})
    if not scenarios:
        st.info("No saved scenarios yet.")
        return

    st.markdown("**Saved scenarios**")
    st.caption("Click 'View' to view its details or 'Remove' to delete it.")

    remove_id = None

    for sid, item in scenarios.items():
        name = item.get("name", sid)

        with st.container(border=True):
            row = st.columns([0.35, 0.2, 0.2], gap="small")

            with row[0]:
                st.markdown(f"**{name}**")
                st.caption(f"ID: {sid}")

            with row[1]:
                view = st.button("View", key=f"sc_view_{sid}", use_container_width=True)

            with row[2]:
                if st.button("Remove", key=f"sc_rm_{sid}", use_container_width=True):
                    remove_id = sid

            if view:
                st.json(item.get("config", {}))

    if remove_id is not None:
        st.session_state["scenarios"].pop(remove_id, None)

        if st.session_state.get("active_scenario_id") == remove_id:
            st.session_state["active_scenario_id"] = None
        if st.session_state.get("last_run_scenario_id") == remove_id:
            st.session_state["last_run_scenario_id"] = None

        st.rerun()
