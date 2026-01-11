# ui/interventions.py

import streamlit as st
from constants import LAYER_NAMES
from state import ensure_contact_interventions_defaults


def _validate_window(start_day: int, end_day: int) -> bool:
    return end_day >= start_day


def _contact_summary(it: dict) -> str:
    return f"{it['layer']} · {int(it['reduction_pct'])}% · day {it['start_day']} → {it['end_day']}"


def render_contact_interventions() -> None:
    ensure_contact_interventions_defaults()

    st.caption("Reduce contacts by a given percentage between two days, within a contact layer.")

    with st.container(border=True):

        c1, c2, c3, c4 = st.columns([1.25, 1, 1, 1.4], gap="small")
        with c1:
            layer = st.selectbox("Layer", options=LAYER_NAMES, key="_ci_new_layer")
        with c2:
            start_day = st.number_input("Start day", min_value=0, max_value=10_000, step=1, key="_ci_new_start")
        with c3:
            end_day = st.number_input("End day", min_value=0, max_value=10_000, step=1, key="_ci_new_end")
        with c4:
            red_pct = st.slider("Reduction (%)", min_value=0, max_value=100, step=1, key="_ci_new_red_pct")

        add = st.button("Add", type="primary", use_container_width=True)

        if add:
            start_v = int(start_day)
            end_v = int(end_day)

            if not _validate_window(start_v, end_v):
                st.warning("End day must be greater than or equal to start day.")
            else:
                st.session_state["contact_interventions"].append(
                    {
                        "layer": layer,
                        "start_day": start_v,
                        "end_day": end_v,
                        "reduction_pct": int(red_pct),
                    }
                )
                st.rerun()

    items = st.session_state.get("contact_interventions", [])

    st.markdown("**Added interventions**")
    if not items:
        st.info("No contact interventions added yet.")
        return

    remove_idx = None
    for idx, it in enumerate(items):
        with st.container(border=True):
            header = st.columns([0.65, 0.35])
            with header[0]:
                st.markdown(f"**{_contact_summary(it)}**")
            with header[1]:
                if st.button("Remove", key=f"ci_remove_{idx}", use_container_width=True):
                    remove_idx = idx

    if remove_idx is not None:
        st.session_state["contact_interventions"].pop(remove_idx)
        st.rerun()
