# pages/Vaccination_Planner.py
#
# Sub-screen for age-stratified vaccination:
#   1. Build "what-if" vaccination campaigns with a different target coverage per
#      age band (added to the scenario shared with the main dashboard).
#   2. Enter actual vaccination coverage by age band to set the model's starting
#      immunity landscape, so runs can be calibrated against real data.

import streamlit as st
import pandas as pd

from layout.header import show_dashboard_header
from layout.sidebar import render_sidebar
from layout.logos import show_logos
from constants import DEFAULT_AGE_GROUPS

st.set_page_config(
    page_title="Vaccination Planner",
    page_icon="assets/epydemix-icon.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

show_dashboard_header()
render_sidebar()

st.markdown("## Vaccination Planner — age-stratified")

model = st.session_state.get("selected_model", "SEIR (Measles)")
geography = st.session_state.get("selected_geography", "—")
st.caption(
    f"Current scenario context: **{model}** · **{geography}**. "
    "Edits here update the same scenario used on the main **Dashboard** page — "
    "switch back there and click **Run** to simulate."
)

st.session_state.setdefault("vaccination_campaigns", [])

# =============================================================================
# 1. Age-stratified vaccination campaign builder (what-if)
# =============================================================================
st.header("1 · Age-stratified campaign (what-if)")
st.caption(
    "Set a different target coverage per age band, then add them to the scenario. "
    "Each non-zero row becomes one vaccination campaign. Use this to explore "
    "'what-if' coverage strategies (e.g. boost adolescents, or drop infant coverage)."
)

strat_default = pd.DataFrame(
    {
        "Age group": DEFAULT_AGE_GROUPS,
        "Coverage %": [0.0] * len(DEFAULT_AGE_GROUPS),
        "Vaccine efficacy %": [80.0] * len(DEFAULT_AGE_GROUPS),
        "Start day": [0] * len(DEFAULT_AGE_GROUPS),
        "End day": [250] * len(DEFAULT_AGE_GROUPS),
    }
)

edited = st.data_editor(
    strat_default,
    key="strat_campaign_editor",
    hide_index=True,
    num_rows="fixed",
    use_container_width=True,
    column_config={
        "Age group": st.column_config.TextColumn("Age group", disabled=True),
        "Coverage %": st.column_config.NumberColumn("Coverage %", min_value=0.0, max_value=100.0, step=1.0),
        "Vaccine efficacy %": st.column_config.NumberColumn("Vaccine efficacy %", min_value=0.0, max_value=100.0, step=1.0),
        "Start day": st.column_config.NumberColumn("Start day", min_value=0, max_value=10000, step=1),
        "End day": st.column_config.NumberColumn("End day", min_value=0, max_value=10000, step=1),
    },
)

c1, c2, c3 = st.columns([1.4, 0.8, 0.8], gap="small")
with c1:
    name_prefix = st.text_input("Campaign name prefix", value="Stratified")
with c2:
    rollout = st.selectbox("Rollout", options=["flat", "ramp"])
with c3:
    ramp_days = st.number_input("Ramp-up days", min_value=1, max_value=10000, value=14, step=1)

b1, b2 = st.columns(2, gap="small")
with b1:
    if st.button("Add as campaigns", type="primary", use_container_width=True):
        added = 0
        for _, r in edited.iterrows():
            cov = float(r["Coverage %"]) / 100.0
            if cov <= 0:
                continue
            start_day = int(r["Start day"])
            end_day = int(r["End day"])
            if end_day < start_day:
                st.warning(f"{r['Age group']}: end day < start day — skipped.")
                continue
            st.session_state["vaccination_campaigns"].append({
                "name": f"{name_prefix}: {r['Age group']}",
                "start_day": start_day,
                "end_day": end_day,
                "target_age_groups": [r["Age group"]],
                "coverage": cov,
                "ve_sus": float(r["Vaccine efficacy %"]) / 100.0,
                "rollout": {"shape": rollout, "ramp_up_days": ramp_days if rollout == "ramp" else 0},
            })
            added += 1
        if added:
            st.success(f"Added {added} age-stratified campaign(s) to the scenario.")
        else:
            st.info("No rows with coverage > 0 to add.")
with b2:
    if st.button("Clear all campaigns", use_container_width=True):
        st.session_state["vaccination_campaigns"] = []
        st.success("Cleared all vaccination campaigns.")

# Current campaigns summary
camps = st.session_state.get("vaccination_campaigns", [])
st.markdown("**Campaigns currently in the scenario**")
if camps:
    summary_df = pd.DataFrame([
        {
            "Name": c.get("name", "Campaign"),
            "Age groups": ", ".join(c.get("target_age_groups", [])),
            "Coverage %": round(c.get("coverage", 0) * 100, 1),
            "VE %": round(c.get("ve_sus", 0) * 100, 1),
            "Start": c.get("start_day"),
            "End": c.get("end_day"),
            "Rollout": c.get("rollout", {}).get("shape", "flat"),
        }
        for c in camps
    ])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
else:
    st.info("No vaccination campaigns yet.")

st.divider()

# =============================================================================
# 2. Actual vaccination coverage by age (for calibration)
# =============================================================================
st.header("2 · Actual vaccination coverage by age (calibration)")
st.caption(
    "Enter your real vaccination / up-to-date coverage per age band. Applying it "
    "sets the model's starting immunity landscape by age (the partially-immune Sₚ "
    "pool), replacing the uniform 'Background immunity' slider — so a run can be "
    "calibrated against your actual data. Currently applies to the pertussis model."
)

existing = st.session_state.get("age_immunity_pct") or {}
cov_default = pd.DataFrame(
    {
        "Age group": DEFAULT_AGE_GROUPS,
        "% immune / up-to-date": [float(existing.get(ag, 0.0)) for ag in DEFAULT_AGE_GROUPS],
    }
)

cov_edited = st.data_editor(
    cov_default,
    key="age_immunity_editor",
    hide_index=True,
    num_rows="fixed",
    use_container_width=True,
    column_config={
        "Age group": st.column_config.TextColumn("Age group", disabled=True),
        "% immune / up-to-date": st.column_config.NumberColumn(
            "% immune / up-to-date", min_value=0.0, max_value=100.0, step=1.0,
            help="Share of this age band that is immune / up-to-date on vaccination.",
        ),
    },
)

active = bool(st.session_state.get("use_age_immunity", False))
st.caption(f"Age-stratified immunity is currently **{'ON' if active else 'OFF'}**.")

d1, d2 = st.columns(2, gap="small")
with d1:
    if st.button("Use as initial immunity", type="primary", use_container_width=True):
        st.session_state["age_immunity_pct"] = {
            r["Age group"]: float(r["% immune / up-to-date"]) for _, r in cov_edited.iterrows()
        }
        st.session_state["use_age_immunity"] = True
        st.success("Saved. The pertussis model will start from this age-stratified immunity. "
                   "Go to the Dashboard and click Run.")
with d2:
    if st.button("Revert to uniform slider", use_container_width=True):
        st.session_state["use_age_immunity"] = False
        st.success("Reverted. Runs will use the uniform 'Background immunity' slider again.")

st.divider()
show_logos()
