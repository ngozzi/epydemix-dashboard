# pages/Vaccination_Planner.py
#
# Sub-screen for age-stratified vaccination:
#   1. Build "what-if" vaccination campaigns with a different target coverage per
#      age band (added to the scenario shared with the main dashboard).
#   2. Enter actual vaccination coverage by age band to set the model's starting
#      immunity landscape, so runs can be calibrated against real data.

import copy
import streamlit as st
import pandas as pd

from layout.header import show_dashboard_header
from layout.sidebar import render_sidebar
from layout.logos import show_logos
from constants import DEFAULT_AGE_GROUPS
from data.campaign_store import (
    load_saved_campaigns, save_campaign_set, delete_campaign_set, DEFAULT_SET_NAME,
)
from data.coverage_store import (
    load_saved_coverage, save_coverage_profile, delete_coverage_profile,
    DEFAULT_PROFILE_NAME as DEFAULT_COVERAGE_NAME,
)

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
    "'what-if' coverage strategies (e.g. boost adolescents, or drop infant coverage). "
    "The grid is pre-filled with the CDC DTaP/Tdap schedule as a starting point."
)

# Pre-fill the builder grid with the DTaP/Tdap schedule mapped onto the age bands
# (0-4 primary series 92%, 5-19 adolescent booster 90%, adults decennial ~7%).
_DTAP_COVERAGE = {"0-4": 92.0, "5-19": 90.0, "20-49": 7.0, "50-64": 7.0, "65+": 7.0}
_DTAP_VE = {"0-4": 80.0, "5-19": 80.0, "20-49": 70.0, "50-64": 70.0, "65+": 70.0}
strat_default = pd.DataFrame(
    {
        "Age group": DEFAULT_AGE_GROUPS,
        "Coverage %": [_DTAP_COVERAGE[ag] for ag in DEFAULT_AGE_GROUPS],
        "Vaccine efficacy %": [_DTAP_VE[ag] for ag in DEFAULT_AGE_GROUPS],
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

# ---- Saved campaign sets (persist between sessions via JSON) -----------------
st.subheader("Saved campaign sets (persist between sessions)")
st.caption(
    "Named campaign sets are stored on disk (data/saved_campaigns.json) and stay "
    "available across sessions. The **DTaP/Tdap (CDC)** set is provided by default. "
    "These sets are also loadable from the pertussis **Vaccination campaigns** menu "
    "on the main Dashboard."
)

saved = load_saved_campaigns()
set_names = list(saved.keys())

s1, s2, s3 = st.columns([1.6, 0.7, 0.7], gap="small")
with s1:
    sel_set = st.selectbox("Saved set", options=set_names, key="planner_saved_set")
with s2:
    st.markdown("<div style='height:1.8em'></div>", unsafe_allow_html=True)
    if st.button("Load into scenario", use_container_width=True, key="planner_load_set"):
        st.session_state["vaccination_campaigns"] = copy.deepcopy(saved.get(sel_set, []))
        st.success(f"Loaded '{sel_set}' into the scenario ({len(saved.get(sel_set, []))} campaigns).")
with s3:
    st.markdown("<div style='height:1.8em'></div>", unsafe_allow_html=True)
    disabled_del = (sel_set == DEFAULT_SET_NAME)
    if st.button("Delete set", use_container_width=True, key="planner_del_set", disabled=disabled_del,
                 help="The default DTaP/Tdap set cannot be deleted." if disabled_del else None):
        delete_campaign_set(sel_set)
        st.success(f"Deleted '{sel_set}'.")
        st.rerun()

sv1, sv2 = st.columns([1.6, 0.8], gap="small")
with sv1:
    save_name = st.text_input("Save current scenario campaigns as", value="My campaign set", key="planner_save_name")
with sv2:
    st.markdown("<div style='height:1.8em'></div>", unsafe_allow_html=True)
    if st.button("Save set", type="primary", use_container_width=True, key="planner_save_set"):
        name = (save_name or "").strip()
        current = st.session_state.get("vaccination_campaigns", [])
        if not name:
            st.warning("Enter a name for the campaign set.")
        elif not current:
            st.warning("There are no campaigns in the scenario to save.")
        else:
            save_campaign_set(name, current)
            st.success(f"Saved '{name}' ({len(current)} campaigns) — it will persist between sessions.")

st.divider()

# =============================================================================
# 2. Vaccination coverage by age (saved profiles)
# =============================================================================
st.header("2 · Vaccination coverage by age (saved profiles)")
st.caption(
    "Named coverage profiles map each age band to a percent immune / up-to-date. "
    "They are stored on disk (data/saved_coverage.json) and persist between sessions; "
    "the **Lane County (observed)** profile — derived from the same case data used in "
    "the calibration area (percent of cases up-to-date by age) — is provided by default. "
    "Applying a profile sets "
    "the model's starting immunity by age (the partially-immune Sₚ pool), overriding the "
    "uniform 'Background immunity' slider. Saved profiles are also selectable from the "
    "main **Dashboard** under Initial conditions."
)

cov_profiles = load_saved_coverage()
prof_names = list(cov_profiles.keys())

p1, p2 = st.columns([1.6, 0.7], gap="small")
with p1:
    sel_prof = st.selectbox("Coverage profile", options=prof_names, key="planner_cov_profile")
with p2:
    st.markdown("<div style='height:1.8em'></div>", unsafe_allow_html=True)
    disabled_pdel = (sel_prof == DEFAULT_COVERAGE_NAME)
    if st.button("Delete profile", use_container_width=True, key="planner_del_cov", disabled=disabled_pdel,
                 help="The default Oregon profile cannot be deleted." if disabled_pdel else None):
        delete_coverage_profile(sel_prof)
        st.success(f"Deleted '{sel_prof}'.")
        st.rerun()

# Grid pre-filled from the selected profile (keyed by profile so switching resets it)
prof_vals = cov_profiles.get(sel_prof, {})
cov_default = pd.DataFrame(
    {
        "Age group": DEFAULT_AGE_GROUPS,
        "% immune / up-to-date": [float(prof_vals.get(ag, 0.0)) for ag in DEFAULT_AGE_GROUPS],
    }
)
cov_edited = st.data_editor(
    cov_default,
    key=f"age_immunity_editor_{sel_prof}",
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
active_prof = st.session_state.get("age_immunity_profile")
st.caption(
    f"Age-stratified immunity is currently **{'ON' if active else 'OFF'}**"
    + (f" — profile: **{active_prof}**." if active and active_prof else ".")
)

a1, a2 = st.columns(2, gap="small")
with a1:
    if st.button("Apply to scenario", type="primary", use_container_width=True, key="cov_apply"):
        st.session_state["age_immunity_pct"] = {
            r["Age group"]: float(r["% immune / up-to-date"]) for _, r in cov_edited.iterrows()
        }
        st.session_state["use_age_immunity"] = True
        st.session_state["age_immunity_profile"] = sel_prof
        st.success("Applied. The pertussis model will start from this age-stratified immunity. "
                   "Go to the Dashboard and click Run.")
with a2:
    if st.button("Revert to uniform slider", use_container_width=True, key="cov_revert"):
        st.session_state["use_age_immunity"] = False
        st.success("Reverted. Runs use the uniform 'Background immunity' slider again.")

cs1, cs2 = st.columns([1.6, 0.8], gap="small")
with cs1:
    cov_save_name = st.text_input("Save current grid as profile", value=sel_prof, key="cov_save_name")
with cs2:
    st.markdown("<div style='height:1.8em'></div>", unsafe_allow_html=True)
    if st.button("Save profile", use_container_width=True, key="cov_save"):
        name = (cov_save_name or "").strip()
        if not name:
            st.warning("Enter a profile name.")
        else:
            mapping = {r["Age group"]: float(r["% immune / up-to-date"]) for _, r in cov_edited.iterrows()}
            save_coverage_profile(name, mapping)
            st.success(f"Saved profile '{name}' — persists between sessions and is selectable on the Dashboard.")
            st.rerun()

st.divider()
show_logos()
