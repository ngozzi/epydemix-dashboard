import streamlit as st
from constants import DEFAULT_AGE_GROUPS
from state import ensure_vax_state_defaults, ensure_vax_settings_defaults

ROLLOUT_SHAPES = ["flat", "ramp"]


def _validate_campaign(start_day: int, end_day: int, ramp_days: int, rollout: str) -> str | None:
    if end_day < start_day:
        return "End day must be greater than or equal to start day."

    if rollout == "ramp":
        window = end_day - start_day + 1
        if ramp_days < 1:
            return "Ramp-up days must be at least 1."
        if ramp_days > window:
            return "Ramp-up days must be ≤ campaign duration."

    return None


def _vax_summary(item: dict) -> str:
    ages = ",".join(item["target_age_groups"]) if item.get("target_age_groups") else "—"
    cov = int(round(item["coverage"] * 100))
    ve = int(round(item["ve_sus"] * 100))
    rollout = item.get("rollout", {}).get("shape", "flat")
    if rollout == "ramp":
        ru = item.get("rollout", {}).get("ramp_up_days", "?")
        rollout_str = f"ramp({ru}d)"
    else:
        rollout_str = "flat"

    comps = ",".join(item.get("target_compartments", ["S"]))
    name = item.get("name", "Campaign")
    return f"{name} · ages {ages} · cov {cov}% · VE {ve}% · day {item['start_day']}→{item['end_day']} · {rollout_str} · [{comps}]"


def render_vaccination_settings(compartments):
    ensure_vax_settings_defaults()

    st.checkbox("Advanced options", key="_vx_show_advanced")
    if st.session_state.get("_vx_show_advanced", False):
        st.multiselect(
            "Target compartments",
            options=compartments,
            key="vax_target_compartments",
            help="If you include E/I/R, some doses may be wasted depending on the epidemic state.",
        )

        # persist into a structured place (avoid relying only on widget key)
        st.session_state["vaccination_settings"]["target_compartments"] = list(
            st.session_state.get("vax_target_compartments", ["S"])
        )
        
        st.caption("These options apply to all vaccination campaigns in this scenario. Default assumption is S-only. Including E/R can make targeting more realistic (and typically less optimistic).")


def render_vaccination_campaigns(model: str, age_groups: list[str] | None = None) -> None:
    if model == "SEIR (Measles)":
        compartments = ["S", "E", "I", "R"]
    elif model == "SEIRS (Influenza)":
        compartments = ["S", "E", "I", "R"]
    elif model == "SEIHR (COVID-19)":
        compartments = ["S", "E", "I", "H", "R"]
    else:
        raise ValueError(f"Invalid model: {model}")

    age_groups = age_groups or DEFAULT_AGE_GROUPS
    ensure_vax_state_defaults(age_groups)
    
    st.caption(
        "Add one or more vaccination campaigns. Vaccination is modeled as all-or-nothing and is applied to susceptibility. "
        "Choose different coverage, effectiveness, rollout shapes and (optionally) which compartments are eligible."
    )

    # ---- Add new campaign card
    with st.container(border=True):

        r1 = st.columns([1.2, 0.9, 0.9], gap="small")
        with r1[0]:
            st.text_input("Name", key="_vx_new_name")
        with r1[1]:
            st.number_input("Start day", min_value=0, max_value=10_000, step=1, key="_vx_new_start")
        with r1[2]:
            st.number_input("End day", min_value=0, max_value=10_000, step=1, key="_vx_new_end")

        r2 = st.columns([1.1, 1.1], gap="small")
        with r2[0]:
            st.slider("Target coverage (%)", 0, 100, step=1, key="_vx_new_cov_pct")
        with r2[1]:
            st.slider("Vaccine efficacy (%)", 0, 100, step=1, key="_vx_new_ve_pct")

        r3 = st.columns([1.4, 0.8, 0.8], gap="small")
        with r3[0]:
            st.multiselect("Target age groups", options=age_groups, key="_vx_new_age_groups")
        with r3[1]:
            st.selectbox("Rollout", options=ROLLOUT_SHAPES, key="_vx_new_rollout")
        with r3[2]:
            # Only meaningful for ramp, but we can keep it visible; validation will handle.
            st.number_input("Ramp-up days", min_value=1, max_value=10_000, step=1, key="_vx_new_ramp_days")

        add = st.button("Add campaign", type="primary", use_container_width=True)

        if add:
            name = st.session_state["_vx_new_name"].strip() or "Campaign"
            start_day = int(st.session_state["_vx_new_start"])
            end_day = int(st.session_state["_vx_new_end"])
            cov = float(st.session_state["_vx_new_cov_pct"]) / 100.0
            ve = float(st.session_state["_vx_new_ve_pct"]) / 100.0
            rollout = st.session_state["_vx_new_rollout"]
            ramp_days = int(st.session_state["_vx_new_ramp_days"])
            target_ages = list(st.session_state.get("_vx_new_age_groups", []))

            if not target_ages:
                st.warning("Select at least one target age group.")
                return

            err = _validate_campaign(start_day, end_day, ramp_days, rollout)
            if err:
                st.warning(err)
                return

            item = {
                "name": name,
                "start_day": start_day,
                "end_day": end_day,
                "target_age_groups": target_ages,
                "coverage": cov,     # fraction [0,1]
                "ve_sus": ve,        # fraction [0,1]
                "rollout": {
                    "shape": rollout,  # "flat" | "ramp"
                    "ramp_up_days": ramp_days if rollout == "ramp" else 0,
                },
            }

            st.session_state["vaccination_campaigns"].append(item)
            #st.rerun()

    # ---- Vaccination settings
    render_vaccination_settings(compartments)

    # ---- Existing campaigns list
    items = st.session_state.get("vaccination_campaigns", [])
    st.markdown("**Added campaigns**")

    if not items:
        st.info("No vaccination campaigns added yet.")
        return

    remove_idx = None
    for idx, it in enumerate(items):
        with st.container(border=True):
            st.markdown(f"**{_vax_summary(it)}**")

            # Optional: view details inline without expander nesting
            if st.button("View details", key=f"vx_view_{idx}"):
                st.json(it)

    if remove_idx is not None:
        st.session_state["vaccination_campaigns"].pop(remove_idx)
        st.rerun()
