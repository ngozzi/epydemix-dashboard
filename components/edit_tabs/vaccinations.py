import streamlit as st
from typing import List, Dict
from epydemix.population import load_epydemix_population


def _ensure_defaults(cfg) -> None:
    st.session_state.setdefault("vaccinations_list", [])
    st.session_state.setdefault("_vx_name", "")
    st.session_state.setdefault("_vx_start", 0)
    st.session_state.setdefault("_vx_end", 250)
    st.session_state.setdefault("_vx_coverage", 0)
    st.session_state.setdefault("_vx_effectiveness", 0)
    st.session_state.setdefault("_vx_age_targets", None)  # list of selected age group names
    # per-compartment checkbox state containers
    for comp in cfg.compartments:
        st.session_state.setdefault(f"_vx_can_{comp}", False)
        st.session_state.setdefault(f"_vx_eff_{comp}", False)
    # apply model defaults once
    st.session_state.setdefault("_vx_defaults_applied", False)
    if not st.session_state["_vx_defaults_applied"] and getattr(cfg, "vaccination", None):
        vacc = cfg.vaccination or {}
        default_can = set(vacc.get("vaccinable_compartments", []))
        default_eff = set(vacc.get("effective_on_compartments", []))
        for comp in cfg.compartments:
            st.session_state[f"_vx_can_{comp}"] = comp in default_can
            st.session_state[f"_vx_eff_{comp}"] = comp in default_eff
        st.session_state["_vx_defaults_applied"] = True


def _collect_selected_compartments(cfg) -> Dict[str, List[str]]:
    vaccinable: List[str] = []
    effective_on: List[str] = []
    for comp in cfg.compartments:
        if st.session_state.get(f"_vx_can_{comp}"):
            vaccinable.append(comp)
        if st.session_state.get(f"_vx_eff_{comp}"):
            effective_on.append(comp)
    return {"vaccinable": vaccinable, "effective_on": effective_on}


def render_vaccinations_tab(cfg) -> None:
    """Render the Vaccinations tab to configure vaccine campaigns."""
    st.subheader("Vaccinations")

    _ensure_defaults(cfg)

    st.text_input(
        "Campaign name",
        key="_vx_name",
        placeholder="e.g., Healthcare workers Q1",
        help="Optional label to identify this campaign",
    )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        st.number_input("Start day", min_value=0, max_value=730, step=1, key="_vx_start")
    with c2:
        st.number_input("End day", min_value=0, max_value=730, step=1, key="_vx_end")
    with c3:
        st.slider("Vaccine coverage (%)", min_value=0, max_value=100, step=1, key="_vx_coverage")
    with c4:
        st.slider("Vaccine effectiveness (%)", min_value=0, max_value=100, step=1, key="_vx_effectiveness")

    # Determine age groups from selected country if available
    age_groups = None
    country = st.session_state.get("country_name")
    try:
        if country:
            pop = load_epydemix_population(country)
            age_groups = list(pop.Nk_names)
    except Exception:
        age_groups = None

    cc1, cc2 = st.columns(2)
    with cc1:
        with st.expander("Vaccinable compartments", expanded=False):
            for comp in cfg.compartments:
                st.checkbox(comp, key=f"_vx_can_{comp}")
    with cc2:
        with st.expander("Compartments affected by vaccine", expanded=False):
            for comp in cfg.compartments:
                st.checkbox(comp, key=f"_vx_eff_{comp}")

    # Age group targets selector
    with st.expander("Target age groups", expanded=False):
        if age_groups:
            if st.session_state.get("_vx_age_targets") is None:
                st.session_state["_vx_age_targets"] = list(age_groups)
            default_targets = list(st.session_state.get("_vx_age_targets", age_groups))
            selection = st.multiselect(
                "Select age groups to target",
                options=age_groups,
                default=default_targets,
                key="_vx_age_targets",
                help="By default, all age groups are targeted"
            )
        else:
            st.info("Age groups unavailable until a country is selected. All groups will be targeted by default.")

    if st.button("Add vaccination campaign"):
        name_v = (st.session_state.get("_vx_name") or "").strip()
        start_v = int(st.session_state["_vx_start"])
        end_v = int(st.session_state["_vx_end"])
        if end_v < start_v:
            st.warning("End day must be greater than or equal to start day.")
        else:
            selections = _collect_selected_compartments(cfg)
            new_item = {
                "name": name_v or f"Campaign {len(st.session_state['vaccinations_list']) + 1}",
                "start": start_v,
                "end": end_v,
                "coverage": float(st.session_state["_vx_coverage"]) / 100.0,
                "effectiveness": float(st.session_state["_vx_effectiveness"]) / 100.0,
                "vaccinable_compartments": selections["vaccinable"],
                "effective_on_compartments": selections["effective_on"],
                "target_age_groups": list(st.session_state.get("_vx_age_targets") or (age_groups or [])),
            }
            st.session_state["vaccinations_list"].append(new_item)

    items = st.session_state.get("vaccinations_list", [])
    if items:
        st.markdown("Added vaccination campaigns")
        rem_idx = None
        for idx, it in enumerate(items):
            c_name, c_start, c_end, c_cov, c_eff, c_btn = st.columns([1.2, 0.8, 0.8, 1, 1, 0.6])
            with c_name:
                st.write(f"Name: {it.get('name', '—')}")
            with c_start:
                st.write(f"Start Day: {it['start']}")
            with c_end:
                st.write(f"End Day: {it['end']}")
            with c_cov:
                st.write(f"Coverage: {int(round(100*it['coverage']))}%")
            with c_eff:
                st.write(f"Effectiveness: {int(round(100*it['effectiveness']))}%")
            with c_btn:
                if st.button("Remove", key=f"vx_remove_{idx}"):
                    rem_idx = idx
        if rem_idx is not None:
            del st.session_state["vaccinations_list"][rem_idx]
            st.rerun()
    else:
        st.info("No vaccination campaigns added yet.")

    # Store normalized structure for downstream consumption
    st.session_state["vaccinations"] = list(st.session_state.get("vaccinations_list", []))

    st.caption("Notes: Coverage is the fraction targeted; effectiveness is per-dose protection.")


