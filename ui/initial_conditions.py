# ui/initial_conditions.py

import streamlit as st
from state import ensure_initial_conditions_defaults
from constants import DEFAULT_AGE_GROUPS
from data.coverage_store import load_saved_coverage


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

        # Pertussis has an 8-compartment structure, so the two coarse sliders
        # above are further split across the naive and partial-immunity tracks.
        if model == "SEIRS (Pertussis)":
            st.caption("Partial-immunity structure (pertussis only).")
            c3, c4 = st.columns(2, gap="small")

            with c3:
                partial_infection_pct = st.number_input(
                    "Infections in partial track (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state["initial_conditions"].get("partial_infection_pct", 33.0)),
                    step=1.0,
                    help="Share of the initial infections seeded in the partial-immunity track (Eₚ/Iₚ) rather than the naive track (E/I).",
                )

            with c4:
                partial_immune_pct = st.number_input(
                    "Sₚ share of immune pool (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state["initial_conditions"].get("partial_immune_pct", 71.0)),
                    step=1.0,
                    help="Share of the background-immunity pool placed in partially-susceptible Sₚ. The remainder is split between Rₚ and R.",
                )

            st.session_state["initial_conditions"]["partial_infection_pct"] = float(partial_infection_pct)
            st.session_state["initial_conditions"]["partial_immune_pct"] = float(partial_immune_pct)

            # Age-stratified starting immunity from a saved coverage profile
            # (managed on the Vaccination Planner page). When enabled it overrides
            # the uniform "Background immunity" slider above.
            profiles = load_saved_coverage()
            use_age = st.checkbox(
                "Use a saved coverage profile as starting immunity (age-stratified)",
                value=bool(st.session_state.get("use_age_immunity", False)),
                help="Overrides the uniform 'Background immunity' with per-age-band immunity. "
                     "Create/edit profiles on the Vaccination Planner page.",
            )
            if use_age and profiles:
                names = list(profiles.keys())
                cur = st.session_state.get("age_immunity_profile")
                idx = names.index(cur) if cur in names else 0
                sel = st.selectbox("Coverage profile", options=names, index=idx, key="ic_coverage_profile")
                prof = profiles[sel]
                st.session_state["age_immunity_pct"] = {ag: float(prof.get(ag, 0.0)) for ag in DEFAULT_AGE_GROUPS}
                st.session_state["age_immunity_profile"] = sel
                st.session_state["use_age_immunity"] = True
                st.caption(
                    "Applied — "
                    + " · ".join(f"{ag} {float(prof.get(ag, 0.0)):.0f}%" for ag in DEFAULT_AGE_GROUPS)
                )
            else:
                st.session_state["use_age_immunity"] = False
