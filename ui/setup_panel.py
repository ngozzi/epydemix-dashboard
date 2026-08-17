# ui/setup_panel.py

import streamlit as st
import numpy as np
from ui.initial_conditions import render_initial_conditions
from ui.model_params import render_model_params
from ui.interventions import render_contact_interventions
from ui.scenarios import render_save_run_controls, render_saved_scenarios_list
from ui.vaccinations import render_vaccination_campaigns
from schemas import INITIAL_CONDITION_DEFAULTS, MODEL_PARAM_SCHEMAS
from state import reset_model_params_to_defaults, reset_initial_conditions_to_defaults, reset_workspace
from data.observed_datasets import OBSERVED_DATASETS


def render_observed_panel(model: str, geography: str) -> None:
    """Pertussis: select an observed case dataset to overlay, seed from, or
    auto-calibrate against."""
    st.session_state.setdefault("observed_dataset", None)
    st.session_state.setdefault("observed_mode", "Off")

    names = ["None"] + list(OBSERVED_DATASETS.keys())
    current = st.session_state.get("observed_dataset") or "None"
    choice = st.selectbox(
        "Observed dataset",
        options=names,
        index=names.index(current) if current in names else 0,
        help="Real-world case data (by age and vaccination status) to compare against or seed from.",
    )
    st.session_state["observed_dataset"] = None if choice == "None" else choice

    if choice == "None":
        st.caption("Select a dataset to overlay it on results, seed the outbreak from it, or auto-calibrate.")
        return

    meta = OBSERVED_DATASETS[choice]
    st.caption(meta.get("note", ""))
    hint = meta.get("geography_hint")
    if hint and hint != geography:
        st.info(f"This dataset is for **{hint}**. Set Geography to match for a like-for-like comparison.")

    st.radio(
        "Use the dataset as",
        options=["Off", "Overlay / compare", "Seed initial state"],
        key="observed_mode",
        help=(
            "Overlay / compare: show observed vs. modeled case age-distribution and "
            "vaccinated share after a run.  Seed initial state: start the outbreak "
            "from the observed counts (No→naive, Yes→partial), then project forward."
        ),
    )

    # ---- Auto-calibrate (coarse) --------------------------------------------
    st.markdown("**Auto-calibrate (coarse)**")
    st.caption(
        "Grid-search R₀, σ (partial infectiousness) and the Sₚ share of the immune "
        "pool to best match the observed vaccinated-share and age distribution. "
        "Uses a reduced number of stochastic runs for speed."
    )
    if st.button("Calibrate to observed", use_container_width=True):
        from state import build_current_config
        from engine.run import run_scenario
        from engine.calibration import calibrate_to_observed
        import engine.run as _runmod

        raw = meta["raw"]
        with st.spinner("Running coarse calibration (27 evaluations)…"):
            base = build_current_config(model, geography)
            old_nsim = _runmod.N_SIM
            _runmod.N_SIM = 5  # speed up the search
            try:
                res = calibrate_to_observed(base, raw, run_scenario)
            finally:
                _runmod.N_SIM = old_nsim

        best = res["best"]
        # Apply best params to the structured store and clear widget keys so the
        # inputs re-initialise to the calibrated values on rerun.
        mp_store = st.session_state["model_params"][model]
        mp_store["R0"] = float(best["R0"])
        mp_store["rel_infectiousness_partial"] = float(best["rel_infectiousness_partial"])
        st.session_state["initial_conditions"]["partial_immune_pct"] = float(best["partial_immune_pct"])
        for k in ("R0", "rel_infectiousness_partial"):
            st.session_state.pop(f"param_{model}_{k}", None)

        st.session_state["_calib_result"] = res
        st.rerun()

    res = st.session_state.get("_calib_result")
    if res:
        b = res["best"]
        st.success(
            f"Best fit → R₀={b['R0']}, σ={b['rel_infectiousness_partial']}, "
            f"Sₚ share={b['partial_immune_pct']}% · modeled vaccinated share "
            f"{b['modeled_partial_share']:.0%} vs observed "
            f"{res['observed']['overall_partial_share']:.0%} "
            f"(age-dist RMSE {b['age_dist_rmse']:.3f}). Applied — click Run to simulate."
        )


def _on_model_change():
    m = st.session_state["selected_model"]
    reset_model_params_to_defaults(m, MODEL_PARAM_SCHEMAS)
    reset_initial_conditions_to_defaults(m, INITIAL_CONDITION_DEFAULTS)


def render_setup_panel(load_locations_fn, model_param_schemas):
    st.subheader("Setup")

    # Initialize workspace early if scenarios/results exist but workspace doesn't
    scenarios_exist = bool(st.session_state.get("scenarios"))
    results_exist = bool(st.session_state.get("results"))
    
    # Get current selections early (before any UI elements)
    st.session_state.setdefault("selected_model", "SEIR (Measles)")
    st.session_state.setdefault("selected_geography", load_locations_fn()[0])
    
    # Initialize workspace if needed BEFORE checking workspace_active
    if st.session_state.get("workspace") is None and (scenarios_exist or results_exist):
        st.session_state["workspace"] = {
            "model": st.session_state["selected_model"],
            "geography": st.session_state["selected_geography"]
        }
    
    # Now get the workspace state
    workspace = st.session_state.get("workspace")
    workspace_active = (workspace is not None)

    # Show workspace info if active
    if workspace_active:
        with st.container(border=True):
            st.markdown("**Workspace**")
            st.caption(
                f"{workspace['model']} · {workspace['geography']} — scenario comparison is restricted to this context. Start a new session to change the model or geography."
            )

            if st.button("Start new session", type="primary", use_container_width=True):
                reset_workspace()
                st.rerun()

    # Model & geography
    c1, c2 = st.columns(2)
    with c1:
        model = st.selectbox(
            "Model",
            options=["SEIR (Measles)", "SEIRS (Influenza)", "SEIRS (Pertussis)", "SEIHR (COVID-19)"],
            key="selected_model",
            on_change=_on_model_change,
            disabled=workspace_active,
            help="Select the model to use for the simulation.",
        )
    
    with c2:
        # load locations
        locations = load_locations_fn()
        geography = st.selectbox(
            "Geography",
            options=locations,
            key="selected_geography",
            help="Type to search within the list.",
            disabled=workspace_active,
        )

    # Simulation length and time step
    c1, c2 = st.columns(2)
    with c1:
        st.number_input(
            "Simulation length (days)",
            min_value=1,
            max_value=5000,
            value=250,
            step=10,
            help="Total duration of the simulation.",
            key="sim_length",
        )
    
    with c2:
        st.number_input(
            "$\Delta t$ (days)",
            min_value=0.1,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Time step for the simulation. Smaller values result in more accurate simulations but require more computational resources.",
            key="time_step",
        )

    # Scenario controls (use model/geography)
    render_save_run_controls(model, geography)

    st.caption("Change the settings below to create custom scenarios.")

    # Expanders
    with st.expander("Initial conditions", expanded=False):
        render_initial_conditions(model, INITIAL_CONDITION_DEFAULTS)

    with st.expander("Model parameters", expanded=False):
        render_model_params(model, model_param_schemas)

    with st.expander("Contact interventions", expanded=False):
        render_contact_interventions()

    with st.expander("Vaccination campaigns", expanded=False):
        render_vaccination_campaigns(model)

    if model == "SEIRS (Pertussis)":
        with st.expander("Observed data & calibration", expanded=False):
            render_observed_panel(model, geography)

    render_saved_scenarios_list()

    return model, geography