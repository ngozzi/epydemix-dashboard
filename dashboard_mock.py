import streamlit as st
from components.header import show_dashboard_header
from components.sidebar import render_sidebar
from components.welcome_card import show_logos
from utils.helpers import load_locations
import json
import hashlib
from copy import deepcopy

st.set_page_config(page_title="Epydemix Simulation Dashboard", layout="wide", initial_sidebar_state="collapsed")

# ---- Parameter schemas (extend later)
MODEL_PARAM_SCHEMAS = {
    "SIR": [
        {"key": "R0", "label": "R0", "type": "float", "min": 0.1, "max": 10.0, "step": 0.1, "default": 1.6},
        {
            "key": "infectious_period",
            "label": "Infectious period (days)",
            "type": "float",
            "min": 0.5,
            "max": 30.0,
            "step": 0.5,
            "default": 5.0,
        },
    ],
    "SEIR": [
        {"key": "R0", "label": "R0", "type": "float", "min": 0.1, "max": 10.0, "step": 0.1, "default": 1.6},
        {
            "key": "incubation_period",
            "label": "Incubation period (days)",
            "type": "float",
            "min": 0.5,
            "max": 30.0,
            "step": 0.5,
            "default": 3.0,
        },
        {
            "key": "infectious_period",
            "label": "Infectious period (days)",
            "type": "float",
            "min": 0.5,
            "max": 30.0,
            "step": 0.5,
            "default": 5.0,
        },
    ],
}

LAYER_NAMES = ["home", "school", "work", "community"]

def render_saved_scenarios_list():
    scenarios = st.session_state.get("scenarios", {})
    if not scenarios:
        st.info("No saved scenarios yet.")
        return

    st.markdown("**Saved scenarios (snapshots)**")
    st.caption("To modify a scenario, change inputs and save a new snapshot.")

    remove_id = None

    for sid, item in scenarios.items():
        name = item.get("name", sid)

        with st.container(border=True):
            row = st.columns([0.35, 0.20, 0.20, 0.25], gap="small")

            with row[0]:
                st.markdown(f"**{name}**")
                st.caption(f"ID: {sid}")

            with row[1]:
                if st.button("Run", key=f"sc_run_{sid}", use_container_width=True):
                    st.session_state["last_run_scenario_id"] = sid

            with row[2]:
                view = st.button("View", key=f"sc_view_{sid}", use_container_width=True)

            with row[3]:
                if st.button("Remove", key=f"sc_rm_{sid}", use_container_width=True):
                    remove_id = sid

            if view:
                st.json(item.get("config", {}))

    if remove_id is not None:
        st.session_state["scenarios"].pop(remove_id, None)
        # keep state tidy
        if st.session_state.get("active_scenario_id") == remove_id:
            st.session_state["active_scenario_id"] = None
        if st.session_state.get("last_run_scenario_id") == remove_id:
            st.session_state["last_run_scenario_id"] = None
        st.rerun()


def _stable_hash_config(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]

def _ensure_scenario_state_defaults():
    st.session_state.setdefault("scenario_name", "Baseline")
    st.session_state.setdefault("scenarios", {})          # id -> {"name": str, "config": dict}
    st.session_state.setdefault("active_scenario_id", None)
    st.session_state.setdefault("last_run_scenario_id", None)

def build_current_config(model: str, geography: str) -> dict:
    """Collect current UI state into a serializable scenario config dict."""
    cfg = {
        "model": model,
        "geography": geography,
        "sim_length": int(st.session_state.get("sim_length", 250)),
        "initial_conditions": deepcopy(st.session_state.get("initial_conditions", {})),
        "model_params": deepcopy(st.session_state.get("model_params", {}).get(model, {})),
        "contact_interventions": deepcopy(st.session_state.get("contact_interventions", [])),
    }
    return cfg

def save_current_scenario(model: str, geography: str) -> str:
    cfg = build_current_config(model, geography)
    sid = _stable_hash_config(cfg)
    st.session_state["scenarios"][sid] = {
        "name": st.session_state.get("scenario_name", sid),
        "config": cfg,
    }
    st.session_state["active_scenario_id"] = sid
    return sid

def run_current_scenario(model: str, geography: str) -> str:
    # For now: "run" just saves and marks as last run.
    sid = save_current_scenario(model, geography)
    st.session_state["last_run_scenario_id"] = sid
    return sid


def _ensure_ic_state_defaults():
    st.session_state.setdefault("initial_conditions", {})
    st.session_state["initial_conditions"].setdefault("infected_pct", 0.1)
    st.session_state["initial_conditions"].setdefault("immune_pct", 0.0)


def render_initial_conditions():
    _ensure_ic_state_defaults()

    st.markdown("**Initial conditions**")
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


def _ensure_contact_state_defaults():
    st.session_state.setdefault("contact_interventions", [])  # list of dicts
    st.session_state.setdefault("_ci_new_layer", LAYER_NAMES[0])
    st.session_state.setdefault("_ci_new_start", 0)
    st.session_state.setdefault("_ci_new_end", 250)
    st.session_state.setdefault("_ci_new_red_pct", 0)

def _validate_window(start_day: int, end_day: int) -> bool:
    return end_day >= start_day

def _contact_summary(it: dict) -> str:
    return f"{it['layer']} · {int(it['reduction_pct'])}% · day {it['start_day']} → {it['end_day']}"

def render_contact_interventions():
    _ensure_contact_state_defaults()

    st.markdown("**Contact interventions**")
    st.caption("Reduce contacts by a given percentage between two days, within a contact layer.")

    # --- Add new (draft) card
    with st.container(border=True):
        st.markdown("**Add intervention**")

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

    # --- Existing intervention cards
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

def init_model_params_if_needed(model: str):
    """Ensure session_state has defaults for the selected model."""
    st.session_state.setdefault("model_params", {})
    st.session_state["model_params"].setdefault(model, {})

    for p in MODEL_PARAM_SCHEMAS[model]:
        st.session_state["model_params"][model].setdefault(p["key"], p["default"])

def render_model_params(model: str):
    """Render parameter inputs for the selected model from the schema."""
    init_model_params_if_needed(model)

    st.markdown("**Model parameters**")
    with st.container(border=True):
        for p in MODEL_PARAM_SCHEMAS[model]:
            k = p["key"]
            widget_key = f"param_{model}_{k}"

            # Use a local value from the session_state store
            current = st.session_state["model_params"][model][k]

            if p["type"] == "float":
                val = st.number_input(
                    p["label"],
                    min_value=float(p["min"]),
                    max_value=float(p["max"]),
                    value=float(current),
                    step=float(p["step"]),
                    key=widget_key,
                )
            else:
                # Extend with ints/selectboxes/etc. later
                val = st.text_input(p["label"], value=str(current), key=widget_key)

            # Persist back to the structured store
            st.session_state["model_params"][model][k] = val

def main(): 

    show_dashboard_header()
    left_col, right_col = st.columns([2., 3], gap="large")

    with left_col:
        st.subheader("Setup")

        # ---- Scenario name
        _ensure_scenario_state_defaults()
        st.text_input("Scenario name", key="scenario_name")

        # ---- Model and Geography selectors
        c1, c2 = st.columns(2)
        with c1:
            # --- Model selector
            model = st.selectbox(
                "Model",
                options=[
                    "SIR",
                    "SEIR"
                ],
                index=1,
            )

        with c2:
            # --- Geography selector
            geography = st.selectbox(
                "Geography",
                options=load_locations(),
                index=0,
                help="Type to search within the list.",
            )

        # ---- Simulation length (context)
        sim_length = st.number_input(
            "Simulation length (days)",
            min_value=1,
            max_value=5000,
            value=250, 
            step=10,
            help="Total duration of the simulation.",
            key="sim_length",
        )

        # Initial conditions
        with st.expander("Initial conditions", expanded=False):
            render_initial_conditions()

        # Model-dependent parameters
        with st.expander("Model parameters", expanded=False):
            render_model_params(model)

        with st.expander("Contact interventions", expanded=False):
            render_contact_interventions()


        # ---- Save and run buttons
        b1, b2 = st.columns(2, gap="small")
        with b1:
            if st.button("Save", use_container_width=True):
                sid = save_current_scenario(model, geography)  # <-- model/geography must exist; see note below
                st.success(f"Saved scenario: {st.session_state['scenarios'][sid]['name']}")
        with b2:
            if st.button("Run", type="primary", use_container_width=True):
                sid = run_current_scenario(model, geography)

        render_saved_scenarios_list()


    with right_col:
        st.subheader("Visualisation")
        # Temporary echo for debugging
        st.write(
            {
                "model": model,
                "geography": geography,
                "sim_length": st.session_state.get("sim_length"),
                "initial_conditions": st.session_state.get("initial_conditions", {}),
                "params": st.session_state.get("model_params", {}).get(model, {}),
                "contact_interventions": st.session_state.get("contact_interventions", []),
                "saved_scenarios": list(st.session_state.get("scenarios", {}).keys()),
                "last_run_scenario_id": st.session_state.get("last_run_scenario_id"),
            }
        )


    with st.sidebar:
        render_sidebar()

    show_logos()


if __name__ == "__main__":
    main()
