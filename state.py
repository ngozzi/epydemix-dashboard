# state.py

from __future__ import annotations
import json
import hashlib
from copy import deepcopy
from typing import Any, Dict
import streamlit as st
import pandas as pd
import numpy as np 
from engine.run import run_scenario
from constants import LAYER_NAMES, DEFAULT_AGE_GROUPS, START_DATE
from helpers import load_population, daily_doses_by_age
from epydemix import EpiModel
from epydemix.utils import compute_simulation_dates
from datetime import timedelta


WORKSPACE_KEYS_TO_CLEAR = [
    # scenario system
    "scenarios",
    "active_scenario_id",
    "last_run_scenario_id",
    "results",
    # interventions / inputs that belong to a workspace
    "contact_interventions",
    "vaccination_campaigns",
    "vaccination_settings",
    # model-specific inputs
    "model_params",
    "initial_conditions",
    # optional: scenario name draft
    "scenario_name",
]


def ensure_workspace_defaults() -> None:
    """
    Workspace defines the context under which scenario comparison is valid.
    Once set, model/geography should not change without starting a new session.
    """
    st.session_state.setdefault("workspace", None)  # None or {"model": str, "geography": str}


def reset_workspace() -> None:
    """
    Clear all state that should be tied to the current workspace.
    Keeps UI/theme preferences and other global state intact.
    """
    for k in WORKSPACE_KEYS_TO_CLEAR:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state["workspace"] = None


def ensure_vax_state_defaults(age_groups):
    st.session_state.setdefault("vaccination_campaigns", [])  # list of dicts

    # Draft (new campaign) widget state defaults
    st.session_state.setdefault("_vx_new_name", "Campaign 1")
    st.session_state.setdefault("_vx_new_start", 0)
    st.session_state.setdefault("_vx_new_end", 90)
    st.session_state.setdefault("_vx_new_cov_pct", 20)   # percent
    st.session_state.setdefault("_vx_new_ve_pct", 50)    # percent
    st.session_state.setdefault("_vx_new_rollout", "flat")
    st.session_state.setdefault("_vx_new_ramp_days", 14)
    st.session_state.setdefault("_vx_new_age_groups", age_groups[:1] if age_groups else [])
    st.session_state.setdefault("_vx_show_advanced", False)


def ensure_vax_settings_defaults():
    st.session_state.setdefault("vaccination_settings", {})
    st.session_state["vaccination_settings"].setdefault("target_compartments", ["S"])


def _stable_hash_config(cfg: Dict[str, Any]) -> str:
    cfg_for_hash = {
        k: v
        for k, v in cfg.items()
        if k not in {"population", "daily_doses_by_age", "daily_doses_by_age_daily", "spectral_radius_df"}  
    }
    payload = json.dumps(cfg_for_hash, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def ensure_scenario_state_defaults() -> None:
    st.session_state.setdefault("scenario_name", "Baseline")
    st.session_state.setdefault("scenarios", {})  # id -> {"name": str, "config": dict}
    st.session_state.setdefault("active_scenario_id", None)
    st.session_state.setdefault("last_run_scenario_id", None)


def ensure_results_defaults() -> None:
    st.session_state.setdefault("results", {})  # scenario_id -> pd.DataFrame


def build_current_config(model: str, geography: str) -> Dict[str, Any]:
    """
    Collect current UI state into a serializable scenario config dict.
    Keep this as the single source of truth for scenario serialization.
    """
    # import population
    population = load_population(geography)

    # compute daily doses by age
    df_doses = daily_doses_by_age(
        campaigns=st.session_state.get("vaccination_campaigns", []),
        Nk=population.Nk,
        sim_length=int(st.session_state.get("sim_length", 250)),
        age_groups=DEFAULT_AGE_GROUPS,
        dt=float(st.session_state.get("time_step", 0.2)),
    )
    df_doses_daily = df_doses.groupby(["t"], as_index=False).mean().reset_index()
    
    # compute spectral radius of contact matrix
    epi_model = EpiModel() 
    epi_model.set_population(population)
    for intervention in st.session_state.get("contact_interventions", {}):
        if intervention["layer"] == "all":
            for layer in LAYER_NAMES:
                epi_model.add_intervention(
                    layer_name=layer,
                    start_date=START_DATE + timedelta(days=intervention["start_day"]),
                    end_date=START_DATE + timedelta(days=intervention["end_day"]),
                    reduction_factor=1.0 - (intervention["reduction_pct"] / 100.)
                )
        else:
            epi_model.add_intervention(
                    layer_name=intervention["layer"],
                    start_date=START_DATE + timedelta(days=intervention["start_day"]),
                    end_date=START_DATE + timedelta(days=intervention["end_day"]),
                    reduction_factor=1.0 - (intervention["reduction_pct"] / 100.)
                )
    simulation_dates = compute_simulation_dates(START_DATE, START_DATE + timedelta(days=int(st.session_state.get("sim_length", 250))))
    epi_model.compute_contact_reductions(simulation_dates)

    spectral_radius_df = pd.DataFrame() 
    for layer in ["home", "school", "work", "community", "overall"]:
        if layer == "overall": 
            C = sum(population.contact_matrices.values())
        else: 
            C = population.contact_matrices[layer]
        rho_0 = np.linalg.eigvals(C).real.max()
        rhos = []
        for date in simulation_dates: 
            rhos.append(np.linalg.eigvals(epi_model.Cs[date][layer]).real.max()) 
        df_temp = pd.DataFrame(data={"rho": rhos, "rho_perc": 100 * np.array(rhos) / rho_0})
        df_temp["t"] = np.arange(0, len(rhos))
        df_temp["layer"] = layer
        spectral_radius_df = pd.concat((spectral_radius_df, df_temp), ignore_index=True)
    
    cfg: Dict[str, Any] = {
        "model": model,
        "geography": geography,
        "population": deepcopy(population),
        "sim_length": int(st.session_state.get("sim_length", 250)),
        "time_step": float(st.session_state.get("time_step", 0.2)),
        "initial_conditions": deepcopy(st.session_state.get("initial_conditions", {})),
        "model_params": deepcopy(st.session_state.get("model_params", {}).get(model, {})),
        "contact_interventions": deepcopy(st.session_state.get("contact_interventions", [])),
        "vaccination_settings": deepcopy(st.session_state.get("vaccination_settings", {})),
        "vaccination_campaigns": deepcopy(st.session_state.get("vaccination_campaigns", [])),
        "daily_doses_by_age": deepcopy(df_doses),
        "daily_doses_by_age_daily": deepcopy(df_doses_daily),
        "spectral_radius_df": deepcopy(spectral_radius_df),
    }
    return cfg


def save_current_scenario(model: str, geography: str) -> str:
    ensure_scenario_state_defaults()

    cfg = build_current_config(model, geography)
    sid = _stable_hash_config(cfg)

    st.session_state["scenarios"][sid] = {
        "name": st.session_state.get("scenario_name", sid),
        "config": cfg,
    }
    st.session_state["active_scenario_id"] = sid
    return sid


def run_current_scenario(model: str, geography: str) -> str:
    """
    Run current scenario:
    - Save scenario snapshot (stable id by config hash)
    - Execute model-specific engine runner
    - Store results in session_state["results"][sid]
    - Mark sid as last run
    """
    ensure_scenario_state_defaults()
    ensure_results_defaults()

    sid = save_current_scenario(model, geography)
    cfg = st.session_state["scenarios"][sid]["config"]

    df_comp, df_trans = run_scenario(cfg)  
    st.session_state["results"][sid] = {
        "compartments": df_comp,
        "transitions": df_trans,
    }

    st.session_state["last_run_scenario_id"] = sid
    return sid


def ensure_initial_conditions_defaults(model: str, ic_defaults: dict) -> None:
    st.session_state.setdefault("initial_conditions", {})

    defaults = ic_defaults.get(model, {"infected_pct": 0.1, "immune_pct": 0.0})

    st.session_state["initial_conditions"].setdefault("infected_pct", float(defaults["infected_pct"]))
    st.session_state["initial_conditions"].setdefault("immune_pct", float(defaults["immune_pct"]))


def ensure_contact_interventions_defaults() -> None:
    st.session_state.setdefault("contact_interventions", [])  # list of dicts
    st.session_state.setdefault("_ci_new_layer", LAYER_NAMES[0])
    st.session_state.setdefault("_ci_new_start", 0)
    st.session_state.setdefault("_ci_new_end", 250)
    st.session_state.setdefault("_ci_new_red_pct", 0)


def ensure_model_params_defaults(model: str, model_param_schemas: Dict[str, Any]) -> None:
    """
    Ensure session_state has defaults for the selected model.
    `model_param_schemas` should be MODEL_PARAM_SCHEMAS from schemas.py.
    """
    st.session_state.setdefault("model_params", {})
    st.session_state["model_params"].setdefault(model, {})

    for p in model_param_schemas[model]:
        st.session_state["model_params"][model].setdefault(p["key"], p["default"])


def reset_model_params_to_defaults(model: str, model_param_schemas: dict) -> None:
    st.session_state.setdefault("model_params", {})
    st.session_state["model_params"][model] = {}

    for p in model_param_schemas[model]:
        k = p["key"]
        default = p["default"]

        # reset the structured store
        st.session_state["model_params"][model][k] = default


def reset_initial_conditions_to_defaults(model: str, ic_defaults: dict) -> None:
    """
    Reset initial conditions for the given model to schema defaults, and clear
    corresponding widget keys so Streamlit re-applies defaults on next render.
    """
    st.session_state.setdefault("initial_conditions", {})

    defaults = ic_defaults.get(model, {"infected_pct": 0.1, "immune_pct": 0.0})

    st.session_state["initial_conditions"] = {
        "infected_pct": float(defaults["infected_pct"]),
        "immune_pct": float(defaults["immune_pct"]),
    }

