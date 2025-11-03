from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import numexpr as ne
import streamlit as st
from epydemix.model import EpiModel

# ----------------- Data structures -----------------
@dataclass
class ModelConfig:
    name: str
    compartments: List[str]
    parameters: Dict[str, Dict[str, Any]]
    derived_parameters: Dict[str, str]
    engine_parameters: Dict[str, str]              
    transitions: List[Dict[str, Any]]
    initial_conditions: Dict[str, float] = field(default_factory=dict)
    param_display_names: Dict[str, str] = field(default_factory=dict)

# ----------------- Loading/validation -----------------
def load_model_config_from_file(path: str) -> ModelConfig:
    with open(path, "r") as f:
        cfg = json.load(f)
    return _validate_and_normalize(cfg)

def load_model_config_from_json_bytes(b: bytes) -> ModelConfig:
    cfg = json.loads(b.decode("utf-8"))
    return _validate_and_normalize(cfg)

def _validate_and_normalize(cfg: Dict[str, Any]) -> ModelConfig:
    required_top = ["name", "compartments", "parameters", "derived_parameters", "transitions"]
    for k in required_top:
        if k not in cfg:
            raise ValueError(f"Missing top-level key '{k}' in config")

    # engine_parameters optional but recommended
    engine_params = cfg.get("engine_parameters", {})

    # Light validation
    if not isinstance(cfg["compartments"], list) or not cfg["compartments"]:
        raise ValueError("compartments must be a non-empty list")

    for p, spec in cfg["parameters"].items():
        for req in ["default", "min", "max", "step"]:
            if req not in spec:
                raise ValueError(f"parameter '{p}' missing '{req}'")
        if spec["min"] > spec["max"]:
            raise ValueError(f"parameter '{p}' has min > max")

    for tr in cfg["transitions"]:
        for req in ["source", "target", "type", "params"]:
            if req not in tr:
                raise ValueError(f"transition missing '{req}'")
        if tr["type"] not in ("mediated", "spontaneous"):
            raise ValueError("transition type must be 'mediated' or 'spontaneous'")

    # initial_conditions optional
    init_cond = cfg.get("initial_conditions", {})

    # Handle display names (optional)
    display_names = cfg.get("param_display_names", {})
    # If missing, fallback: each parameter’s name is its display name
    if not display_names:
        display_names = {p: p for p in cfg["parameters"].keys()}

    return ModelConfig(
        name=cfg["name"],
        compartments=cfg["compartments"],
        parameters=cfg["parameters"],
        derived_parameters=cfg["derived_parameters"],
        engine_parameters=engine_params,
        transitions=cfg["transitions"],
        initial_conditions=init_cond, 
        param_display_names=display_names
    )

# ----------------- Context and derived eval -----------------
def compute_spectral_radius(population) -> float:
    # population.contact_matrices is a dict[layer] -> np.array
    layers = list(population.contact_matrices.keys())
    C = np.array([population.contact_matrices[layer] for layer in layers])
    overall = C.sum(axis=0)
    return np.max(np.linalg.eigvals(overall)).real

def eval_derived(derived_specs: Dict[str, str], params: Dict[str, float], context: Dict[str, float]) -> Dict[str, float]:
    env = {**params, **context}
    out = {}
    for k, expr in derived_specs.items():
        try:
            out[k] = float(ne.evaluate(expr, local_dict=env))
        except Exception as e:
            raise ValueError(f"Failed to evaluate derived '{k} = {expr}': {e}")
    return out

# ----------------- Build model -----------------
def build_epimodel_from_config(cfg: ModelConfig, derived: Dict[str, float], population) -> EpiModel:
    m = EpiModel()
    m.set_population(population)
    m.add_compartments(cfg.compartments)

    for tr in cfg.transitions:
        # Resolve transition params: tokens can be names from 'derived' OR compartment names.
        resolved: List[Any] = []
        for token in tr["params"]:
            if isinstance(token, (int, float)):
                resolved.append(token)
            elif isinstance(token, str):
                if token in derived:
                    resolved.append(derived[token])  # numeric
                else:
                    # keep as string (e.g. "Infected" for mediated transitions)
                    resolved.append(token)
            else:
                raise ValueError(f"Unsupported param token type: {type(token)}")
        m.add_transition(
            source=tr["source"],
            target=tr["target"],
            kind=tr["type"],
            params=tuple(resolved) if tr["type"] == "mediated" else resolved[0]
        )
    return m

# ----------------- Sidebar rendering helpers -----------------
def render_config_params(cfg: ModelConfig, prefix: str = "param_") -> Dict[str, float]:
    """Render sliders for 'parameters' and return dict of chosen values."""
    chosen: Dict[str, float] = {}
    with st.expander("Select model parameters.", expanded=True):
        for pname, spec in cfg.parameters.items():
            dtype = spec.get("dtype", "float")
            label = cfg.param_display_names[pname]
            widget_key = f"{prefix}{pname}"
            # Seed session once
            if widget_key not in st.session_state:
                st.session_state[widget_key] = spec["default"]

            if dtype == "int":
                chosen[pname] = st.slider(
                    label,
                    min_value=int(spec["min"]),
                    max_value=int(spec["max"]),
                    value=int(st.session_state[widget_key]),
                    step=int(spec["step"]),
                    key=widget_key,
                )
            else:
                chosen[pname] = st.slider(
                    label,
                    min_value=float(spec["min"]),
                    max_value=float(spec["max"]),
                    value=float(spec["default"]),
                    #value=float(st.session_state[widget_key]),
                    step=float(spec["step"]),
                    #key=widget_key,
                )
            
    return chosen

# ----------------- Overrides (optional) -----------------
def compute_override_value(pname: str, value: float, cfg: ModelConfig, base_params: Dict[str, float], context: Dict[str, float]) -> Tuple[str, float]:
    """
    Given a high-level user parameter name (e.g., 'R0' or 'infectious_period') and an override value,
    compute which engine parameter to override and with what numeric value.
    Returns (engine_param_name, override_value).
    """
    # Recompute derived with the overridden high-level pname -> value, others from base_params
    overridden_params = {**base_params, pname: value}
    derived = eval_derived(cfg.derived_parameters, overridden_params, context)

    # Decide which engine parameter to touch.
    # Heuristic:
    # - If overriding R0, we want to change 'beta'
    # - If overriding infectious_period, we want to change 'mu'
    # You can also generalize by keeping a mapping in config if needed.
    target_derived = None
    if pname.lower() in ("r0", "r_0"):
        target_derived = "beta"
    elif "infectious" in pname.lower():
        target_derived = "mu"
    else:
        # fallback: if pname appears in engine_parameters mapping directly
        target_derived = pname if pname in cfg.engine_parameters else None

    if not target_derived:
        raise ValueError(f"Don't know which engine parameter to override for '{pname}'")

    engine_param = cfg.engine_parameters.get(target_derived, target_derived)
    override_value = derived[target_derived]
    return engine_param, float(override_value)
