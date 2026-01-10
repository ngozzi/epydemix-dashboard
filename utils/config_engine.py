from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
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
    vaccination: Dict[str, Any] = field(default_factory=dict)

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
        param_display_names=display_names,
        vaccination=cfg.get("vaccination", {})
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
    Given a user-facing parameter name and an override value, compute which engine
    parameter to override and the numeric value to use.

    Strategy:
      - If the user overrides a parameter that is itself directly used by the engine
        (i.e. appears in cfg.engine_parameters keys), override that directly.
      - Otherwise, recompute derived parameters with the override applied and choose
        the engine-used derived parameter(s) that depend on this pname. If multiple
        are affected, pick the one with the most specific dependency (fewest inputs).
    """
    if not cfg.engine_parameters:
        raise ValueError("Model config is missing engine_parameters mapping.")

    engine_targets: Set[str] = set(cfg.engine_parameters.keys())

    # 1) Direct mapping: if pname itself is an engine-target derived
    if pname in engine_targets:
        engine_param = cfg.engine_parameters[pname]
        # Use the value as-is (user provided engine-level value)
        return engine_param, float(value)

    # 2) Compute new derived values with the override applied
    overridden_params = {**base_params, pname: value}
    new_derived = eval_derived(cfg.derived_parameters, overridden_params, context)

    # For selection, we need dependency info: which derived depends on which base params
    def extract_deps(expr: str, param_names: Set[str], derived_names: Set[str]) -> Set[str]:
        tokens: Set[str] = set()
        name = ""
        for ch in expr:
            if ch.isalnum() or ch == "_":
                name += ch
            else:
                if name:
                    tokens.add(name)
                    name = ""
        if name:
            tokens.add(name)
        # Keep only references to base params or other deriveds
        return {t for t in tokens if t in param_names or t in derived_names}

    param_names: Set[str] = set(cfg.parameters.keys())
    derived_names: Set[str] = set(cfg.derived_parameters.keys())

    # Build a map of derived -> direct dependencies on base params (flattening derived->params assuming one level)
    direct_dep_map: Dict[str, Set[str]] = {}
    for dname, expr in cfg.derived_parameters.items():
        deps = extract_deps(expr, param_names, derived_names)
        # If a derived depends on other deriveds, we conservatively include all base params
        # referenced by those deriveds as well (one-level expansion covers common cases).
        expanded: Set[str] = set()
        for dep in deps:
            if dep in param_names:
                expanded.add(dep)
            elif dep in derived_names:
                # include that derived's immediate param deps if available
                inner = extract_deps(cfg.derived_parameters[dep], param_names, derived_names)
                expanded |= {x for x in inner if x in param_names}
        direct_dep_map[dname] = expanded if expanded else set()

    # 3) Identify which engine-target derived params are affected by this pname
    affected: List[str] = [d for d in engine_targets if pname in direct_dep_map.get(d, set())]

    if not affected:
        # No engine parameter depends on this pname; nothing to do
        raise ValueError(f"No engine parameters depend on '{pname}'.")

    # 4) If multiple affected, choose the most specific (fewest dependencies)
    affected.sort(key=lambda d: (len(direct_dep_map.get(d, set())), d))
    target_derived: str = affected[0]

    engine_param = cfg.engine_parameters.get(target_derived, target_derived)
    override_value = float(new_derived[target_derived])
    return engine_param, override_value
