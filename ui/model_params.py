# ui/model_params.py

import streamlit as st
from state import ensure_model_params_defaults


def render_model_params(model: str, model_param_schemas: dict) -> None:
    ensure_model_params_defaults(model, model_param_schemas)
    st.caption("Specify model parameters.")
    with st.container(border=True):
        for p in model_param_schemas[model]:
            k = p["key"]
            widget_key = f"param_{model}_{k}"

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
                val = st.text_input(p["label"], value=str(current), key=widget_key)

            st.session_state["model_params"][model][k] = val
