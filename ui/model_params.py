# ui/model_params.py

import streamlit as st
from state import ensure_model_params_defaults
from constants import DEFAULT_AGE_GROUPS


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
                st.session_state["model_params"][model][k] = val

            elif p["type"] == "by_age_float":
                st.markdown(f"**{p['label']}**")

                # Create columns for each age group
                cols = st.columns(len(DEFAULT_AGE_GROUPS))
                val = []
                
                for i, (col, age_group) in enumerate(zip(cols, DEFAULT_AGE_GROUPS)):
                    with col:
                        age_val = st.number_input(
                            age_group,
                            min_value=float(p["min"]),
                            max_value=float(p["max"]),
                            value=float(current[i]),
                            step=float(p["step"]),
                            key=f"{widget_key}_{i}",
                            label_visibility="visible"
                        )
                        val.append(age_val)
                        st.session_state["model_params"][model][f"{k}_{i}"] = age_val

            elif p["type"] == "discrete":
                val = st.selectbox(p["label"], options=p["options"], index=p["options"].index(current), key=widget_key)
                st.session_state["model_params"][model][k] = val
            
            else:
                val = st.text_input(p["label"], value=str(current), key=widget_key)
                st.session_state["model_params"][model][k] = val

            
