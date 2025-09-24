import streamlit as st

def render_parameter_overrides_tab(config):
    """Render the Parameter Overrides tab dynamically from model config."""

    st.subheader("Parameter Overrides")

    overrides = {}
    # 2-column grid for overrides
    param_names = list(config.parameters.keys())
    for i in range(0, len(param_names), 2):
        row = param_names[i:i+2]
        c1, c2 = st.columns(2)
        for col, pname in zip((c1, c2), row):
            spec = config.parameters[pname]
            disp = config.param_display_names[pname]
            minv, maxv, step, default = spec["min"], spec["max"], spec["step"], spec["default"]

            with col:
                with st.expander(f"Override {disp}", expanded=False):
                    st.checkbox(
                        "Enable override",
                        key=f"{pname}_ovr_en",
                        value=st.session_state.get(f"{pname}_ovr_en", False),
                    )

                    cst, cen = st.columns(2)
                    with cst:
                        st.number_input(
                            "Start day", min_value=0, max_value=730,
                            value=st.session_state.get(f"{pname}_ovr_start", 0),
                            step=1, key=f"{pname}_ovr_start"
                        )
                    with cen:
                        st.number_input(
                            "End day",
                            min_value=int(st.session_state.get(f"{pname}_ovr_start", 0)),
                            max_value=730,
                            value=st.session_state.get(f"{pname}_ovr_end", 250),
                            step=1, key=f"{pname}_ovr_end"
                        )

                    st.slider(
                        "Override value",
                        min_value=float(minv), max_value=float(maxv), step=float(step),
                        value=float(st.session_state.get(f"{pname}_ovr_value", default)),
                        key=f"{pname}_ovr_value",
                    )

    
    # Build dict from session_state
    for pname in config.parameters.keys():
        if st.session_state.get(f"{pname}_ovr_en", False):
            overrides[pname] = {
                "start_day": int(st.session_state[f"{pname}_ovr_start"]),
                "end_day":   int(st.session_state[f"{pname}_ovr_end"]),
                "param":     float(st.session_state[f"{pname}_ovr_value"]),
            }

    st.session_state.parameter_overrides = overrides

    # Summary
    st.markdown("### Overrides summary")
    if overrides:
        for k, v in overrides.items():
            disp = config.param_display_names[k]
            st.write(f"**{disp}**: days {v['start_day']}–{v['end_day']}, value {v['param']}")
    else:
        st.info("No overrides enabled.")

    # Notes
    st.caption(
        "Notes: Parameter overrides adjust the value of the model parameters (e.g. R₀ or infectious period) only within their day range."
    )