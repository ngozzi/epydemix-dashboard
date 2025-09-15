import streamlit as st

def render_parameter_overrides_tab(config):
    """Render the Parameter Overrides tab dynamically from model config."""

    st.subheader("🦠 Parameter Overrides")

    overrides = {}

    for pname, spec in config.parameters.items():
        disp = config.param_display_names[pname]  
        minv, maxv, step, default = spec["min"], spec["max"], spec["step"], spec["default"]

        with st.expander(f"Override {disp}", expanded=False):
            # Enable checkbox
            st.checkbox(
                "Enable override",
                key=f"{pname}_ovr_en",
                value=st.session_state.get(f"{pname}_ovr_en", False),
            )

            c1, c2 = st.columns(2)
            with c1:
                st.number_input(
                    "Start day",
                    min_value=0,
                    max_value=730,
                    value=st.session_state.get(f"{pname}_ovr_start", 0),
                    step=1,
                    key=f"{pname}_ovr_start"
                )
            with c2:
                st.number_input(
                    "End day",
                    min_value=int(st.session_state.get(f"{pname}_ovr_start", 0)),
                    max_value=730,
                    value=st.session_state.get(f"{pname}_ovr_end", 250),
                    step=1,
                    key=f"{pname}_ovr_end",
                )

            st.slider(
                "Override value",
                min_value=float(minv),
                max_value=float(maxv),
                step=float(step),
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
            st.write(f"**{k}**: days {v['start_day']}–{v['end_day']}, value {v['param']}")
    else:
        st.info("No overrides enabled.")
