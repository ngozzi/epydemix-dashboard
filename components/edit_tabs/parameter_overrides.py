import streamlit as st

def _ensure_param_override_defaults(config):
    st.session_state.setdefault("parameter_overrides_list", [])  # list of {name,start,end,value}
    # defaults for the input widgets
    first_name = next(iter(config.parameters.keys())) if config.parameters else None
    st.session_state.setdefault("_po_name", first_name)
    st.session_state.setdefault("_po_start", 0)
    st.session_state.setdefault("_po_end", 250)
    # value default depends on selected param; set lazily in render

def _derive_overrides_dict_from_list(ovr_list):
    # Backward-compat: last override per parameter wins
    result = {}
    for item in ovr_list:
        result[item["name"]] = {
            "start_day": int(item["start"]),
            "end_day": int(item["end"]),
            "param": float(item["value"]),
        }
    return result

def render_parameter_overrides_tab(config):
    """Render the Parameter Overrides tab dynamically from model config."""

    st.subheader("Parameter Overrides")

    _ensure_param_override_defaults(config)

    param_names = list(config.parameters.keys())
    if not param_names:
        st.info("No parameters available to override.")
        st.session_state.parameter_overrides = {}
        return

    # Current spec for selected param
    selected_name = st.session_state.get("_po_name", param_names[0])
    if selected_name not in config.parameters:
        selected_name = param_names[0]
        st.session_state["_po_name"] = selected_name
    spec = config.parameters[selected_name]
    disp = config.param_display_names.get(selected_name, selected_name)
    minv, maxv, step, default = spec["min"], spec["max"], spec["step"], spec["default"]

    # Inputs row
    c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1.6])
    with c1:
        st.selectbox("Parameter", options=param_names, format_func=lambda n: config.param_display_names.get(n, n), key="_po_name")
    with c2:
        st.number_input("Start day", min_value=0, max_value=730, step=1, key="_po_start")
    with c3:
        # independent end; validate on add
        st.number_input("End day", min_value=0, max_value=730, step=1, key="_po_end")
    with c4:
        # ensure default for value is set consistent with selected param
        current_value_default = float(st.session_state.get("_po_value", default))
        # clamp to range
        if current_value_default < float(minv) or current_value_default > float(maxv):
            current_value_default = float(default)
        st.slider(
            f"Override value ({disp})",
            min_value=float(minv), max_value=float(maxv), step=float(step),
            value=current_value_default, key="_po_value"
        )

    if st.button("Add override"):
        start_v = int(st.session_state["_po_start"])
        end_v = int(st.session_state["_po_end"])
        if end_v < start_v:
            st.warning("End day must be greater than or equal to start day.")
        else:
            new_item = {
                "name": st.session_state["_po_name"],
                "start": start_v,
                "end": end_v,
                "value": float(st.session_state["_po_value"]),
            }
            st.session_state["parameter_overrides_list"].append(new_item)

    # List of overrides
    items = st.session_state.get("parameter_overrides_list", [])
    if items:
        st.markdown("Added overrides")
        remove_index = None
        for idx, it in enumerate(items):
            lcol, scol, ecol, vcol, dcol = st.columns([1.6, 0.8, 0.8, 1.2, 0.6])
            with lcol:
                disp_name = config.param_display_names.get(it["name"], it["name"]) if hasattr(config, 'param_display_names') else it["name"]
                st.code(disp_name, language=None)
            with scol:
                st.write(f"Start Day: {it['start']}")
            with ecol:
                st.write(f"End Day: {it['end']}")
            with vcol:
                st.write(f"Value: {it['value']}")
            with dcol:
                if st.button("Remove", key=f"po_remove_{idx}"):
                    remove_index = idx
        if remove_index is not None:
            del st.session_state["parameter_overrides_list"][remove_index]
            st.rerun()
    else:
        st.info("No overrides added yet.")

    # Backward compatible dict
    st.session_state.parameter_overrides = _derive_overrides_dict_from_list(
        st.session_state.get("parameter_overrides_list", [])
    )

    st.caption(
        "Notes: Overrides adjust model parameters only within the specified day range."
    )