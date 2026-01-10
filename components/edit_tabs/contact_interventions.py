import streamlit as st

LAYER_NAMES = ["home", "school", "work", "community"]

def _ensure_state_defaults():
    st.session_state.setdefault("contact_interventions", [])  # list of {layer,start,end,reduction}
    st.session_state.setdefault("_ci_layer", LAYER_NAMES[0])
    st.session_state.setdefault("_ci_start", 0)
    st.session_state.setdefault("_ci_end", 250)
    st.session_state.setdefault("_ci_red_pct", 0)

def _derive_interventions_dict_from_list(int_list):
    # Backward-compat: build a dict mapping last intervention per layer
    result = {}
    for item in int_list:
        result[item["layer"]] = {
            "start": int(item["start"]),
            "end": int(item["end"]),
            "reduction": float(item["reduction"]),
        }
    return result

def render_contact_interventions_tab():
    """Render the Contact Interventions tab."""

    st.subheader("Contact Interventions")

    _ensure_state_defaults()

    # Input controls
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.2])
    with c1:
        st.selectbox("Layer", options=LAYER_NAMES, key="_ci_layer")
    with c2:
        st.number_input(
            "Start day", min_value=0, max_value=730, step=1, key="_ci_start"
        )
    with c3:
        # End day input independent; enforce end>=start when adding
        st.number_input(
            "End day", min_value=0, max_value=730, step=1, key="_ci_end"
        )
    with c4:
        st.slider(
            "Reduction (%)", min_value=0, max_value=100, step=1, key="_ci_red_pct"
        )

    if st.button("Add intervention"):
        start_v = int(st.session_state["_ci_start"])
        end_v = int(st.session_state["_ci_end"])
        if end_v < start_v:
            st.warning("End day must be greater than or equal to start day.")
        else:
            new_item = {
                "layer": st.session_state["_ci_layer"],
                "start": start_v,
                "end": end_v,
                "reduction": float(st.session_state["_ci_red_pct"]) / 100.0,
            }
            st.session_state["contact_interventions"].append(new_item)
            # optional: do not force rerun; Streamlit will rerun once due to the button itself

    # Display list of added interventions
    items = st.session_state.get("contact_interventions", [])
    if items:
        st.markdown("Added interventions")
        remove_index = None
        for idx, it in enumerate(items):
            lcol, scol, ecol, rcol, dcol = st.columns([1.2, 0.8, 0.8, 1, 0.6])
            with lcol:
                st.code(it["layer"], language=None)
            with scol:
                st.write(f"Start Day: {it['start']}")
            with ecol:
                st.write(f"End Day: {it['end']}")
            with rcol:
                pct = int(round(it["reduction"] * 100))
                st.write(f"Reduction: {pct}%")
            with dcol:
                if st.button("Remove", key=f"ci_remove_{idx}"):
                    remove_index = idx
        if remove_index is not None:
            del st.session_state["contact_interventions"][remove_index]
            st.rerun()
    else:
        st.info("No interventions added yet.")

    # Backward-compat output for downstream consumers
    st.session_state.interventions = _derive_interventions_dict_from_list(
        st.session_state.get("contact_interventions", [])
    )

    st.caption(
        "Notes: Add one or more interventions. Reduction scales contacts within the given window."
    )

