import streamlit as st

LAYER_NAMES = ["home", "school", "work", "community"]

def render_contact_interventions_tab():
    """Render the Contact Interventions tab."""

    st.subheader("🤝 Contact Interventions")

    # Initialize defaults in session_state
    for layer in LAYER_NAMES:
        st.session_state.setdefault(f"{layer}_en", False)
        st.session_state.setdefault(f"{layer}_start", 0)
        st.session_state.setdefault(f"{layer}_end", 250) # default to 250 days
        st.session_state.setdefault(f"{layer}_red", 0)

    interventions = {}

    for sel in LAYER_NAMES:
        with st.expander(f"Configure: {sel}", expanded=False):
            # Checkbox
            st.checkbox(
                f"Enable intervention on {sel}",
                key=f"{sel}_en",
                value=bool(st.session_state[f"{sel}_en"]),
            )

            cur_start = int(st.session_state[f"{sel}_start"])
            cur_end   = int(st.session_state[f"{sel}_end"])

            c1, c2 = st.columns(2)
            with c1:
                st.number_input(
                    "Start day",
                    min_value=0,
                    max_value=730,
                    value=cur_start,
                    step=1,
                    key=f"{sel}_start",
                )
            new_start = int(st.session_state[f"{sel}_start"])
            safe_end_default = max(new_start, cur_end)

            with c2:
                st.number_input(
                    "End day",
                    min_value=new_start,
                    max_value=730,
                    value=safe_end_default,
                    step=1,
                    key=f"{sel}_end",
                )

            st.slider(
                "Reduction of contacts (%)",
                min_value=0, max_value=100, step=1,
                value=int(st.session_state[f"{sel}_red"]),
                key=f"{sel}_red",
            )

    # Collect enabled interventions
    for layer in LAYER_NAMES:
        if st.session_state[f"{layer}_en"]:
            interventions[layer] = {
                "start": int(st.session_state[f"{layer}_start"]),
                "end":   int(st.session_state[f"{layer}_end"]),
                "reduction": int(st.session_state[f"{layer}_red"]) / 100.0,
            }

    # Store in session_state
    st.session_state.interventions = interventions

    # Summary
    st.markdown("### Intervention summary")
    if interventions:
        for k, v in interventions.items():
            st.write(f"**{k}**: days {v['start']}–{v['end']}, reduction {int(v['reduction']*100)}%")
    else:
        st.info("No interventions enabled.")
