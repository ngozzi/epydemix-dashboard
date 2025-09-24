import streamlit as st
from utils.helpers import load_locations
from utils.config_engine import load_model_config_from_file, load_model_config_from_json_bytes

BUILTINS = {"SIR": "models/sir.json", "SEIR": "models/seir.json", "SIS": "models/sis.json"}

def render_model_location_tab(): 
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model")

        with st.expander("Select or Upload Model", expanded=True):
            model_choice_ui = st.selectbox("Predefined Models", list(BUILTINS.keys()), key="model_choice_ui")
            model_uploaded_config = st.file_uploader("Upload custom model config (JSON)", type="json")

    with col2:
        st.subheader("Geography")
        with st.expander("Select Geography", expanded=True):
            country_name_ui = st.selectbox(
                "Select Geography",
                options=load_locations(),
                key="country_name_ui",
                label_visibility="collapsed"
            )

    save_btn = st.button("Save Settings", use_container_width=True)
    if save_btn:
        if model_uploaded_config is not None:
            st.session_state["model_config"] = load_model_config_from_json_bytes(model_uploaded_config.read())
        else:
            st.session_state["model_config"] = load_model_config_from_file(BUILTINS[model_choice_ui])
        st.session_state["country_name"] = country_name_ui
        st.session_state["model_loaded"] = True
        st.success("Settings saved")
