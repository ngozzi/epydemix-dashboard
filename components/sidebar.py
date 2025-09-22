import streamlit as st
from utils.helpers import load_locations, reset_all_state
from utils.config_engine import load_model_config_from_file, load_model_config_from_json_bytes
from .about import get_about_message

BUILTINS = {"SEIR": "models/seir.json", "SIR": "models/sir.json", "SIS": "models/sis.json"}

def render_sidebar():
    st.sidebar.title("⚙️ Model and Population Setup")
    with st.sidebar.form("model_setup"):
        with st.expander("Select or Upload Model", expanded=False):
            model_choice = st.selectbox("Predefined Models", list(BUILTINS.keys()), index=1)
            model_uploaded_config = st.file_uploader("Upload custom model config (JSON)", type="json")

        with st.expander("Select Geography", expanded=False):
            country_name = st.selectbox(
                "Select Geography",
                options=load_locations(),
                index=0
            )
        # TODO: add upload of population file
        
        create_btn = st.form_submit_button("🚀 Create Model", use_container_width=True)
        if create_btn:
            # TODO: add reset of states and edit tabs when creating a new model
            if model_uploaded_config is not None:
                st.session_state.model_config = load_model_config_from_json_bytes(model_uploaded_config.read())
            else:
                st.session_state.model_config = load_model_config_from_file(BUILTINS[model_choice])
            st.session_state.country_name = country_name
            st.session_state.model_loaded = True
            st.session_state.simulation_results = None
            st.session_state.edit_mode = True

        st.markdown("### 📖 About")
        with st.expander("Readme", expanded=False):
            st.markdown(get_about_message())
