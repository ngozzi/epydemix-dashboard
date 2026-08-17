# --- Native-thread safety -----------------------------------------------------
# Streamlit runs the app in a worker thread. numpy/BLAS and numexpr (used by
# epydemix) spawning their own threads from there can cause native segfaults on
# some platforms. Force single-threaded native math before those libraries are
# imported. Set as defaults so an explicit environment override still wins.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
try:
    import numexpr
    numexpr.set_num_threads(1)
except Exception:
    pass
# -----------------------------------------------------------------------------

import streamlit as st
from layout.header import show_dashboard_header
from layout.sidebar import render_sidebar
from layout.logos import show_logos
from helpers import load_locations
from schemas import MODEL_PARAM_SCHEMAS
from ui.setup_panel import render_setup_panel
from ui.viz_panel import render_viz_panel
from state import ensure_workspace_defaults



def main():

    st.set_page_config(
        page_title="EpyScenario Dashboard",
        layout="wide",
        page_icon="assets/epydemix-icon.svg", 
        initial_sidebar_state="collapsed",
    )

    st.markdown("""
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 5rem;
            }
        </style>
        """, unsafe_allow_html=True)

    show_dashboard_header()
    ensure_workspace_defaults()

    left_col, right_col = st.columns([2.0, 3.0], gap="large")

    with left_col:
        model, geography = render_setup_panel(load_locations, MODEL_PARAM_SCHEMAS)

    with right_col:
        render_viz_panel(model, geography)

    with st.sidebar:
        render_sidebar()

    st.divider()
    show_logos()


if __name__ == "__main__":
    main()
