import streamlit as st
from components.welcome_card import show_welcome_card
from utils.session_state_init import init_session_state
    

def main():
    # Set page config
    st.set_page_config(page_title="Epydemix Dashboard", layout="wide")
    st.title("Epydemix Simulation Dashboard")
    st.caption("Skeleton: tabs, forms, state management, and run flow only")

    # Sidebar global info on skeleton
    with st.sidebar:
        st.markdown("### About")
        st.write("Run epidemic simulations with configurable models and geographies.")
        st.markdown("### Contacts")
        st.write("- Website: https://epydemix.org\n- Email: info@epydemix.org\n- GitHub: https://github.com/epydemix")

    if st.session_state.get("show_welcome"):
        show_welcome_card()

if __name__ == "__main__":
    main()


