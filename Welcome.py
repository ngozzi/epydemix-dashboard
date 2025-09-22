import streamlit as st

from components.welcome_card import show_welcome_card
from components.layout import show_sidebar_logo, show_fixed_logo


st.set_page_config(page_title="Welcome • EpyDemix", layout="wide")

# Header logos row (ISI + partners)
show_fixed_logo()

# Sidebar population
with st.sidebar:
    show_sidebar_logo()
    st.markdown("### About")
    st.write(
        "Run epidemic simulations with configurable models and geographies."
    )
    st.markdown("### Contacts")
    st.write(
        "- Website: https://epydemix.org\n- Email: info@epydemix.org\n- GitHub: https://github.com/epydemix"
    )
    st.markdown("### Quick Links")
    st.write("Documentation, FAQ, Changelog")

# Main hero / welcome content
st.title("Welcome to EpyDemix")
show_welcome_card()

if st.button("Go to Dashboard", type="primary"):
    st.switch_page("dashboard.py")


