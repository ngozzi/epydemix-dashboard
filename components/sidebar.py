import streamlit as st
from .about import get_about_message

def show_sidebar_logo():
    st.sidebar.markdown(
        """
        <a href="https://epydemix.org" target="_blank">
            <img src="https://cdn.prod.website-files.com/67bde9057c9d78157874e100/67c1d1122127f0a9ce202197_epydemix-logo-p-500.png" 
                 style="width:100%;">
        </a>
        """,
        unsafe_allow_html=True
    )

def render_sidebar():
    show_sidebar_logo()
    st.markdown("## About")
    with st.expander("Readme", expanded=False):
        st.markdown(get_about_message())

    st.markdown("---")
    st.markdown("## Contacts")
    st.markdown("""
            <style>
            .contact-list{list-style:none;padding-left:0;margin:6px 0 0}
            .contact-list li{margin:8px 0;display:flex;align-items:center;gap:10px}
            .contact-list a{color:#e5e7eb;text-decoration:none}
            .contact-list a:hover{color:#fff;text-decoration:underline}
            .icon{width:18px;height:18px;stroke:#e5e7eb}
            .contact-list li:hover .icon{stroke:#fff}
            </style>
            <ul class="contact-list">
            <li>
                <svg class="icon" viewBox="0 0 24 24" fill="none"><path d="M3 7l9 6 9-6M5 6h14a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2z" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
                <a href="mailto:epydemix@isi.it">epydemix@isi.it</a>
            </li>
            <li>
                <svg class="icon" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke-width="1.6"/><path d="M7 12h10M12 7v10" stroke-width="1.6" stroke-linecap="round"/></svg>
                <a href="https://epydemix.org" target="_blank">Website</a>
            </li>
            <li>
                <svg class="icon" viewBox="0 0 24 24" fill="none"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3a3.4 3.4 0 0 0-.9-2.6c3-.3 6.1-1.5 6.1-6.7a5.2 5.2 0 0 0-1.4-3.6 4.8 4.8 0 0 0-.1-3.6s-1.2-.3-3.9 1.4a13.5 13.5 0 0 0-7 0C5.1 2.6 3.9 2.9 3.9 2.9a4.8 4.8 0 0 0-.1 3.6 5.2 5.2 0 0 0-1.4 3.6c0 5.2 3.1 6.4 6.1 6.7A3.4 3.4 0 0 0 9 19" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
                <a href="https://github.com/epistorm/epydemix" target="_blank">GitHub</a>
            </li>
            </ul>
            """, unsafe_allow_html=True)

