import streamlit as st
from helpers import data_uri

def show_sidebar_logo():
    epydemix = data_uri("assets/epydemix-icon.svg")
    st.sidebar.markdown(
        f"""
        <a href="https://epydemix.org" target="_blank">
            <img src="{epydemix}" 
                style="width: 50px; display: block; margin-left: 0;">
        </a>
        """,
        unsafe_allow_html=True
    )

def render_sidebar():
    show_sidebar_logo()
    # Add custom CSS for the sidebar
    st.markdown("""
        <style>
            /* Sidebar styling - dark grey background */
            [data-testid="stSidebar"] {
                background: #1a1a1a;
            }
            
            /* Sidebar content padding */
            [data-testid="stSidebar"] > div:first-child {
                padding-top: 2rem;
            }
            
            /* Style the navigation links */
            [data-testid="stSidebarNav"] {
                padding-top: 1rem;
            }
            
            /* Style navigation items */
            [data-testid="stSidebarNav"] a {
                background-color: rgba(255, 255, 255, 0.03);
                border-radius: 8px;
                padding: 0.75rem 1rem;
                margin-bottom: 0.5rem;
                transition: all 0.3s ease;
                border-left: 3px solid transparent;
            }
            
            [data-testid="stSidebarNav"] a:hover {
                background-color: rgba(255, 174, 66, 0.1);
                border-left-color: #FFAE42;
                transform: translateX(4px);
            }
            
            /* Active page highlight */
            [data-testid="stSidebarNav"] a[aria-current="page"] {
                background-color: rgba(255, 174, 66, 0.15);
                border-left-color: #FFAE42;
                font-weight: 600;
            }
            
            /* Contacts section styling */
            .contacts-section {
                margin-top: 3rem;
                padding: 1.5rem;
                background: rgba(255, 255, 255, 0.03);
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            
            .contacts-title {
                font-size: 1.25rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                color: #f1f5f9;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .contact-link {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                padding: 0.75rem;
                margin-bottom: 0.75rem;
                background: rgba(255, 255, 255, 0.02);
                border-radius: 8px;
                text-decoration: none;
                color: #cbd5e1;
                transition: all 0.3s ease;
                border: 1px solid transparent;
            }
            
            .contact-link:hover {
                background: rgba(255, 174, 66, 0.1);
                border-color: rgba(255, 174, 66, 0.3);
                color: #FFAE42;
                transform: translateX(4px);
            }
            
            .contact-icon {
                font-size: 1.25rem;
                min-width: 24px;
            }
            
            .contact-text {
                font-size: 0.95rem;
            }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
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

