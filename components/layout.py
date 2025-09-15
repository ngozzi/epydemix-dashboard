import streamlit as st

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

def show_fixed_logo():
    st.markdown(
        """
        <style>
        .fixed-logo {
            position: fixed;
            bottom: 60px;   /* adjust vertical position */
            right: 20px;    /* adjust horizontal position */
            width: 100px;
            z-index: 100;
        }
        </style>
        <a href="https://www.isi.it" target="_blank">
            <img src="https://www.isi.it/wp-content/uploads/2023/11/isi-logo-white.svg" class="fixed-logo">
        </a>
        """,
        unsafe_allow_html=True
    )
