import streamlit as st

st.set_page_config(page_title="Dice Pool Simulator", page_icon="\U0001F3B2", layout="wide")

pg = st.navigation([
    st.Page("pages/1_d6_pool.py", title="D6 Pool", icon="\U0001F3B2"),
    st.Page("pages/2_d10_pool.py", title="D10 Pool", icon="\U0001F3B2"),
])

pg.run()
