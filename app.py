import streamlit as st

st.set_page_config(page_title="Dice Pool Simulator", page_icon="\U0001F3B2", layout="wide")
st.title("\U0001F3B2 Dice Pool Probability Simulator")

st.markdown("Select a dice system to get started.")

col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/1_D6_Pool.py", label="\U0001F3B2 D6 Pool", use_container_width=True)
    st.caption(
        "Six-sided dice with marks on 4+, Advantage/Disadvantage tiers, "
        "Safe/Blessed/Cursed/Unnatural postures, and multiple complication methods."
    )
with col2:
    st.page_link("pages/2_D10_Pool.py", label="\U0001F3B2 D10 Pool", use_container_width=True)
    st.caption(
        "Ten-sided dice with configurable mark threshold, tiered Blessed/Cursed, "
        "Advantage as a numeric slider, and a Risk Die complication system."
    )
