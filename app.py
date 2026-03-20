import streamlit as st

st.set_page_config(page_title="Dice Pool Simulator", page_icon="\U0001F3B2", layout="wide")
st.title("\U0001F3B2 Dice Pool Probability Simulator")

st.markdown(
    "Select a dice system from the sidebar to get started."
)

st.markdown("""
### Available Systems

**D6 Pool** — Six-sided dice with marks on 4+, Advantage/Disadvantage tiers,
Safe/Blessed/Cursed/Unnatural postures, and multiple complication calculation methods.

**D10 Pool** — Ten-sided dice with configurable mark threshold, tiered
Blessed/Cursed, Advantage as a numeric slider, and a Risk Die complication system.
""")
