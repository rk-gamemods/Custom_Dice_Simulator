import streamlit as st
import numpy as np

from systems import d6_pool
from shared import INFO_CSS, render_results
import charts

st.set_page_config(page_title="D6 Pool Simulator", page_icon="\U0001F3B2", layout="wide")
st.title("\U0001F3B2 D6 Dice Pool Simulator")
st.markdown(INFO_CSS, unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.page_link("app.py", label="Home")
    st.page_link("pages/2_D10_Pool.py", label="D10 Pool")
    st.divider()
    st.header("D6 Pool Settings")
    pool_size = st.slider("Dice Pool Size", 1, 30, 6)
    dr = st.slider("Difficulty Rating (DR)", 0, 20, 3)

    st.header("Advantage / Disadvantage")
    advantage_label = st.radio(
        "Advantage Level",
        d6_pool.ADVANTAGE_LEVELS,
        index=d6_pool.ADVANTAGE_DEFAULT_INDEX,
    )

    st.header("Postures")
    safe = st.checkbox("Safe — reroll 1s once")
    blessed = st.checkbox("Blessed — 6s produce 2 marks")
    cursed = st.checkbox("Cursed — 1s cancel a mark")
    unnatural = st.checkbox("Unnatural — Blessed + Cursed combined")
    if unnatural:
        blessed = True
        cursed = True

    st.header("Calculation Variants")
    comp_label = st.selectbox(
        "Complication Method",
        list(d6_pool.COMPLICATION_METHODS.keys()),
        help=d6_pool.COMPLICATION_HELP,
    )
    comp_method = d6_pool.COMPLICATION_METHODS[comp_label]
    st.info(d6_pool.COMPLICATION_DESCRIPTIONS[comp_method])

    st.header("Simulation")
    n_trials = st.slider("Number of trials", 100_000, 1_000_000, 100_000, step=100_000)
    st.caption(
        "Postures stack freely. Advantage/Disadvantage sets the mark "
        "threshold; Safe, Blessed, Cursed, and Unnatural layer on top."
    )

# ── Derived values ───────────────────────────────────────────────────────────
threshold = d6_pool.THRESHOLD_MAP[advantage_label]
posture_summary = d6_pool.posture_summary_label(
    advantage_label, safe, blessed, cursed, unnatural,
)
st.subheader(f"{pool_size}d6 {posture_summary} vs DR {dr}")

# ── Main simulation ──────────────────────────────────────────────────────────
rng = np.random.default_rng()
marks, complication = d6_pool.simulate(
    pool_size, threshold, safe, blessed, cursed, n_trials, rng, comp_method, dr,
)
render_results(marks, complication, dr, n_trials)

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON CHARTS
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Posture & Advantage Comparisons")
st.caption(
    f"Each line simulates {d6_pool.COMPARE_TRIALS:,} trials per pool size "
    f"(1\u201330) at DR {dr}. Independent of sidebar posture selection."
)
comp_rng = np.random.default_rng(42)

# Posture comparison
posture_succ, posture_comp = {}, {}
for name, (s, bl, cu) in d6_pool.POSTURE_CONFIGS.items():
    sc, cc = d6_pool.success_curve(threshold, s, bl, cu, dr, comp_method, comp_rng)
    posture_succ[name] = sc
    posture_comp[name] = cc

st.subheader("Posture Comparison")
st.plotly_chart(charts.comparison_lines(
    d6_pool.POOL_RANGE, posture_succ, d6_pool.POSTURE_COLORS,
    f"Success % by Posture ({advantage_label} threshold, DR {dr})",
    "Success Probability (%)", hline_y=50,
), use_container_width=True)
st.plotly_chart(charts.comparison_lines(
    d6_pool.POOL_RANGE, posture_comp, d6_pool.POSTURE_COLORS,
    f"Complication % by Posture ({advantage_label} threshold, DR {dr})",
    "Complication Probability (%)",
), use_container_width=True)

# Advantage comparison
adv_curves = {}
for name, thresh_val in d6_pool.THRESHOLD_MAP.items():
    sc, _ = d6_pool.success_curve(thresh_val, safe, blessed, cursed, dr, comp_method, comp_rng)
    adv_curves[name] = sc

st.subheader("Advantage Level Comparison")
st.plotly_chart(charts.comparison_lines(
    d6_pool.POOL_RANGE, adv_curves, d6_pool.ADVANTAGE_COLORS,
    f"Success % by Advantage Level (DR {dr}, {posture_summary} postures)",
    "Success Probability (%)", hline_y=50,
), use_container_width=True)

# Heatmap
heat = d6_pool.heatmap_data(threshold, safe, blessed, cursed, comp_method, comp_rng)
st.subheader("Pool Size vs Difficulty — Heatmap")
st.caption("Hover to read exact probabilities. Uses your current posture settings.")
st.plotly_chart(charts.success_heatmap(
    d6_pool.POOL_RANGE, d6_pool.DR_RANGE, heat,
    f"Success Probability Heatmap ({posture_summary})",
), use_container_width=True)
