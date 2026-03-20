import streamlit as st
import numpy as np
from statistics import mode, StatisticsError

from systems.d6_pool import (
    THRESHOLD_MAP, ADVANTAGE_LEVELS, ADVANTAGE_DEFAULT_INDEX,
    COMPLICATION_METHODS, COMPLICATION_DESCRIPTIONS, COMPLICATION_HELP,
    POSTURE_CONFIGS, POSTURE_COLORS, ADVANTAGE_COLORS,
    POOL_RANGE, DR_RANGE,
    simulate, success_curve, heatmap_data, posture_summary_label,
)
import charts

st.set_page_config(page_title="Dice Pool Simulator", page_icon="\U0001F3B2", layout="wide")
st.title("\U0001F3B2 Dice Pool Probability Simulator")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Dice Pool Settings")
    pool_size = st.slider("Dice Pool Size", 1, 30, 6)
    dr = st.slider("Difficulty Rating (DR)", 0, 20, 3)

    st.header("Advantage / Disadvantage")
    advantage_label = st.radio(
        "Advantage Level", ADVANTAGE_LEVELS, index=ADVANTAGE_DEFAULT_INDEX,
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
        list(COMPLICATION_METHODS.keys()),
        help=COMPLICATION_HELP,
    )
    comp_method = COMPLICATION_METHODS[comp_label]

    st.markdown(
        "<style>div[data-testid='stAlert'] p { font-size: 0.78rem; }</style>",
        unsafe_allow_html=True,
    )
    st.info(COMPLICATION_DESCRIPTIONS[comp_method])

    st.header("Simulation")
    n_trials = st.slider("Number of trials", 100_000, 1_000_000, 100_000, step=100_000)

    st.caption(
        "Postures stack freely. Advantage/Disadvantage sets the mark "
        "threshold; Safe, Blessed, Cursed, and Unnatural layer on top."
    )

# ── Derived values ───────────────────────────────────────────────────────────
threshold = THRESHOLD_MAP[advantage_label]
posture_summary = posture_summary_label(advantage_label, safe, blessed, cursed, unnatural)
st.subheader(f"{pool_size}d6 {posture_summary} vs DR {dr}")

# ── Main simulation ──────────────────────────────────────────────────────────
rng = np.random.default_rng()
marks, complication = simulate(
    pool_size, threshold, safe, blessed, cursed, n_trials, rng, comp_method, dr,
)
dos = marks - dr
success = dos >= 0

# ── Metrics ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
col1.metric("Chance of Success", f"{np.mean(success) * 100:.2f}%")
col2.metric("Chance of Complication", f"{np.mean(complication) * 100:.2f}%")

# ── Charts ───────────────────────────────────────────────────────────────────
st.plotly_chart(charts.dos_histogram(dos, n_trials), use_container_width=True)
st.plotly_chart(charts.outcome_breakdown(success, complication), use_container_width=True)

# ── Marks statistics ─────────────────────────────────────────────────────────
marks_int = marks.astype(int)
try:
    marks_mode = mode(marks_int)
except StatisticsError:
    marks_mode = "N/A"

st.subheader("Marks Statistics")
st.table({
    "Statistic": ["Mean", "Median", "Mode"],
    "Value": [f"{np.mean(marks):.2f}", f"{np.median(marks):.1f}", str(marks_mode)],
})

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON CHARTS
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Posture & Advantage Comparisons")
st.caption(
    f"Each line simulates {50_000:,} trials per pool size (1\u201330) "
    f"at DR {dr}. These charts are independent of your sidebar posture selection."
)

comp_rng = np.random.default_rng(42)

# ── Posture comparison ───────────────────────────────────────────────────────
posture_succ_curves, posture_comp_curves = {}, {}
for name, (s, bl, cu) in POSTURE_CONFIGS.items():
    sc, cc = success_curve(threshold, s, bl, cu, dr, comp_method, comp_rng)
    posture_succ_curves[name] = sc
    posture_comp_curves[name] = cc

st.subheader("Posture Comparison")
st.plotly_chart(charts.comparison_lines(
    POOL_RANGE, posture_succ_curves, POSTURE_COLORS,
    f"Success % by Posture ({advantage_label} threshold, DR {dr})",
    "Success Probability (%)", hline_y=50,
), use_container_width=True)
st.plotly_chart(charts.comparison_lines(
    POOL_RANGE, posture_comp_curves, POSTURE_COLORS,
    f"Complication % by Posture ({advantage_label} threshold, DR {dr})",
    "Complication Probability (%)",
), use_container_width=True)

# ── Advantage comparison ─────────────────────────────────────────────────────
adv_curves = {}
for name, thresh_val in THRESHOLD_MAP.items():
    sc, _ = success_curve(thresh_val, safe, blessed, cursed, dr, comp_method, comp_rng)
    adv_curves[name] = sc

st.subheader("Advantage Level Comparison")
st.plotly_chart(charts.comparison_lines(
    POOL_RANGE, adv_curves, ADVANTAGE_COLORS,
    f"Success % by Advantage Level (DR {dr}, {posture_summary} postures)",
    "Success Probability (%)", hline_y=50,
), use_container_width=True)

# ── Heatmap ──────────────────────────────────────────────────────────────────
heat = heatmap_data(threshold, safe, blessed, cursed, comp_method, comp_rng)

st.subheader("Pool Size vs Difficulty — Heatmap")
st.caption("Hover to read exact probabilities. Uses your current posture settings.")
st.plotly_chart(charts.success_heatmap(
    POOL_RANGE, DR_RANGE, heat, f"Success Probability Heatmap ({posture_summary})",
), use_container_width=True)
