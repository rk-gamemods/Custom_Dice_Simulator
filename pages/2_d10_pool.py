import streamlit as st
import numpy as np

from systems import d10_pool
from shared import INFO_CSS, render_results
import charts

st.set_page_config(page_title="D10 Pool Simulator", page_icon="\U0001F3B2", layout="wide")
st.title("\U0001F3B2 D10 Dice Pool Simulator")
st.markdown(INFO_CSS, unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.page_link("app.py", label="Home")
    st.page_link("pages/1_d6_pool.py", label="D6 Pool")
    st.divider()
    st.header("D10 Pool Settings")
    pool_size = st.slider("Dice Pool Size", 1, 30, 6)
    dr = st.slider("Difficulty Rating (DR)", 0, 20, 3)

    st.header("Mark Threshold")
    base_thresh = st.slider(
        "Base threshold (marks on roll >=)",
        2, 9, d10_pool.DEFAULT_THRESHOLD,
    )
    adv_shift = st.slider(
        "Advantage / Disadvantage",
        d10_pool.ADVANTAGE_RANGE[0], d10_pool.ADVANTAGE_RANGE[1], 0,
        help="Positive = Advantage (lowers threshold). Negative = Disadvantage (raises threshold).",
    )
    threshold = max(1, min(10, base_thresh - adv_shift))
    st.caption(f"Effective threshold: **{threshold}+** on d10")

    st.header("Postures")
    safe_label = st.selectbox("Safe", list(d10_pool.SAFE_OPTIONS.keys()))
    safe_level = d10_pool.SAFE_OPTIONS[safe_label]

    blessed_label = st.selectbox("Blessed", list(d10_pool.BLESSED_OPTIONS.keys()))
    blessed_thresh = d10_pool.BLESSED_OPTIONS[blessed_label]
    st.info(d10_pool.BLESSED_DESCRIPTIONS[blessed_thresh])

    cursed_label = st.selectbox("Cursed", list(d10_pool.CURSED_OPTIONS.keys()))
    cursed_thresh = d10_pool.CURSED_OPTIONS[cursed_label]
    st.info(d10_pool.CURSED_DESCRIPTIONS[cursed_thresh])

    st.header("Risk Die")
    risk_label = st.selectbox(
        "Risk Die Complication",
        list(d10_pool.RISK_DIE_OPTIONS.keys()),
        help=d10_pool.RISK_DIE_HELP,
    )
    risk_die_thresh = d10_pool.RISK_DIE_OPTIONS[risk_label]
    st.info(d10_pool.RISK_DIE_DESCRIPTIONS[risk_die_thresh])

    st.header("Simulation")
    n_trials = st.slider("Number of trials", 100_000, 1_000_000, 100_000, step=100_000)

# ── Derived values ───────────────────────────────────────────────────────────
posture_summary = d10_pool.posture_summary_label(
    threshold, safe_level, blessed_thresh, cursed_thresh, risk_die_thresh,
)
st.subheader(f"{pool_size}d10 {posture_summary} vs DR {dr}")

# ── Main simulation ──────────────────────────────────────────────────────────
rng = np.random.default_rng()
marks, complication = d10_pool.simulate(
    pool_size, threshold, safe_level, blessed_thresh, cursed_thresh,
    risk_die_thresh, n_trials, rng, dr,
)
render_results(marks, complication, dr, n_trials)

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON CHARTS
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Posture Comparisons")
st.caption(
    f"Each line simulates {d10_pool.COMPARE_TRIALS:,} trials per pool size "
    f"(1\u201330) at DR {dr}."
)
comp_rng = np.random.default_rng(42)

# Blessed tier comparison
blessed_succ = {}
for label, bt in d10_pool.BLESSED_OPTIONS.items():
    sc, _ = d10_pool.success_curve(
        threshold, safe_level, bt, cursed_thresh, risk_die_thresh, dr, comp_rng,
    )
    blessed_succ[label] = sc

blessed_colors = {
    "Off": "#636EFA",
    "Blessed 10 (10s count double)": "#00CC96",
    "Blessed 9+ (9s and 10s count double)": "#FFA15A",
    "Blessed 8+ (8, 9, 10s count double)": "#EF553B",
}
st.subheader("Blessed Tier Comparison")
st.plotly_chart(charts.comparison_lines(
    d10_pool.POOL_RANGE, blessed_succ, blessed_colors,
    f"Success % by Blessed Tier (threshold {threshold}+, DR {dr})",
    "Success Probability (%)", hline_y=50,
), use_container_width=True)

# Cursed tier comparison
cursed_succ = {}
for label, ct in d10_pool.CURSED_OPTIONS.items():
    sc, _ = d10_pool.success_curve(
        threshold, safe_level, blessed_thresh, ct, risk_die_thresh, dr, comp_rng,
    )
    cursed_succ[label] = sc

cursed_colors = {
    "Off": "#636EFA",
    "Cursed 1 (1s cancel a mark)": "#FFA15A",
    "Cursed 2 (1s and 2s cancel a mark)": "#EF553B",
    "Cursed 3 (1s, 2s, and 3s cancel a mark)": "#9467bd",
}
st.subheader("Cursed Tier Comparison")
st.plotly_chart(charts.comparison_lines(
    d10_pool.POOL_RANGE, cursed_succ, cursed_colors,
    f"Success % by Cursed Tier (threshold {threshold}+, DR {dr})",
    "Success Probability (%)", hline_y=50,
), use_container_width=True)

# Advantage sweep comparison
adv_succ = {}
adv_colors = {}
color_scale = ["#EF553B", "#FFA15A", "#FECB52", "#636EFA", "#00CC96", "#19D3F3", "#AB63FA"]
for i, shift in enumerate(range(-3, 4)):
    t = max(1, min(10, base_thresh - shift))
    label = f"Threshold {t}+ (shift {shift:+d})"
    sc, _ = d10_pool.success_curve(
        t, safe_level, blessed_thresh, cursed_thresh, risk_die_thresh, dr, comp_rng,
    )
    adv_succ[label] = sc
    adv_colors[label] = color_scale[i]

st.subheader("Advantage / Disadvantage Comparison")
st.plotly_chart(charts.comparison_lines(
    d10_pool.POOL_RANGE, adv_succ, adv_colors,
    f"Success % by Threshold (base {base_thresh}+, DR {dr})",
    "Success Probability (%)", hline_y=50,
), use_container_width=True)

# Heatmap
heat = d10_pool.heatmap_data(
    threshold, safe_level, blessed_thresh, cursed_thresh, risk_die_thresh, comp_rng,
)
st.subheader("Pool Size vs Difficulty — Heatmap")
st.caption("Hover to read exact probabilities. Uses your current posture settings.")
st.plotly_chart(charts.success_heatmap(
    d10_pool.POOL_RANGE, d10_pool.DR_RANGE, heat,
    f"Success Probability Heatmap ({posture_summary})",
), use_container_width=True)
