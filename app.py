import streamlit as st
import numpy as np
from statistics import mode, StatisticsError

from systems import d6_pool, d10_pool
import charts

st.set_page_config(page_title="Dice Pool Simulator", page_icon="\U0001F3B2", layout="wide")
st.title("\U0001F3B2 Dice Pool Probability Simulator")

# Shared CSS for smaller info box text
st.markdown(
    "<style>div[data-testid='stAlert'] p { font-size: 0.78rem; }</style>",
    unsafe_allow_html=True,
)


# ── Shared output helpers ────────────────────────────────────────────────────
def render_results(marks, complication, dr, n_trials):
    """Render metrics, charts, and stats table from simulation results."""
    dos = marks - dr
    success = dos >= 0

    col1, col2 = st.columns(2)
    col1.metric("Chance of Success", f"{np.mean(success) * 100:.2f}%")
    col2.metric("Chance of Complication", f"{np.mean(complication) * 100:.2f}%")

    st.plotly_chart(charts.dos_histogram(dos, n_trials), use_container_width=True)
    st.plotly_chart(charts.outcome_breakdown(success, complication), use_container_width=True)

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
# D6 POOL TAB
# ══════════════════════════════════════════════════════════════════════════════
def render_d6_tab():
    with st.sidebar:
        st.header("D6 Pool Settings")
        pool_size = st.slider("Dice Pool Size", 1, 30, 6, key="d6_pool")
        dr = st.slider("Difficulty Rating (DR)", 0, 20, 3, key="d6_dr")

        st.header("Advantage / Disadvantage")
        advantage_label = st.radio(
            "Advantage Level",
            d6_pool.ADVANTAGE_LEVELS,
            index=d6_pool.ADVANTAGE_DEFAULT_INDEX,
            key="d6_adv",
        )

        st.header("Postures")
        safe = st.checkbox("Safe — reroll 1s once", key="d6_safe")
        blessed = st.checkbox("Blessed — 6s produce 2 marks", key="d6_blessed")
        cursed = st.checkbox("Cursed — 1s cancel a mark", key="d6_cursed")
        unnatural = st.checkbox("Unnatural — Blessed + Cursed combined", key="d6_unnatural")
        if unnatural:
            blessed = True
            cursed = True

        st.header("Calculation Variants")
        comp_label = st.selectbox(
            "Complication Method",
            list(d6_pool.COMPLICATION_METHODS.keys()),
            help=d6_pool.COMPLICATION_HELP,
            key="d6_comp",
        )
        comp_method = d6_pool.COMPLICATION_METHODS[comp_label]
        st.info(d6_pool.COMPLICATION_DESCRIPTIONS[comp_method])

        st.header("Simulation")
        n_trials = st.slider("Number of trials", 100_000, 1_000_000, 100_000,
                             step=100_000, key="d6_trials")
        st.caption(
            "Postures stack freely. Advantage/Disadvantage sets the mark "
            "threshold; Safe, Blessed, Cursed, and Unnatural layer on top."
        )

    # Derived values
    threshold = d6_pool.THRESHOLD_MAP[advantage_label]
    posture_summary = d6_pool.posture_summary_label(
        advantage_label, safe, blessed, cursed, unnatural,
    )
    st.subheader(f"{pool_size}d6 {posture_summary} vs DR {dr}")

    # Main simulation
    rng = np.random.default_rng()
    marks, complication = d6_pool.simulate(
        pool_size, threshold, safe, blessed, cursed, n_trials, rng, comp_method, dr,
    )
    render_results(marks, complication, dr, n_trials)

    # Comparison charts
    st.divider()
    st.header("Posture & Advantage Comparisons")
    st.caption(
        f"Each line simulates {d6_pool.COMPARE_TRIALS:,} trials per pool size "
        f"(1\u201330) at DR {dr}. Independent of sidebar posture selection."
    )
    comp_rng = np.random.default_rng(42)

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

    heat = d6_pool.heatmap_data(threshold, safe, blessed, cursed, comp_method, comp_rng)
    st.subheader("Pool Size vs Difficulty — Heatmap")
    st.caption("Hover to read exact probabilities. Uses your current posture settings.")
    st.plotly_chart(charts.success_heatmap(
        d6_pool.POOL_RANGE, d6_pool.DR_RANGE, heat,
        f"Success Probability Heatmap ({posture_summary})",
    ), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# D10 POOL TAB
# ══════════════════════════════════════════════════════════════════════════════
def render_d10_tab():
    with st.sidebar:
        st.header("D10 Pool Settings")
        pool_size = st.slider("Dice Pool Size", 1, 30, 6, key="d10_pool")
        dr = st.slider("Difficulty Rating (DR)", 0, 20, 3, key="d10_dr")

        st.header("Mark Threshold")
        base_thresh = st.slider(
            "Base threshold (marks on roll >=)",
            2, 9, d10_pool.DEFAULT_THRESHOLD, key="d10_base_thresh",
        )
        adv_shift = st.slider(
            "Advantage / Disadvantage",
            d10_pool.ADVANTAGE_RANGE[0], d10_pool.ADVANTAGE_RANGE[1], 0,
            key="d10_adv",
            help="Positive = Advantage (lowers threshold). Negative = Disadvantage (raises threshold).",
        )
        threshold = base_thresh - adv_shift
        threshold = max(1, min(10, threshold))
        st.caption(f"Effective threshold: **{threshold}+** on d10")

        st.header("Postures")
        safe_label = st.selectbox("Safe", list(d10_pool.SAFE_OPTIONS.keys()), key="d10_safe")
        safe_level = d10_pool.SAFE_OPTIONS[safe_label]

        blessed_label = st.selectbox(
            "Blessed", list(d10_pool.BLESSED_OPTIONS.keys()), key="d10_blessed",
        )
        blessed_thresh = d10_pool.BLESSED_OPTIONS[blessed_label]
        st.info(d10_pool.BLESSED_DESCRIPTIONS[blessed_thresh])

        cursed_label = st.selectbox(
            "Cursed", list(d10_pool.CURSED_OPTIONS.keys()), key="d10_cursed",
        )
        cursed_thresh = d10_pool.CURSED_OPTIONS[cursed_label]
        st.info(d10_pool.CURSED_DESCRIPTIONS[cursed_thresh])

        st.header("Risk Die")
        risk_label = st.selectbox(
            "Risk Die Complication",
            list(d10_pool.RISK_DIE_OPTIONS.keys()),
            help=d10_pool.RISK_DIE_HELP,
            key="d10_risk",
        )
        risk_die_thresh = d10_pool.RISK_DIE_OPTIONS[risk_label]
        st.info(d10_pool.RISK_DIE_DESCRIPTIONS[risk_die_thresh])

        st.header("Simulation")
        n_trials = st.slider("Number of trials", 100_000, 1_000_000, 100_000,
                             step=100_000, key="d10_trials")

    # Derived values
    posture_summary = d10_pool.posture_summary_label(
        threshold, safe_level, blessed_thresh, cursed_thresh, risk_die_thresh,
    )
    st.subheader(f"{pool_size}d10 {posture_summary} vs DR {dr}")

    # Main simulation
    rng = np.random.default_rng()
    marks, complication = d10_pool.simulate(
        pool_size, threshold, safe_level, blessed_thresh, cursed_thresh,
        risk_die_thresh, n_trials, rng, dr,
    )
    render_results(marks, complication, dr, n_trials)

    # Comparison charts
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
        "Off": "#636EFA", "Blessed 10 (10s count double)": "#00CC96",
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
        "Off": "#636EFA", "Cursed 1 (1s cancel a mark)": "#FFA15A",
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


# ── Tab selection ────────────────────────────────────────────────────────────
tab_d6, tab_d10 = st.tabs(["D6 Pool", "D10 Pool"])

with tab_d6:
    render_d6_tab()

with tab_d10:
    render_d10_tab()
