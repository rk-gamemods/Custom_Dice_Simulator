import streamlit as st
import numpy as np
import plotly.graph_objects as go
from statistics import mode, StatisticsError

st.set_page_config(page_title="Dice Pool Simulator", page_icon="\U0001F3B2", layout="wide")
st.title("\U0001F3B2 Dice Pool Probability Simulator")

# ── Shared simulation helper ────────────────────────────────────────────────
POOL_RANGE = np.arange(1, 31)
COMPARE_TRIALS = 50_000


COMPLICATION_METHODS = {
    "Standard (strict): 1s > Marks": "standard",
    "Standard (loose): 1s >= Marks": "standard_loose",
    "Broad (strict): 1s + 4s > Marks": "broad_strict",
    "Broad (loose): 1s + 4s >= Marks": "broad_loose",
    "Proportional: 1s >= half Marks (round up)": "proportional",
    "Split: on fail 1s > Marks, on pass 4s > DoS": "split",
}


def simulate(n_dice, thresh, is_safe, is_blessed, is_cursed, trials, rng,
             comp_method="standard", difficulty=0):
    """Run a single simulation config and return (marks, complication) arrays."""
    rolls = rng.integers(1, 7, size=(trials, n_dice))
    if is_safe:
        mask = rolls == 1
        rolls = np.where(mask, rng.integers(1, 7, size=(trials, n_dice)), rolls)
    m = np.sum(rolls >= thresh, axis=1).astype(np.int64)
    if is_blessed:
        m += np.sum(rolls == 6, axis=1)
    if is_cursed:
        m -= np.sum(rolls == 1, axis=1)
    ones = np.sum(rolls == 1, axis=1)
    fours = np.sum(rolls == 4, axis=1)
    if comp_method == "broad_strict":
        comp = (ones + fours) > m
    elif comp_method == "broad_loose":
        comp = (ones + fours) >= m
    elif comp_method == "proportional":
        half_marks = np.ceil(np.maximum(m, 0) / 2).astype(np.int64)
        comp = ones >= half_marks
    elif comp_method == "split":
        passed = m >= difficulty
        dos = m - difficulty
        comp = np.where(passed, fours > dos, ones > m)
    elif comp_method == "standard_loose":
        comp = ones >= m
    else:
        comp = ones > m
    return m, comp

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Dice Pool Settings")
    pool_size = st.slider("Dice Pool Size", 1, 30, 6)
    dr = st.slider("Difficulty Rating (DR)", 0, 20, 3)

    st.header("Advantage / Disadvantage")
    advantage_label = st.radio(
        "Advantage Level",
        ["Double Disadvantage", "Disadvantage", "Normal", "Advantage", "Double Advantage"],
        index=2,
    )

    st.header("Postures")
    safe = st.checkbox("Safe — reroll 1s once")
    blessed = st.checkbox("Blessed — 6s produce 2 marks")
    cursed = st.checkbox("Cursed — 1s cancel a mark")
    unnatural = st.checkbox("Unnatural — Blessed + Cursed combined")

    # Unnatural forces both Blessed and Cursed on
    if unnatural:
        blessed = True
        cursed = True

    st.header("Calculation Variants")
    comp_label = st.selectbox(
        "Complication Method",
        list(COMPLICATION_METHODS.keys()),
        help="Standard: 1s > Marks. Broad: (1s + 4s) > Marks — complications trigger more often and spread across success/failure.",
    )
    comp_method = COMPLICATION_METHODS[comp_label]

    st.header("Simulation")
    n_trials = st.slider("Number of trials", 100_000, 1_000_000, 100_000, step=100_000)

    st.caption(
        "Postures stack freely. Advantage/Disadvantage sets the mark "
        "threshold; Safe, Blessed, Cursed, and Unnatural layer on top."
    )

# ── Threshold mapping ────────────────────────────────────────────────────────
THRESHOLD_MAP = {
    "Double Advantage": 2,
    "Advantage": 3,
    "Normal": 4,
    "Disadvantage": 5,
    "Double Disadvantage": 6,
}
threshold = THRESHOLD_MAP[advantage_label]

# ── Build active-posture label ───────────────────────────────────────────────
posture_parts = []
if advantage_label != "Normal":
    posture_parts.append(advantage_label)
if safe:
    posture_parts.append("Safe")
if unnatural:
    posture_parts.append("Unnatural")
else:
    if blessed:
        posture_parts.append("Blessed")
    if cursed:
        posture_parts.append("Cursed")
posture_summary = " + ".join(posture_parts) if posture_parts else "Normal"
st.subheader(f"{pool_size}d6 {posture_summary} vs DR {dr}")

# ── Simulation (vectorised with NumPy) ───────────────────────────────────────
rng = np.random.default_rng()
marks, complication = simulate(pool_size, threshold, safe, blessed, cursed, n_trials, rng, comp_method, dr)

# Degrees of Success (positive) / Failure (negative)
dos = marks - dr
success = dos >= 0

# ── Metrics ──────────────────────────────────────────────────────────────────
chance_success = np.mean(success) * 100
chance_complication = np.mean(complication) * 100

col1, col2 = st.columns(2)
col1.metric("Chance of Success", f"{chance_success:.2f}%")
col2.metric("Chance of Complication", f"{chance_complication:.2f}%")

# ── Primary chart: DoS distribution ──────────────────────────────────────────
dos_min, dos_max = int(dos.min()), int(dos.max())
bins = np.arange(dos_min, dos_max + 2) - 0.5
counts, edges = np.histogram(dos, bins=bins)
centres = ((edges[:-1] + edges[1:]) / 2).astype(int)
colours = ["#d62728" if c < 0 else "#2ca02c" for c in centres]

fig_dos = go.Figure(
    go.Bar(
        x=centres,
        y=counts / n_trials * 100,
        marker_color=colours,
        hovertemplate="DoS %{x}: %{y:.2f}%<extra></extra>",
    )
)
fig_dos.update_layout(
    title="Distribution of Degrees of Success / Failure",
    xaxis_title="Degrees of Success",
    yaxis_title="Probability (%)",
    bargap=0.05,
)
fig_dos.add_vline(x=-0.5, line_dash="dash", line_color="grey", opacity=0.5)
st.plotly_chart(fig_dos, use_container_width=True)

# ── Secondary chart: success with / without complication ─────────────────────
success_no_comp = np.mean(success & ~complication) * 100
success_with_comp = np.mean(success & complication) * 100
failure_with_comp = np.mean(~success & complication) * 100
failure_no_comp = np.mean(~success & ~complication) * 100

fig_comp = go.Figure(
    go.Bar(
        x=[
            "Success",
            "Success + Complication",
            "Failure",
            "Failure + Complication",
        ],
        y=[success_no_comp, success_with_comp, failure_no_comp, failure_with_comp],
        marker_color=["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"],
        text=[
            f"{success_no_comp:.2f}%",
            f"{success_with_comp:.2f}%",
            f"{failure_no_comp:.2f}%",
            f"{failure_with_comp:.2f}%",
        ],
        textposition="auto",
    )
)
fig_comp.update_layout(
    title="Outcome Breakdown",
    yaxis_title="Probability (%)",
)
st.plotly_chart(fig_comp, use_container_width=True)

# ── Data table ───────────────────────────────────────────────────────────────
marks_int = marks.astype(int)
try:
    marks_mode = mode(marks_int)
except StatisticsError:
    marks_mode = "N/A"

st.subheader("Marks Statistics")
st.table(
    {
        "Statistic": ["Mean", "Median", "Mode"],
        "Value": [
            f"{np.mean(marks):.2f}",
            f"{np.median(marks):.1f}",
            str(marks_mode),
        ],
    }
)

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON CHARTS — these run independent sims across pool sizes
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Posture & Advantage Comparisons")
st.caption(
    f"Each line simulates {COMPARE_TRIALS:,} trials per pool size (1–30) "
    f"at DR {dr}. These charts are independent of your sidebar posture selection."
)

comp_rng = np.random.default_rng(42)  # fixed seed for stable comparisons


def success_curve(thresh, is_safe, is_blessed, is_cursed):
    """Return arrays of (success%, complication%) for pool sizes 1-30."""
    succ = np.empty(len(POOL_RANGE))
    comp = np.empty(len(POOL_RANGE))
    for i, n in enumerate(POOL_RANGE):
        m, cc = simulate(n, thresh, is_safe, is_blessed, is_cursed, COMPARE_TRIALS, comp_rng, comp_method, dr)
        succ[i] = np.mean((m - dr) >= 0) * 100
        comp[i] = np.mean(cc) * 100
    return succ, comp


# ── Posture comparison (using current advantage threshold) ───────────────────
POSTURE_CONFIGS = {
    "Normal":    (False, False, False),
    "Safe":      (True,  False, False),
    "Blessed":   (False, True,  False),
    "Cursed":    (False, False, True),
    "Unnatural": (False, True,  True),
    "Safe + Blessed": (True, True, False),
    "Safe + Unnatural": (True, True, True),
}
POSTURE_COLORS = {
    "Normal": "#636EFA",
    "Safe": "#00CC96",
    "Blessed": "#FFA15A",
    "Cursed": "#EF553B",
    "Unnatural": "#AB63FA",
    "Safe + Blessed": "#19D3F3",
    "Safe + Unnatural": "#FF6692",
}

fig_posture_succ = go.Figure()
fig_posture_comp = go.Figure()

for name, (s, bl, cu) in POSTURE_CONFIGS.items():
    sc, cc = success_curve(threshold, s, bl, cu)
    fig_posture_succ.add_trace(go.Scatter(
        x=POOL_RANGE, y=sc, mode="lines", name=name,
        line=dict(color=POSTURE_COLORS[name]),
        hovertemplate="%{x}d6: %{y:.2f}%<extra>" + name + "</extra>",
    ))
    fig_posture_comp.add_trace(go.Scatter(
        x=POOL_RANGE, y=cc, mode="lines", name=name,
        line=dict(color=POSTURE_COLORS[name]),
        hovertemplate="%{x}d6: %{y:.2f}%<extra>" + name + "</extra>",
    ))

fig_posture_succ.update_layout(
    title=f"Success % by Posture ({advantage_label} threshold, DR {dr})",
    xaxis_title="Pool Size (d6)",
    yaxis_title="Success Probability (%)",
    hovermode="x unified",
)
fig_posture_succ.add_hline(y=50, line_dash="dot", line_color="grey", opacity=0.4)

fig_posture_comp.update_layout(
    title=f"Complication % by Posture ({advantage_label} threshold, DR {dr})",
    xaxis_title="Pool Size (d6)",
    yaxis_title="Complication Probability (%)",
    hovermode="x unified",
)

st.subheader("Posture Comparison")
st.plotly_chart(fig_posture_succ, use_container_width=True)
st.plotly_chart(fig_posture_comp, use_container_width=True)

# ── Advantage comparison (using current postures) ────────────────────────────
ADV_LEVELS = {
    "Double Disadvantage": 6,
    "Disadvantage": 5,
    "Normal": 4,
    "Advantage": 3,
    "Double Advantage": 2,
}
ADV_COLORS = {
    "Double Disadvantage": "#EF553B",
    "Disadvantage": "#FFA15A",
    "Normal": "#636EFA",
    "Advantage": "#00CC96",
    "Double Advantage": "#19D3F3",
}

fig_adv = go.Figure()
for name, thresh_val in ADV_LEVELS.items():
    sc, _ = success_curve(thresh_val, safe, blessed, cursed)
    fig_adv.add_trace(go.Scatter(
        x=POOL_RANGE, y=sc, mode="lines", name=name,
        line=dict(color=ADV_COLORS[name]),
        hovertemplate="%{x}d6: %{y:.2f}%<extra>" + name + "</extra>",
    ))

fig_adv.update_layout(
    title=f"Success % by Advantage Level (DR {dr}, {posture_summary} postures)",
    xaxis_title="Pool Size (d6)",
    yaxis_title="Success Probability (%)",
    hovermode="x unified",
)
fig_adv.add_hline(y=50, line_dash="dot", line_color="grey", opacity=0.4)

st.subheader("Advantage Level Comparison")
st.plotly_chart(fig_adv, use_container_width=True)

# ── Heatmap: Pool Size vs DR → Success % ────────────────────────────────────
DR_RANGE = np.arange(0, 21)
heat_data = np.empty((len(DR_RANGE), len(POOL_RANGE)))

for j, n in enumerate(POOL_RANGE):
    m, _ = simulate(n, threshold, safe, blessed, cursed, COMPARE_TRIALS, comp_rng, comp_method, dr)
    for i, d in enumerate(DR_RANGE):
        heat_data[i, j] = np.mean(m >= d) * 100

fig_heat = go.Figure(go.Heatmap(
    z=heat_data,
    x=POOL_RANGE,
    y=DR_RANGE,
    colorscale="RdYlGn",
    zmin=0,
    zmax=100,
    colorbar=dict(title="Success %"),
    hovertemplate="Pool: %{x}d6<br>DR: %{y}<br>Success: %{z:.1f}%<extra></extra>",
))
fig_heat.update_layout(
    title=f"Success Probability Heatmap ({posture_summary})",
    xaxis_title="Pool Size (d6)",
    yaxis_title="Difficulty Rating (DR)",
    yaxis=dict(dtick=1),
    xaxis=dict(dtick=1),
)

st.subheader("Pool Size vs Difficulty — Heatmap")
st.caption("Hover to read exact probabilities. Uses your current posture settings.")
st.plotly_chart(fig_heat, use_container_width=True)
