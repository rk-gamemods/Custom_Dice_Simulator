import streamlit as st
import numpy as np
import plotly.graph_objects as go
from statistics import mode, StatisticsError

st.set_page_config(page_title="Dice Pool Simulator", page_icon="\U0001F3B2", layout="wide")
st.title("\U0001F3B2 Dice Pool Probability Simulator")

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

# Shape: (n_trials, pool_size)
rolls = rng.integers(1, 7, size=(n_trials, pool_size))

# Safe: reroll any 1s once
if safe:
    ones_mask = rolls == 1
    rerolls = rng.integers(1, 7, size=(n_trials, pool_size))
    rolls = np.where(ones_mask, rerolls, rolls)

# Base marks: each die >= threshold scores 1 mark
marks = np.sum(rolls >= threshold, axis=1).astype(np.int64)

# Blessed: each 6 adds +1 extra mark (so a 6 = 2 marks total)
if blessed:
    marks += np.sum(rolls == 6, axis=1)

# Cursed: each 1 cancels a mark (-1)
if cursed:
    marks -= np.sum(rolls == 1, axis=1)

# Complications: more 1s in the final pool than total marks
ones_count = np.sum(rolls == 1, axis=1)
complication = ones_count > marks

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
