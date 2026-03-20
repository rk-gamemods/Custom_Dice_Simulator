"""Shared UI helpers used across all dice system pages."""

import streamlit as st
import numpy as np
from statistics import mode, StatisticsError

import charts

# Shared CSS for smaller info box text
INFO_CSS = "<style>div[data-testid='stAlert'] p { font-size: 0.78rem; }</style>"


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
