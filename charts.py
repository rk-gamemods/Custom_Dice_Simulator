"""Reusable chart builders for dice simulation visualizations."""

import numpy as np
import plotly.graph_objects as go


def dos_histogram(dos_values, n_trials):
    """Red/green bar chart of Degrees of Success distribution."""
    dos_min, dos_max = int(dos_values.min()), int(dos_values.max())
    bins = np.arange(dos_min, dos_max + 2) - 0.5
    counts, edges = np.histogram(dos_values, bins=bins)
    centres = ((edges[:-1] + edges[1:]) / 2).astype(int)
    colours = ["#d62728" if c < 0 else "#2ca02c" for c in centres]

    fig = go.Figure(go.Bar(
        x=centres,
        y=counts / n_trials * 100,
        marker_color=colours,
        hovertemplate="DoS %{x}: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Distribution of Degrees of Success / Failure",
        xaxis_title="Degrees of Success",
        yaxis_title="Probability (%)",
        bargap=0.05,
    )
    fig.add_vline(x=-0.5, line_dash="dash", line_color="grey", opacity=0.5)
    return fig


def outcome_breakdown(success, complication):
    """Four-bar chart: success/failure x complication."""
    s_nc = np.mean(success & ~complication) * 100
    s_c = np.mean(success & complication) * 100
    f_nc = np.mean(~success & ~complication) * 100
    f_c = np.mean(~success & complication) * 100

    fig = go.Figure(go.Bar(
        x=["Success", "Success + Complication", "Failure", "Failure + Complication"],
        y=[s_nc, s_c, f_nc, f_c],
        marker_color=["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"],
        text=[f"{v:.2f}%" for v in (s_nc, s_c, f_nc, f_c)],
        textposition="auto",
    ))
    fig.update_layout(title="Outcome Breakdown", yaxis_title="Probability (%)")
    return fig


def comparison_lines(pool_range, curves, colors, title, ylabel, hline_y=None):
    """Overlaid line chart for comparing multiple configurations.

    curves: dict of {name: y_values}
    colors: dict of {name: color_hex}
    """
    fig = go.Figure()
    for name, y in curves.items():
        fig.add_trace(go.Scatter(
            x=pool_range, y=y, mode="lines", name=name,
            line=dict(color=colors.get(name)),
            hovertemplate="%{x}d6: %{y:.2f}%<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Pool Size (d6)",
        yaxis_title=ylabel,
        hovermode="x unified",
    )
    if hline_y is not None:
        fig.add_hline(y=hline_y, line_dash="dot", line_color="grey", opacity=0.4)
    return fig


def success_heatmap(pool_range, dr_range, data, title):
    """Pool size vs DR heatmap colored by success probability."""
    fig = go.Figure(go.Heatmap(
        z=data,
        x=pool_range,
        y=dr_range,
        colorscale="RdYlGn",
        zmin=0,
        zmax=100,
        colorbar=dict(title="Success %"),
        hovertemplate="Pool: %{x}d6<br>DR: %{y}<br>Success: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Pool Size (d6)",
        yaxis_title="Difficulty Rating (DR)",
        yaxis=dict(dtick=1),
        xaxis=dict(dtick=1),
    )
    return fig
