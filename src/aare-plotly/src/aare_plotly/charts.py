from collections.abc import Sequence
from typing import TYPE_CHECKING

import plotly.colors
import plotly.graph_objects as go

from aare_plotly.color import adjust_lightness

if TYPE_CHECKING:
    import polars as pl
    import pandas as pd


def quantile_line_chart(
    df: "pd.DataFrame | pl.DataFrame",
    val_cols: str | Sequence[str],
    x_col: str = "run_ts",
    lower_quant: float = 0.10,
    inner_lower_quant: float = 0.25,
    inner_upper_quant: float = 0.75,
    upper_quant: float = 0.90,
    *,
    title: str,
    subtitle: str | None = None,
    yaxis_label: str,
    xaxis_label: str | None = None,
) -> go.Figure:
    """
    Plot multiple value columns with quantile bands.

    Expected quantile columns:
        <val_col>_q10.0
        <val_col>_q25.0
        etc.
    """

    if isinstance(val_cols, str):
        val_cols = [val_cols]

    x = df[x_col]

    base_colors = plotly.colors.DEFAULT_PLOTLY_COLORS

    fig = go.Figure()

    weak_bands = []
    strong_bands = []
    lines = []

    for idx, val_col in enumerate(val_cols):
        base_color = base_colors[idx % len(base_colors)]

        # Strong band = darker + more saturated
        strong_quant_color = adjust_lightness(
            base_color,
            lightness_mult=0.85,
            saturation_mult=1.15,
            alpha=0.5,
        )

        # Weak band = lighter + less saturated
        weak_quant_color = adjust_lightness(
            base_color,
            lightness_mult=1.35,
            saturation_mult=0.75,
            alpha=0.5,
        )

        upper_q_col = f"{val_col}_q{upper_quant * 100}"
        lower_q_col = f"{val_col}_q{lower_quant * 100}"
        inner_upper_q_col = f"{val_col}_q{inner_upper_quant * 100}"
        inner_lower_q_col = f"{val_col}_q{inner_lower_quant * 100}"

        # Outer quantile region
        weak_bands.append(
            go.Scatter(
                name=f"{val_col} Q {upper_quant:.0%}",
                x=x,
                y=df[upper_q_col],
                mode="lines",
                line=dict(width=0, color=weak_quant_color),
                legendgroup=val_col,
                showlegend=False,
            )
        )

        weak_bands.append(
            go.Scatter(
                name=f"{val_col} Q {lower_quant:.0%}",
                x=x,
                y=df[lower_q_col],
                mode="lines",
                line=dict(width=0, color=weak_quant_color),
                fill="tonexty",
                fillcolor=weak_quant_color,
                legendgroup=val_col,
                showlegend=False,
            )
        )

        # Inner quantile region
        strong_bands.append(
            go.Scatter(
                name=f"{val_col} Q {inner_upper_quant:.0%}",
                x=x,
                y=df[inner_upper_q_col],
                mode="lines",
                line=dict(width=0, color=strong_quant_color),
                legendgroup=val_col,
                showlegend=False,
            )
        )

        strong_bands.append(
            go.Scatter(
                name=f"{val_col} Q {inner_lower_quant:.0%}",
                x=x,
                y=df[inner_lower_q_col],
                mode="lines",
                line=dict(width=0, color=strong_quant_color),
                fill="tonexty",
                fillcolor=strong_quant_color,
                legendgroup=val_col,
                showlegend=False,
            )
        )

        # Main line drawn last so it stays on top
        lines.append(
            go.Scatter(
                name=val_col,
                x=x,
                y=df[val_col],
                mode="lines",
                line=dict(
                    color=base_color,
                    width=2,
                ),
                legendgroup=val_col,
            )
        )

    # plot after creation in this specific order for best layering
    fig.add_traces(weak_bands + strong_bands + lines)

    fig.update_layout(
        yaxis=dict(
            title=dict(
                text=yaxis_label,
            )
        ),
        xaxis=dict(
            title=dict(
                text=xaxis_label,
            )
        ),
        title=dict(
            text=title,
            subtitle=dict(
                text=subtitle,
            ),
        ),
        hovermode="x unified",
    )

    return fig
