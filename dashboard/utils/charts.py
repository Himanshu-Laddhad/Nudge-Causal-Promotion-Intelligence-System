"""
Reusable Plotly chart builders for CPIS dashboard.
All charts use the CPIS dark theme (matching notebook style).
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── Theme constants ──────────────────────────────────────────────────────────
BG_PLOT  = '#1a1a2e'
BG_PAPER = '#16213e'
TEXT     = '#ecf0f1'
GRID     = '#2c3e50'
PALETTE  = {
    'Naive XGB':    '#e74c3c',
    'T-Learner':    '#3498db',
    'S-Learner':    '#2ecc71',
    'X-Learner':    '#f39c12',
    'Causal Forest': '#1abc9c',
    'DR-Learner':   '#9b59b6',
}


def apply_dark_theme(fig: go.Figure, title: str = '', height: int = 450) -> go.Figure:
    fig.update_layout(
        plot_bgcolor=BG_PLOT,
        paper_bgcolor=BG_PAPER,
        font=dict(color=TEXT, family='Inter, sans-serif'),
        title=dict(text=title, font=dict(size=15, color=TEXT)),
        height=height,
        margin=dict(l=50, r=30, t=60, b=50),
        legend=dict(bgcolor='rgba(26,26,46,0.8)', bordercolor=GRID, borderwidth=1),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(color=TEXT)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(color=TEXT)),
    )
    return fig


def qini_curve_plot(models_data: list[dict]) -> go.Figure:
    """
    models_data: list of dicts with keys 'name', 'y_true', 'scores', 'treatment'
    Each dict produces one Qini curve.
    """
    fig = go.Figure()
    for m in models_data:
        name = m['name']
        scores = np.asarray(m['scores'], dtype=float)
        y      = np.asarray(m['y_true'],  dtype=float)
        t      = np.asarray(m['treatment'], dtype=float)

        order  = np.argsort(-scores)
        y_s, t_s = y[order], t[order]
        n      = len(y)

        cum_t = np.cumsum(t_s)
        cum_c = np.cumsum(1 - t_s)
        safe_c = np.where(cum_c == 0, 1, cum_c)

        qini  = np.cumsum(y_s * t_s) - np.cumsum(y_s * (1 - t_s)) * (cum_t / safe_c)
        fracs = np.linspace(0, 100, n + 1)[1:]
        color = PALETTE.get(name, TEXT)

        fig.add_trace(go.Scatter(
            x=fracs,
            y=np.concatenate([[0], qini]),
            mode='lines',
            name=name,
            line=dict(color=color, width=2),
        ))

    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 0],
        mode='lines',
        name='Random',
        line=dict(color='#7f8c8d', dash='dash', width=1),
    ))
    fig.update_xaxes(title_text='% Population Targeted')
    fig.update_yaxes(title_text='Cumulative Incremental Conversions')
    return apply_dark_theme(fig, 'Qini Curves — Model Comparison', height=500)


def cate_histogram(scores: np.ndarray, model_name: str, color: str = '#9b59b6') -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores, nbinsx=60,
        marker_color=color,
        opacity=0.75,
        name=model_name,
    ))
    fig.add_vline(x=0, line_color='#e74c3c', line_dash='dash', line_width=1.5)
    neg_pct = (scores <= 0).mean() * 100
    return apply_dark_theme(
        fig,
        f'{model_name} CATE Distribution — {neg_pct:.1f}% Deadweight (CATE ≤ 0)',
        height=380,
    )


def bar_comparison(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color_map: dict | None = None,
) -> go.Figure:
    fig = go.Figure()
    colors = [
        color_map.get(v, '#3498db') if color_map else '#3498db'
        for v in df[x_col]
    ]
    fig.add_trace(go.Bar(
        x=df[x_col],
        y=df[y_col],
        marker_color=colors,
        text=df[y_col].round(4),
        textposition='outside',
        textfont=dict(color=TEXT),
    ))
    return apply_dark_theme(fig, title, height=400)


def persuadability_heatmap(data_2x2: dict) -> go.Figure:
    """
    data_2x2: {
        'z': [[v00, v01], [v10, v11]],
        'x_labels': [...],
        'y_labels': [...],
    }
    """
    fig = go.Figure(go.Heatmap(
        z=data_2x2['z'],
        x=data_2x2['x_labels'],
        y=data_2x2['y_labels'],
        colorscale='RdYlGn',
        text=[[f"{v:.2%}" for v in row] for row in data_2x2['z']],
        texttemplate='%{text}',
        textfont=dict(size=16, color='white'),
        showscale=True,
    ))
    return apply_dark_theme(fig, 'Persuadability Matrix — Conversion Rate by Segment', height=380)


def budget_curve_plot(budget_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=budget_df['budget'] / 1000,
        y=budget_df['expected_lift'],
        mode='lines+markers',
        name='CATE-Optimal',
        line=dict(color='#9b59b6', width=2.5),
        marker=dict(size=7),
    ))
    if 'naive_lift' in budget_df.columns:
        fig.add_trace(go.Scatter(
            x=budget_df['budget'] / 1000,
            y=budget_df['naive_lift'],
            mode='lines+markers',
            name='Naive XGB',
            line=dict(color='#e74c3c', width=2, dash='dash'),
            marker=dict(size=7),
        ))
    fig.update_xaxes(title_text='Budget ($K)')
    fig.update_yaxes(title_text='Expected Incremental Conversions')
    return apply_dark_theme(fig, 'Incremental Lift vs Budget: CATE-Optimal vs Naive', height=420)


def segment_bar(
    labels: list[str],
    values: list[float],
    colors: list[str],
    title: str,
    x_title: str = '',
    y_title: str = '',
    height: int = 380,
) -> go.Figure:
    """Generic horizontal or vertical bar for segment breakdowns."""
    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.4f}" for v in values],
        textposition='outside',
        textfont=dict(color=TEXT),
    ))
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
    return apply_dark_theme(fig, title, height=height)


def pie_segments(labels: list[str], values: list[int], colors: list[str], title: str) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.35,
        textinfo='label+percent',
        textfont=dict(color=TEXT, size=13),
    ))
    fig.update_layout(
        plot_bgcolor=BG_PLOT,
        paper_bgcolor=BG_PAPER,
        font=dict(color=TEXT, family='Inter, sans-serif'),
        title=dict(text=title, font=dict(size=15, color=TEXT)),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(bgcolor='rgba(26,26,46,0.8)', bordercolor=GRID, borderwidth=1),
        showlegend=True,
    )
    return fig
