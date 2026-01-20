"""Power System Studio - Visualization helpers.

Plotting utilities used by Streamlit.
- Uses Plotly for interactive charts.
- Avoids seaborn.

Functions return Plotly Figure objects so the app can display them.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def fig_voltage_profile(res_bus: pd.DataFrame, bus_name_map: Optional[dict] = None) -> go.Figure:
    """Plot vm_pu per bus (pandapower results)."""
    if res_bus is None or res_bus.empty or 'vm_pu' not in res_bus.columns:
        fig = go.Figure()
        fig.add_annotation(text='No bus results to plot', showarrow=False)
        return fig

    x = res_bus.index.astype(str)
    if bus_name_map:
        x = [bus_name_map.get(int(i), str(i)) if str(i).isdigit() else str(i) for i in x]

    fig = go.Figure(data=[go.Bar(x=x, y=res_bus['vm_pu'].astype(float).values)])
    fig.update_layout(title='Voltage magnitude (p.u.)', xaxis_title='Bus', yaxis_title='Vm (p.u.)')
    return fig


def fig_angle_profile(res_bus: pd.DataFrame, bus_name_map: Optional[dict] = None) -> go.Figure:
    """Plot va_degree per bus (pandapower results)."""
    if res_bus is None or res_bus.empty or 'va_degree' not in res_bus.columns:
        fig = go.Figure()
        fig.add_annotation(text='No bus angle results to plot', showarrow=False)
        return fig

    x = res_bus.index.astype(str)
    if bus_name_map:
        x = [bus_name_map.get(int(i), str(i)) if str(i).isdigit() else str(i) for i in x]

    fig = go.Figure(data=[go.Scatter(x=x, y=res_bus['va_degree'].astype(float).values, mode='lines+markers')])
    fig.update_layout(title='Voltage angle (deg)', xaxis_title='Bus', yaxis_title='Angle (deg)')
    return fig


def fig_convergence(trace_df: Optional[pd.DataFrame], iterations: Optional[int], converged: bool) -> go.Figure:
    """Best-effort convergence plot.

    pandapower does not always expose per-iteration mismatch. If a trace dataframe is
    provided, we plot numeric columns. Otherwise we show a single-point summary.
    """
    fig = go.Figure()
    if trace_df is not None and not trace_df.empty:
        # plot first numeric column
        numeric_cols = trace_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            col = numeric_cols[0]
            fig.add_trace(go.Scatter(y=trace_df[col].values, mode='lines+markers', name=str(col)))
            fig.update_layout(title='Convergence trace (best-effort)', xaxis_title='Iteration', yaxis_title=str(col))
            return fig

    # fallback summary
    fig.add_trace(go.Scatter(x=[0], y=[1 if converged else 0], mode='markers'))
    title = f"Converged: {converged}"
    if iterations is not None:
        title += f" | iterations: {iterations}"
    fig.update_layout(title=title, xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-0.2, 1.2]))
    return fig


def fig_network(buses: pd.DataFrame, lines: pd.DataFrame, trafos: pd.DataFrame) -> go.Figure:
    """Simple one-line diagram using bus x/y coordinates."""
    fig = go.Figure()
    if buses is None or buses.empty:
        fig.add_annotation(text='No buses to draw', showarrow=False)
        return fig

    # bus positions
    bus_pos = {}
    for _, r in buses.iterrows():
        name = str(r.get('name', ''))
        x = float(r.get('x', 0.0) or 0.0)
        y = float(r.get('y', 0.0) or 0.0)
        bus_pos[name] = (x, y)

    # draw lines
    if lines is not None and not lines.empty:
        for _, r in lines.iterrows():
            fb = str(r.get('from_bus', ''))
            tb = str(r.get('to_bus', ''))
            if fb in bus_pos and tb in bus_pos:
                x0, y0 = bus_pos[fb]
                x1, y1 = bus_pos[tb]
                fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', name=str(r.get('name', 'line'))))

    # draw transformers as dashed lines
    if trafos is not None and not trafos.empty:
        for _, r in trafos.iterrows():
            hv = str(r.get('hv_bus', ''))
            lv = str(r.get('lv_bus', ''))
            if hv in bus_pos and lv in bus_pos:
                x0, y0 = bus_pos[hv]
                x1, y1 = bus_pos[lv]
                fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(dash='dash'), name=str(r.get('name', 'trafo'))))

    # draw buses
    xs = []
    ys = []
    labels = []
    for name, (x, y) in bus_pos.items():
        xs.append(x)
        ys.append(y)
        labels.append(name)

    fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers+text', text=labels, textposition='top center', name='buses'))
    fig.update_layout(title='Single-line diagram (layout from x/y)', xaxis_title='x', yaxis_title='y', showlegend=False)
    return fig
