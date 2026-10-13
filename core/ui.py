"""Shared UI components and layout helpers for HealthSure."""
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import io
import re
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from core.theme import THEME

# Map common color / tone names to THEME palette
ACCENT_MAP: Dict[str, str] = {
    "sky": THEME["ACCENT"],
    "accent": THEME["ACCENT"],
    "indigo": THEME["INDIGO"],
    "emerald": THEME["LOW"],
    "green": THEME["LOW"],
    "low": THEME["LOW"],
    "amber": THEME["MED"],
    "yellow": THEME["MED"],
    "med": THEME["MED"],
    "medium": THEME["MED"],
    "rose": THEME["HIGH"],
    "red": THEME["HIGH"],
    "high": THEME["HIGH"],
    "muted": THEME["MUTED"],
}


def render_html(html_str: str, container: Any = None) -> None:
    """Render HTML safely without markdown indentation issues or code block conversion."""
    if not html_str:
        return
    c = container or st
    # Remove any HTML comments
    clean = re.sub(r"<!--.*?-->", "", html_str, flags=re.DOTALL)
    # Strip leading and trailing whitespace per line to avoid markdown code indentation
    clean_lines = [line.strip() for line in clean.splitlines() if line.strip()]
    clean_html = "\n".join(clean_lines)

    if hasattr(c, "html"):
        c.html(clean_html)
    else:
        c.markdown(clean_html, unsafe_allow_html=True)


def page_header(title: str, subtitle: Optional[str] = None) -> None:
    """Render a modern glassmorphic hero banner card.

    Parameters
    ----------
    title : str
        Primary page title.
    subtitle : str, optional
        Secondary explanatory text.
    """
    subtitle_html = (
        f'<p style="color: {THEME["MUTED"]}; margin: 0; font-size: 0.95rem; '
        f'font-weight: 400; line-height: 1.5;">{subtitle}</p>'
        if subtitle
        else ""
    )

    header_html = f"""
    <div style="
        background: {THEME['CARD']};
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid {THEME['ACCENT']};
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.30);
    ">
        <h1 style="color: {THEME['TEXT']}; margin: 0 0 0.35rem 0; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em;">
            {title}
        </h1>
        {subtitle_html}
    </div>
    """
    render_html(header_html)


def section(title: str, subtitle: Optional[str] = None) -> None:
    """Render a section divider with title, optional subtitle, and gradient bar.

    Parameters
    ----------
    title : str
        Section title heading.
    subtitle : str, optional
        Section description or guidance.
    """
    subtitle_html = (
        f'<p style="color: {THEME["MUTED"]}; margin: 0.2rem 0 0.4rem 0; font-size: 0.875rem;">{subtitle}</p>'
        if subtitle
        else ""
    )

    section_html = f"""
    <div style="margin: 2rem 0 1rem 0;">
        <h2 style="color: {THEME['TEXT']}; margin: 0; font-size: 1.25rem; font-weight: 700; letter-spacing: -0.01em;">
            {title}
        </h2>
        {subtitle_html}
        <div style="
            height: 1px;
            width: 100%;
            background: linear-gradient(90deg, {THEME['ACCENT']} 0%, {THEME['INDIGO']} 50%, transparent 100%);
            opacity: 0.5;
            margin-top: 0.5rem;
        "></div>
    </div>
    """
    render_html(section_html)


def kpi_card(
    label: str,
    value: Any,
    delta: Optional[Any] = None,
    accent: str = "sky",
) -> None:
    """Render a single KPI metric card with glassmorphism and accent bar.

    Parameters
    ----------
    label : str
        Name of the metric.
    value : Any
        Value to display prominently.
    delta : Any, optional
        Change or difference indicator (e.g. '+5.2%', '-1.4%').
    accent : str, default 'sky'
        Accent color key ('sky', 'indigo', 'emerald', 'amber', 'rose', etc.).
    """
    accent_color = ACCENT_MAP.get(str(accent).lower(), THEME["ACCENT"])

    delta_html = ""
    if delta is not None:
        delta_str = str(delta).strip()
        if delta_str.startswith("+"):
            delta_color = THEME["LOW"]
            arrow = "↑"
        elif delta_str.startswith("-"):
            delta_color = THEME["HIGH"]
            arrow = "↓"
        else:
            delta_color = THEME["ACCENT"]
            arrow = "→"

        delta_html = (
            f'<div style="margin-top: 0.4rem;">'
            f'<span style="display: inline-flex; align-items: center; gap: 0.2rem; '
            f'font-size: 0.775rem; font-weight: 600; color: {delta_color}; '
            f'background: {delta_color}22; border: 1px solid {delta_color}44; '
            f'padding: 0.15rem 0.5rem; border-radius: 6px;">{arrow} {delta_str}</span>'
            f'</div>'
        )

    card_html = (
        f'<div style="background: {THEME["CARD"]}; border: 1px solid rgba(255, 255, 255, 0.08); '
        f'border-top: 3px solid {accent_color}; border-radius: 12px; padding: 1.15rem 1.25rem; '
        f'backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); '
        f'box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.25); display: flex; flex-direction: column; '
        f'justify-content: space-between; min-height: 110px;">'
        f'<div style="font-size: 0.775rem; font-weight: 600; text-transform: uppercase; '
        f'letter-spacing: 0.05em; color: {THEME["MUTED"]}; margin-bottom: 0.4rem;">{label}</div>'
        f'<div style="font-size: 1.65rem; font-weight: 700; color: {THEME["TEXT"]}; '
        f'letter-spacing: -0.02em; line-height: 1.2;">{value}</div>'
        f'{delta_html}'
        f'</div>'
    )
    render_html(card_html)


def kpi_row(items: Iterable[Union[Dict[str, Any], Tuple, List]], cols: int = 4) -> None:
    """Render a responsive grid of KPI cards that wraps automatically when items > cols.

    Parameters
    ----------
    items : Iterable[Union[dict, tuple, list]]
        Collection of KPI definitions. Either dicts with keys ('label', 'value', 'delta', 'accent')
        or tuples/lists of (label, value, [delta], [accent]).
    cols : int, default 4
        Maximum number of columns per row before wrapping.
    """
    item_list = list(items)
    if not item_list:
        return

    for i in range(0, len(item_list), cols):
        chunk = item_list[i : i + cols]
        columns = st.columns(cols)
        for col, item in zip(columns, chunk):
            with col:
                if isinstance(item, dict):
                    kpi_card(
                        label=item.get("label", ""),
                        value=item.get("value", ""),
                        delta=item.get("delta", None),
                        accent=item.get("accent", "sky"),
                    )
                elif isinstance(item, (tuple, list)):
                    kpi_card(
                        label=item[0],
                        value=item[1],
                        delta=item[2] if len(item) > 2 else None,
                        accent=item[3] if len(item) > 3 else "sky",
                    )


def empty_state(
    icon: str,
    title: str,
    message: str,
    cta_label: Optional[str] = None,
    cta_key: Optional[str] = None,
) -> bool:
    """Render a centered glass card empty state with optional call-to-action button.

    Parameters
    ----------
    icon : str
        Icon or emoji symbol.
    title : str
        Empty state heading.
    message : str
        Descriptive explanation or next steps.
    cta_label : str, optional
        Button label for call-to-action.
    cta_key : str, optional
        Unique key for the button widget.

    Returns
    -------
    bool
        True if the CTA button was clicked, False otherwise.
    """
    card_html = f"""
    <div style="
        background: {THEME['CARD']};
        border: 1px dashed rgba(255, 255, 255, 0.16);
        border-radius: 14px;
        padding: 2.5rem 1.5rem;
        text-align: center;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin: 1.5rem 0;
    ">
        <div style="font-size: 2.75rem; margin-bottom: 0.75rem; line-height: 1;">{icon}</div>
        <h3 style="color: {THEME['TEXT']}; font-size: 1.2rem; font-weight: 600; margin: 0 0 0.4rem 0;">{title}</h3>
        <p style="color: {THEME['MUTED']}; font-size: 0.9rem; margin: 0 auto; max-width: 480px; line-height: 1.5;">{message}</p>
    </div>
    """
    render_html(card_html)

    clicked = False
    if cta_label:
        col_left, col_mid, col_right = st.columns([1, 1, 1])
        with col_mid:
            clicked = st.button(cta_label, key=cta_key, use_container_width=True)
    return clicked


def risk_badge(tier: str, render: bool = True) -> str:
    """Render and/or return a styled HTML pill badge for risk tier (Low / Medium / High).

    Parameters
    ----------
    tier : str
        Risk tier name (e.g. 'Low', 'Medium', 'High', 'Tier 1', etc.).
    render : bool, default True
        If True, directly renders to Streamlit. Set to False when embedding into custom HTML.

    Returns
    -------
    str
        HTML string of the styled badge pill.
    """
    tier_clean = str(tier).strip()
    tier_lower = tier_clean.lower()

    if "low" in tier_lower or "tier 1" in tier_lower:
        color = THEME["LOW"]
        display_label = "Low Risk" if tier_lower == "low" else tier_clean
    elif "med" in tier_lower or "tier 2" in tier_lower:
        color = THEME["MED"]
        display_label = "Medium Risk" if "med" in tier_lower and "medium" not in tier_lower else tier_clean
    elif "high" in tier_lower or "tier 3" in tier_lower:
        color = THEME["HIGH"]
        display_label = "High Risk" if tier_lower == "high" else tier_clean
    else:
        color = THEME["ACCENT"]
        display_label = tier_clean

    badge_html = f"""<span style="
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: {color}22;
        color: {color};
        border: 1px solid {color}55;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.775rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        white-space: nowrap;
    "><span style="width: 6px; height: 6px; border-radius: 50%; background: {color};"></span>{display_label}</span>"""

    if render:
        render_html(badge_html)
    return badge_html


def latest_value_card(
    label: str,
    value: Any,
    unit: str,
    status: str,
    hint: Optional[str] = None,
) -> None:
    """Render a card displaying a metric's latest value, unit, status badge, and optional reference hint.

    Parameters
    ----------
    label : str
        Clinical or financial parameter name.
    value : Any
        Latest recorded value.
    unit : str
        Unit of measurement (e.g. 'mmHg', 'kg/m²', 'INR').
    status : str
        Clinical or tier status ('Normal', 'Elevated', 'Low', 'High', etc.).
    hint : str, optional
        Target clinical range or guidance note.
    """
    badge_html = risk_badge(status, render=False)

    hint_html = ""
    if hint:
        hint_html = f"""
        <div style="
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            margin-top: 0.75rem;
            padding-top: 0.6rem;
            font-size: 0.775rem;
            color: {THEME['MUTED']};
            display: flex;
            align-items: center;
            gap: 0.35rem;
        ">
            <span style="color: {THEME['ACCENT']};">ℹ</span> {hint}
        </div>
        """

    card_html = f"""
    <div style="
        background: {THEME['CARD']};
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.15rem 1.25rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.25);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
            <span style="
                font-size: 0.775rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: {THEME['MUTED']};
            ">{label}</span>
            {badge_html}
        </div>
        <div style="display: flex; align-items: baseline; gap: 0.35rem;">
            <span style="
                font-size: 1.75rem;
                font-weight: 700;
                color: {THEME['TEXT']};
                letter-spacing: -0.02em;
                line-height: 1.1;
            ">{value}</span>
            <span style="font-size: 0.875rem; color: {THEME['MUTED']}; font-weight: 500;">{unit}</span>
        </div>
        {hint_html}
    </div>
    """
    render_html(card_html)


def get_plotly_layout() -> Dict[str, Any]:
    """Return common Plotly layout configuration with transparent backgrounds and theme styling.

    Returns
    -------
    dict
        Plotly layout dictionary.
    """
    return {
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
        "plot_bgcolor": "rgba(0, 0, 0, 0)",
        "font": {
            "family": "Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, sans-serif",
            "color": THEME["TEXT"],
            "size": 12,
        },
        "xaxis": {
            "gridcolor": "rgba(255, 255, 255, 0.08)",
            "zerolinecolor": "rgba(255, 255, 255, 0.12)",
            "tickfont": {
                "family": "Plus Jakarta Sans, sans-serif",
                "color": THEME["MUTED"],
                "size": 11,
            },
            "title": {
                "font": {
                    "family": "Plus Jakarta Sans, sans-serif",
                    "color": THEME["MUTED"],
                    "size": 12,
                }
            },
        },
        "yaxis": {
            "gridcolor": "rgba(255, 255, 255, 0.08)",
            "zerolinecolor": "rgba(255, 255, 255, 0.12)",
            "tickfont": {
                "family": "Plus Jakarta Sans, sans-serif",
                "color": THEME["MUTED"],
                "size": 11,
            },
            "title": {
                "font": {
                    "family": "Plus Jakarta Sans, sans-serif",
                    "color": THEME["MUTED"],
                    "size": 12,
                }
            },
        },
        "legend": {
            "bgcolor": "rgba(15, 23, 42, 0.70)",
            "bordercolor": "rgba(255, 255, 255, 0.08)",
            "borderwidth": 1,
            "font": {
                "family": "Plus Jakarta Sans, sans-serif",
                "color": THEME["TEXT"],
                "size": 11,
            },
        },
        "hoverlabel": {
            "bgcolor": THEME["BG"],
            "bordercolor": THEME["ACCENT"],
            "font": {
                "family": "Plus Jakarta Sans, sans-serif",
                "color": THEME["TEXT"],
                "size": 12,
            },
        },
    }


def threshold_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    bands: List[Tuple[float, float, str, str]],
    title: str,
    y_label: str,
) -> Optional[go.Figure]:
    """Render a line chart with colored background threshold bands behind the trace.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing x and y coordinates.
    x : str
        Column name for X axis.
    y : str
        Column name for Y axis.
    bands : list of tuples
        Threshold bands formatted as [(low, high, color, label), ...].
    title : str
        Chart title.
    y_label : str
        Y-axis label.

    Returns
    -------
    go.Figure or None
        Plotly figure object, or None if df is empty.
    """
    if df is None or df.empty:
        empty_state("📊", "No Data Available", "No records found to plot.")
        return None

    fig = go.Figure()

    # Draw line trace with accent color and markers
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode="lines+markers",
            name=y_label or str(y),
            line=dict(color=THEME["ACCENT"], width=3, shape="spline"),
            marker=dict(
                size=6,
                color=THEME["ACCENT"],
                symbol="circle",
                line=dict(width=1.5, color="#ffffff"),
            ),
            hovertemplate=f"<b>%{{x}}</b><br>{y_label or y}: %{{y}}<extra></extra>",
        )
    )

    # Add threshold rectangle bands behind the line trace
    for band in bands:
        if len(band) >= 4:
            low, high, color, label = band[:4]
        elif len(band) == 3:
            low, high, color = band
            label = ""
        else:
            continue

        band_color = ACCENT_MAP.get(str(color).lower(), THEME.get(str(color).upper(), color))

        fig.add_hrect(
            y0=low,
            y1=high,
            fillcolor=band_color,
            opacity=0.14,
            line_width=0,
            layer="below",
            annotation_text=str(label) if label else None,
            annotation_position="top left",
            annotation_font=dict(
                family="Plus Jakarta Sans, sans-serif",
                size=10,
                color=THEME["MUTED"],
            ),
        )

    layout = get_plotly_layout()
    layout.update({
        "title": {
            "text": f"<b>{title}</b>",
            "font": {
                "family": "Plus Jakarta Sans, sans-serif",
                "size": 15,
                "color": THEME["TEXT"],
            },
            "x": 0.02,
            "xanchor": "left",
        },
        "xaxis": {
            **layout["xaxis"],
            "title": {
                "text": str(x),
                "font": {"family": "Plus Jakarta Sans, sans-serif", "color": THEME["MUTED"], "size": 12},
            },
        },
        "yaxis": {
            **layout["yaxis"],
            "title": {
                "text": str(y_label or y),
                "font": {"family": "Plus Jakarta Sans, sans-serif", "color": THEME["MUTED"], "size": 12},
            },
        },
        "margin": dict(l=40, r=30, t=50, b=40),
    })

    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)
    return fig


def timeline_gantt(
    df: pd.DataFrame,
    start_col: str,
    end_col: str,
    label_col: str,
    color_col: Optional[str] = None,
) -> Optional[go.Figure]:
    """Render an interactive Gantt chart timeline using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        Timeline event dataset.
    start_col : str
        Column name for start timestamp or point.
    end_col : str
        Column name for end timestamp or point.
    label_col : str
        Column name for item or stage label.
    color_col : str, optional
        Column name for category color differentiation.

    Returns
    -------
    go.Figure or None
        Plotly figure object, or None if df is empty.
    """
    if df is None or df.empty:
        empty_state("📅", "No Timeline Data", "No timeline events available to display.")
        return None

    color_palette = [
        THEME["ACCENT"],
        THEME["INDIGO"],
        THEME["LOW"],
        THEME["MED"],
        THEME["HIGH"],
    ]

    try:
        if color_col and color_col in df.columns:
            fig = px.timeline(
                df,
                x_start=start_col,
                x_end=end_col,
                y=label_col,
                color=color_col,
                color_discrete_sequence=color_palette,
            )
        else:
            fig = px.timeline(
                df,
                x_start=start_col,
                x_end=end_col,
                y=label_col,
                color_discrete_sequence=[THEME["ACCENT"]],
            )
    except Exception:
        # Fallback for numeric range timelines
        fig = go.Figure()
        durations = df[end_col] - df[start_col]
        fig.add_trace(
            go.Bar(
                x=durations,
                y=df[label_col],
                base=df[start_col],
                orientation="h",
                marker=dict(color=THEME["ACCENT"]),
            )
        )

    layout = get_plotly_layout()
    layout.update({
        "yaxis": {
            **layout["yaxis"],
            "autorange": "reversed",
            "title": {
                "text": str(label_col),
                "font": {"family": "Plus Jakarta Sans, sans-serif", "color": THEME["MUTED"], "size": 12},
            },
        },
        "xaxis": {
            **layout["xaxis"],
            "title": {
                "text": "Timeline",
                "font": {"family": "Plus Jakarta Sans, sans-serif", "color": THEME["MUTED"], "size": 12},
            },
        },
        "margin": dict(l=40, r=30, t=30, b=40),
    })

    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)
    return fig


def download_button(
    df: pd.DataFrame,
    filename: str,
    label: str,
    key: str,
    filetype: str = "csv",
) -> bool:
    """Render a styled download button for data export.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export.
    filename : str
        Target file name (e.g. 'export.csv').
    label : str
        Download button label.
    key : str
        Unique Streamlit widget key.
    filetype : str, default 'csv'
        Export format ('csv', 'xlsx', 'excel', or 'json').

    Returns
    -------
    bool
        True if the download button was clicked, False otherwise.
    """
    filetype_clean = filetype.lower().strip()
    if filetype_clean in ("csv", ".csv"):
        data = df.to_csv(index=False).encode("utf-8")
        mime = "text/csv"
    elif filetype_clean in ("xlsx", "excel", ".xlsx"):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        data = buffer.getvalue()
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif filetype_clean in ("json", ".json"):
        data = df.to_json(orient="records", indent=2).encode("utf-8")
        mime = "application/json"
    else:
        data = df.to_csv(index=False).encode("utf-8")
        mime = "text/plain"

    return st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime,
        key=key,
        use_container_width=True,
    )


def sidebar_footer() -> None:
    """Render standardized footer in sidebar across all pages."""
    footer_html = (
        f'<div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.08); '
        f'font-size: 0.75rem; color: {THEME["MUTED"]}; text-align: center; line-height: 1.4;">'
        f'HealthSure • v2.0 • No external AI keys required'
        f'</div>'
    )
    render_html(footer_html, container=st.sidebar)
