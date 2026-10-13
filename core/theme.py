"""Theme definitions and visual styling utilities for HealthSure."""
from typing import Dict
import streamlit as st

THEME: Dict[str, str] = {
    "LOW": "#10b981",
    "MED": "#f59e0b",
    "HIGH": "#ef4444",
    "ACCENT": "#38bdf8",
    "INDIGO": "#4f46e5",
    "BG": "#0b0f19",
    "CARD": "rgba(15,23,42,0.65)",
    "TEXT": "#f1f5f9",
    "MUTED": "#94a3b8",
}

THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* Global Font & Color Overrides */
html, body, [class*="css"], .stApp, .stMarkdown, .stText, p, span, h1, h2, h3, h4, h5, h6, input, select, textarea, button {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

/* Main App Background - Radial Gradient Glow */
.stApp {{
    background: radial-gradient(circle at 15% 15%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(79, 70, 229, 0.10) 0%, transparent 40%),
                radial-gradient(circle at 50% 50%, rgba(15, 23, 42, 0.95) 0%, {THEME["BG"]} 100%) !important;
    background-color: {THEME["BG"]} !important;
    background-attachment: fixed !important;
    color: {THEME["TEXT"]} !important;
}}

/* Sidebar - Persistent Expanded State & Glassmorphism */
[data-testid="stSidebar"] {{
    background-color: rgba(11, 15, 25, 0.90) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    min-width: 280px !important;
}}

/* Disallow collapsing sidebar */
[data-testid="stSidebarCollapseButton"] {{
    display: none !important;
}}

/* Glass Cards */
.glass-card {{
    background: {THEME["CARD"]} !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35) !important;
    color: {THEME["TEXT"]} !important;
}}

/* Gradient Buttons - All Blue / Sky / Indigo Palette */
div.stButton > button,
div.stButton > button[kind="primary"],
div.stButton > button[kind="secondary"],
div.stDownloadButton > button,
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-secondary"],
button[kind="primary"],
button[kind="secondary"] {{
    background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%) !important;
    background-color: #0284c7 !important;
    color: #ffffff !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.35rem !important;
    transition: all 0.25s ease-in-out !important;
    box-shadow: 0 4px 14px 0 rgba(2, 132, 199, 0.35) !important;
}}

div.stButton > button:hover,
div.stButton > button[kind="primary"]:hover,
div.stDownloadButton > button:hover,
button[data-testid="stBaseButton-primary"]:hover {{
    background: linear-gradient(135deg, #0369a1 0%, #0ea5e9 100%) !important;
    background-color: #0369a1 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px 0 rgba(56, 189, 248, 0.45) !important;
    color: #ffffff !important;
}}

div.stButton > button:active,
div.stButton > button[kind="primary"]:active,
div.stDownloadButton > button:active,
button[data-testid="stBaseButton-primary"]:active {{
    transform: translateY(0px) !important;
}}

/* Form Inputs, Select Boxes, and Text Areas */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"],
div[data-baseweb="textarea"] > div {{
    background-color: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 8px !important;
    color: {THEME["TEXT"]} !important;
}}

div[data-baseweb="select"] span {{
    color: {THEME["TEXT"]} !important;
}}

/* Streamlit Native Metric Cards */
[data-testid="stMetric"] {{
    background: {THEME["CARD"]} !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.25) !important;
}}

[data-testid="stMetricLabel"] {{
    color: {THEME["MUTED"]} !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}}

[data-testid="stMetricValue"] {{
    color: {THEME["TEXT"]} !important;
    font-weight: 700 !important;
    font-size: 1.65rem !important;
}}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px !important;
    background-color: rgba(15, 23, 42, 0.5) !important;
    padding: 4px !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 8px !important;
    color: {THEME["MUTED"]} !important;
    font-weight: 500 !important;
}}

.stTabs [aria-selected="true"] {{
    background-color: rgba(56, 189, 248, 0.15) !important;
    color: {THEME["ACCENT"]} !important;
    font-weight: 600 !important;
}}

/* DataFrames and Tables */
[data-testid="stDataFrame"] {{
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}}

/* Typography */
h1, h2, h3, h4 {{
    color: {THEME["TEXT"]} !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}}

p, span, label {{
    color: {THEME["TEXT"]};
}}

/* Sliders */
.stSlider > div > div > div > div {{
    background-color: {THEME["ACCENT"]} !important;
}}

/* Recommendation Cards - All Pure Blue (no red / amber) */
.rec-card,
.rec-card.priority-high,
.rec-card.priority-medium,
.rec-card.priority-low {{
    background: rgba(15, 23, 42, 0.70) !important;
    border: 1px solid rgba(56, 189, 248, 0.22) !important;
    border-left: 4px solid #38bdf8 !important;
    border-radius: 8px !important;
    padding: 11px 15px !important;
    margin: 8px 0 !important;
    font-size: 0.85rem !important;
    line-height: 1.45 !important;
    color: #f1f5f9 !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
}}
</style>
"""


def apply_theme() -> None:
    """Inject the dark glassmorphism CSS theme on every run."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)
