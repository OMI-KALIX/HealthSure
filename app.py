"""HealthSure InsurTech main portal and application entry point."""
import streamlit as st
from core.theme import THEME, apply_theme
from core.ui import page_header, section, sidebar_footer, render_html
from core.data_io import get_dataset, reset_dataset

# 1. Page Configuration
st.set_page_config(
    page_title="HealthSure",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Dark Glassmorphism Design System
apply_theme()

# 3. Sidebar Branding & Dataset Management
with st.sidebar:
    render_html(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.35rem;">
                <span style="font-size: 1.6rem; line-height: 1;">🏥</span>
                <span style="font-size: 1.3rem; font-weight: 700; color: {THEME['TEXT']}; letter-spacing: -0.02em;">
                    HealthSure
                </span>
            </div>
            <div style="font-size: 0.775rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: {THEME['ACCENT']};">
                InsurTech Suite • v2.0.0
            </div>
        </div>
        """
    )

    render_html(
        """
        <div style="height: 1px; width: 100%; background: rgba(255, 255, 255, 0.08); margin: 1rem 0;"></div>
        """
    )

    sidebar_df = get_dataset()
    if sidebar_df is not None and not sidebar_df.empty:
        dataset_label = st.session_state.get("dataset_name", "user_dataset")
        render_html(
            f"""
            <div style="margin-bottom: 0.75rem;">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: {THEME['MUTED']}; font-weight: 600;">Active Dataset</div>
                <div style="color: {THEME['TEXT']}; font-weight: 600; font-size: 0.9rem;">{dataset_label}</div>
                <div style="color: {THEME['MUTED']}; font-size: 0.8rem;">{len(sidebar_df):,} rows • {len(sidebar_df.columns)} cols</div>
            </div>
            """
        )
        if st.button("Clear dataset", key="clear_dataset_sidebar", use_container_width=True):
            reset_dataset()
            st.rerun()
    else:
        st.caption("No dataset loaded in session memory.")

    sidebar_footer()

# 4. Hero Banner Header
page_header(
    title="HealthSure",
    subtitle="Actuarial risk prediction, dataset exploration, and single-patient clinical deep dive.",
)

# 5. Interactive Navigation Cards
section("Modules & Applications", "Select an  HealthSure module to begin analysis.")

col1, col2, col3 = st.columns(3)

with col1:
    render_html(
        f"""
        <div class="glass-card" style="margin-bottom: 0.85rem; min-height: 155px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem; line-height: 1;">📊</div>
                <h3 style="color: {THEME['TEXT']}; font-size: 1.15rem; font-weight: 700; margin: 0 0 0.35rem 0;">
                    Insurance Prediction
                </h3>
                <p style="color: {THEME['MUTED']}; font-size: 0.875rem; margin: 0; line-height: 1.45;">
                    Predict underwriting risk tier and class probabilities for applicants.
                </p>
            </div>
        </div>
        """
    )
    st.page_link(
        "pages/1_Insurance_Prediction.py",
        label="Open module →",
        icon="📊",
        use_container_width=True,
    )

with col2:
    render_html(
        f"""
        <div class="glass-card" style="margin-bottom: 0.85rem; min-height: 155px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem; line-height: 1;">🔎</div>
                <h3 style="color: {THEME['TEXT']}; font-size: 1.15rem; font-weight: 700; margin: 0 0 0.35rem 0;">
                    Dataset Explorer
                </h3>
                <p style="color: {THEME['MUTED']}; font-size: 0.875rem; margin: 0; line-height: 1.45;">
                    Upload, validate, and inspect applicant datasets with visual analytics.
                </p>
            </div>
        </div>
        """
    )
    st.page_link(
        "pages/2_Dataset_Explorer.py",
        label="Open module →",
        icon="🔎",
        use_container_width=True,
    )

with col3:
    render_html(
        f"""
        <div class="glass-card" style="margin-bottom: 0.85rem; min-height: 155px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem; line-height: 1;">🩺</div>
                <h3 style="color: {THEME['TEXT']}; font-size: 1.15rem; font-weight: 700; margin: 0 0 0.35rem 0;">
                    Patient Deep Dive
                </h3>
                <p style="color: {THEME['MUTED']}; font-size: 0.875rem; margin: 0; line-height: 1.45;">
                    In-depth clinical profile analysis, health metrics, and risk factors.
                </p>
            </div>
        </div>
        """
    )
    st.page_link(
        "pages/3_Patient_Deep_Dive.py",
        label="Open module →",
        icon="🩺",
        use_container_width=True,
    )

# 6. Session Dataset Status Strip
section("Session State", "Current dataset loaded into memory.")

active_df = get_dataset()
if active_df is not None and not active_df.empty:
    dataset_name = st.session_state.get("dataset_name", "user_dataset")
    col_status, col_clear = st.columns([4, 1])
    with col_status:
        render_html(
            f"""
            <div class="glass-card" style="display: flex; align-items: center; gap: 0.85rem; padding: 0.95rem 1.25rem;">
                <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background-color: {THEME['LOW']}; box-shadow: 0 0 10px {THEME['LOW']};"></span>
                <div>
                    <span style="font-weight: 700; color: {THEME['TEXT']}; font-size: 0.95rem;">Active Dataset: {dataset_name}</span>
                    <span style="color: {THEME['MUTED']}; font-size: 0.85rem; margin-left: 0.5rem;">
                        ({len(active_df):,} records, {len(active_df.columns)} columns)
                    </span>
                </div>
            </div>
            """
        )
    with col_clear:
        if st.button("Clear dataset", key="clear_dataset_main", use_container_width=True):
            reset_dataset()
            st.rerun()
else:
    render_html(
        f"""
        <div class="glass-card" style="display: flex; align-items: center; gap: 0.85rem; padding: 0.95rem 1.25rem;">
            <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background-color: {THEME['MUTED']};"></span>
            <span style="color: {THEME['MUTED']}; font-size: 0.9rem;">
                No dataset loaded in session memory. Upload files in the Dataset Explorer to analyze cohorts.
            </span>
        </div>
        """
    )
