"""Insurance Premium Prediction and Actuarial Underwriting page for HealthSure."""
from typing import Dict
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from core.theme import THEME, apply_theme
from core.ui import (
    page_header,
    section,
    kpi_row,
    risk_badge,
    download_button,
    get_plotly_layout,
    empty_state,
    sidebar_footer,
    render_html,
)
from core.data_io import (
    read_uploaded_file,
    normalize_columns,
    coerce_types,
    detect_column_roles,
    store_dataset,
    get_dataset,
    reset_dataset,
)
from core.model import (
    engineer_features,
    predict_dataframe,
    predict_single,
)
from core.recommendations import (
    recommend,
    rule_engine,
    recommendations_to_text,
)

_recommendations_to_text = recommendations_to_text

# 1. Page Configuration
st.set_page_config(
    page_title="Insurance Risk Prediction | HealthSure",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Theme
apply_theme()

# 3. Sidebar Footer
with st.sidebar:
    sidebar_footer()

# 4. Header Banner
page_header(
    title="Insurance Risk Prediction",
    subtitle="Individual applicant actuarial underwriting and batch cohort risk tier classification.",
)

# 30 Indian Cities with Tier Classifications
INDIAN_CITIES = [
    "Agra", "Ahmedabad", "Allahabad", "Amritsar", "Bangalore", "Bhopal",
    "Chennai", "Delhi", "Faridabad", "Ghaziabad", "Hyderabad", "Indore",
    "Jaipur", "Kanpur", "Kolkata", "Lucknow", "Ludhiana", "Meerut",
    "Mumbai", "Nagpur", "Nashik", "Patna", "Pune", "Rajkot",
    "Ranchi", "Srinagar", "Surat", "Vadodara", "Varanasi", "Visakhapatnam",
]

TIER_1_CITIES = {
    "bangalore", "chennai", "delhi", "hyderabad", "kolkata", "mumbai", "pune", "ahmedabad"
}
TIER_2_CITIES = {
    "agra", "allahabad", "amritsar", "bhopal", "faridabad", "ghaziabad",
    "indore", "jaipur", "kanpur", "lucknow", "ludhiana", "meerut",
    "nagpur", "nashik", "patna", "rajkot", "ranchi", "srinagar",
    "surat", "vadodara", "varanasi", "visakhapatnam",
}

def get_city_tier(city_name: str) -> int:
    clean = str(city_name).strip().lower()
    if clean in TIER_1_CITIES:
        return 1
    if clean in TIER_2_CITIES:
        return 2
    return 3

# Indian Occupations aligned with model training categories
OCCUPATIONS = [
    "Software Engineer", "Doctor", "Engineer", "Banker", "Accountant",
    "Teacher", "Lawyer", "Government Employee", "Data Analyst", "Businessman",
    "Consultant", "HR Manager", "Marketing Manager", "Sales Manager", "Architect",
    "Civil Servant", "Real Estate Agent", "Pharmacist", "Nurse", "Insurance Agent",
    "Chef", "Electrician", "Shop Owner", "Content Writer", "Lab Technician",
    "Carpenter", "Driver", "Factory Worker", "Plumber", "Retail Manager"
]

# 4. Main Tabs
tab_single, tab_batch = st.tabs(["Single Applicant", "Batch Upload"])

# ==============================================================================
# TAB 1: SINGLE APPLICANT
# ==============================================================================
with tab_single:
    render_html(
        f"""
        <div style="margin-bottom: 1.25rem;">
            <div style="font-size: 1.25rem; font-weight: 700; color: {THEME['TEXT']}; display: flex; align-items: center; gap: 0.5rem;">
                💰 Individual Applicant Underwriting Calculator
            </div>
            <div style="font-size: 0.875rem; color: {THEME['MUTED']}; margin-top: 0.25rem;">
                Predict risk-adjusted premium tiers and generate real-time actuarial policy briefs.
            </div>
        </div>
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=32,
            step=1,
            help="Applicant age in years",
        )
        weight_kg = st.number_input(
            "Weight (kg)",
            min_value=30.0,
            max_value=250.0,
            value=69.50,
            step=0.5,
            format="%.2f",
            help="Body weight in kilograms",
        )
        height_m = st.number_input(
            "Height (meters)",
            min_value=1.00,
            max_value=2.50,
            value=1.91,
            step=0.01,
            format="%.2f",
            help="Standing height in meters",
        )

    with col2:
        income_lpa = st.number_input(
            "Annual Income (Lakhs INR / LPA)",
            min_value=0.5,
            max_value=100.0,
            value=9.50,
            step=0.5,
            format="%.2f",
            help="Annual personal income in Lakhs Per Annum (INR)",
        )
        city = st.selectbox(
            "City of Residence",
            options=INDIAN_CITIES,
            index=INDIAN_CITIES.index("Agra") if "Agra" in INDIAN_CITIES else 0,
            help="Residential city for geographic risk tier assignment",
        )
        smoker_choice = st.selectbox(
            "Smoking Status",
            options=["Non-Smoker (0)", "Smoker (1)"],
            index=0,
            help="Tobacco smoking status",
        )

    with col3:
        occupation = st.selectbox(
            "Occupation",
            options=OCCUPATIONS,
            index=OCCUPATIONS.index("Software Engineer") if "Software Engineer" in OCCUPATIONS else 0,
            help="Primary profession",
        )

        # Live Real-time BMI Readout Card matching screenshot
        live_bmi = float(weight_kg) / (float(height_m) ** 2)
        if live_bmi < 18.5:
            bmi_status = "Underweight"
            bmi_color = THEME["MED"]
        elif live_bmi < 25.0:
            bmi_status = "Normal Weight"
            bmi_color = "#38bdf8"
        elif live_bmi < 30.0:
            bmi_status = "Overweight"
            bmi_color = THEME["MED"]
        else:
            bmi_status = "Obese"
            bmi_color = THEME["HIGH"]

        render_html(
            f"""
            <div style="
                background: rgba(15, 23, 42, 0.75);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 8px;
                padding: 1rem 1.25rem;
                margin-top: 0.35rem;
                backdrop-filter: blur(12px);
            ">
                <div style="font-size: 0.75rem; color: {THEME['MUTED']}; font-weight: 500; margin-bottom: 0.35rem;">
                    Calculated BMI
                </div>
                <div style="font-size: 1.65rem; font-weight: 700; color: {THEME['TEXT']}; letter-spacing: -0.02em; line-height: 1.2;">
                    {live_bmi:.1f} kg/m²
                </div>
                <div style="font-size: 0.85rem; color: {bmi_color}; font-weight: 500; margin-top: 0.35rem;">
                    {bmi_status}
                </div>
            </div>
            """
        )

    render_html("<div style='height: 0.5rem;'></div>")
    submitted = st.button(
        "⚡ Predict Risk Tier & Calculate Premium",
        use_container_width=True,
    )

    # Automatic evaluation on load or when button clicked
    city_tier_num = get_city_tier(city)
    smoker_num = 1 if "1" in smoker_choice else 0

    single_record = {
        "age": int(age),
        "weight": float(weight_kg),
        "height": float(height_m),
        "income_lpa": float(income_lpa),
        "occupation": occupation,
        "city_tier": int(city_tier_num),
        "smoker": int(smoker_num),
    }

    try:
        pred_result = predict_single(single_record)
        tier = pred_result.get("risk_tier", "Unknown")
        p_low = pred_result.get("prob_low", 0.0)
        p_med = pred_result.get("prob_medium", 0.0)
        p_high = pred_result.get("prob_high", 0.0)

        bmi_val = pred_result.get("bmi", live_bmi)
        hri_val = pred_result.get("health_risk_index", 0.0)
        inc_age_val = pred_result.get("income_per_age", 0.0)
        wh_val = pred_result.get("weight_height_ratio", 0.0)
        inc_bmi_val = pred_result.get("income_bmi", 0.0)
        smk_inc_val = pred_result.get("smoker_income", 0.0)
        if pred_result.get("reason"):
            st.info(f"ℹ️ Actuarial Engine Note: {pred_result.get('reason')}")

        section("Actuarial Risk Assessment", "Model classification and class probability distribution.")

        col_res_summary, col_res_chart = st.columns([1, 1])

        with col_res_summary:
            render_html(
                f"""
                <div class="glass-card" style="padding: 1.5rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="font-size: 0.775rem; text-transform: uppercase; color: {THEME['MUTED']}; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                            Predicted Risk Category
                        </div>
                        <div style="margin-bottom: 1.25rem;">
                            {risk_badge(tier, render=False)}
                        </div>
                        <div style="font-size: 0.85rem; color: {THEME['MUTED']}; line-height: 1.8;">
                            Health Risk Index: <b style="color: {THEME['TEXT']};">{hri_val:.2f}</b><br>
                            Calculated BMI: <b style="color: {THEME['TEXT']};">{bmi_val:.2f} kg/m²</b><br>
                            Income per Age: <b style="color: {THEME['TEXT']};">{inc_age_val:.2f} LPA/yr</b>
                        </div>
                    </div>
                    <div style="margin-top: 1.25rem; padding-top: 0.75rem; border-top: 1px solid rgba(255, 255, 255, 0.08); font-size: 0.8rem; color: {THEME['MUTED']};">
                        Classification confidence: <b style="color: {THEME['ACCENT']};">{max(p_low, p_med, p_high) * 100:.1f}%</b>
                    </div>
                </div>
                """
            )

        with col_res_chart:
            prob_df = pd.DataFrame({
                "Tier": ["Low Risk", "Medium Risk", "High Risk"],
                "Probability": [p_low, p_med, p_high],
                "Color": [THEME["LOW"], THEME["MED"], THEME["HIGH"]],
            })

            fig_probs = go.Figure(
                go.Bar(
                    x=prob_df["Probability"],
                    y=prob_df["Tier"],
                    orientation="h",
                    marker=dict(color=prob_df["Color"]),
                    text=[f"{p * 100:.1f}%" for p in prob_df["Probability"]],
                    textposition="auto",
                    textfont=dict(family="Plus Jakarta Sans, sans-serif", color="#ffffff", size=12),
                )
            )

            layout = get_plotly_layout()
            layout.update({
                "title": {
                    "text": "<b>Class Probability Distribution</b>",
                    "font": {"family": "Plus Jakarta Sans, sans-serif", "size": 14, "color": THEME["TEXT"]},
                },
                "xaxis": {
                    **layout["xaxis"],
                    "range": [0, 1.05],
                    "tickformat": ".0%",
                    "title": {"text": "Confidence", "font": {"family": "Plus Jakarta Sans, sans-serif", "color": THEME["MUTED"], "size": 11}},
                },
                "yaxis": {
                    **layout["yaxis"],
                    "autorange": "reversed",
                },
                "margin": dict(l=40, r=30, t=40, b=30),
                "height": 220,
            })
            fig_probs.update_layout(**layout)
            st.plotly_chart(fig_probs, use_container_width=True)

        # Actuarial Policy Brief Card
        render_html(
            f"""
            <div class="glass-card" style="padding: 1.25rem 1.5rem; margin-top: 1rem; border-left: 3px solid {THEME['ACCENT']};">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: {THEME['ACCENT']}; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 0.35rem;">
                    Actuarial Underwriting Brief
                </div>
                <div style="font-size: 0.9rem; color: {THEME['TEXT']}; line-height: 1.6;">
                    Applicant is a <b>{age}</b>-year-old <b>{occupation}</b> residing in <b>{city}</b> (Tier {city_tier_num} jurisdiction). 
                    Biometric screening yields a BMI of <b>{bmi_val:.1f} kg/m² ({bmi_status})</b> with <b>{'positive smoking status' if smoker_num == 1 else 'non-smoker profile'}</b>. 
                    Calculated Health Risk Index stands at <b>{hri_val:.2f}</b>, placing actuarial risk at the <b>{tier} Risk Tier</b> with <b>{max(p_low, p_med, p_high)*100:.1f}%</b> model confidence.
                </div>
            </div>
            """
        )

        # Engineered Features Breakdown
        section("Engineered Actuarial Features", "Mathematical indicators derived for Random Forest inference.")

        eng_kpis = [
            {"label": "BMI", "value": f"{bmi_val:.2f}", "delta": "kg/m²", "accent": "sky"},
            {"label": "Health Risk Index", "value": f"{hri_val:.2f}", "delta": "Index", "accent": "indigo"},
            {"label": "Income Per Age", "value": f"{inc_age_val:.2f}", "delta": "LPA/yr", "accent": "sky"},
            {"label": "Weight / Height", "value": f"{wh_val:.2f}", "delta": "kg/m", "accent": "emerald"},
            {"label": "Income × BMI", "value": f"{inc_bmi_val:.1f}", "delta": "Scale", "accent": "amber"},
            {"label": "Smoker × Income", "value": f"{smk_inc_val:.1f}", "delta": "LPA", "accent": "rose"},
        ]
        kpi_row(eng_kpis, cols=3)

        # Recommendations Section
        st.divider()
        section("📋 Recommendations")

        rec_features = {
            **single_record,
            "bmi": float(bmi_val),
            "health_risk_index": float(hri_val),
            "prob_low": float(p_low),
            "prob_medium": float(p_med),
            "prob_high": float(p_high),
            "gender": "Not Specified",
        }
        rec = recommend(tier, rec_features)

        # Headline card
        st.markdown(f"### {rec['headline']}")

        # Three columns
        c1, c2, c3 = st.columns(3)

        def _render_rec_card(title: str, detail: str) -> str:
            return (
                f"<div class='rec-card' style='background: rgba(15, 23, 42, 0.70); border: 1px solid rgba(56, 189, 248, 0.22); border-left: 4px solid #38bdf8; border-radius: 8px; padding: 11px 15px; margin: 8px 0; color: #f1f5f9; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);'>"
                f"<b style='color: #38bdf8; font-size: 0.9rem;'>{title}</b><br>"
                f"<span style='color: #cbd5e1; font-size: 0.825rem; line-height: 1.45;'>{detail}</span>"
                f"</div>"
            )

        with c1:
            st.markdown("**⚠️ Attention**")
            if rec["warnings"]:
                for w in rec["warnings"]:
                    st.markdown(_render_rec_card(w["title"], w["detail"]), unsafe_allow_html=True)
            else:
                st.info("No major warnings — keep it up!")

        with c2:
            st.markdown("**✅ Actions**")
            if rec["actions"]:
                for a in rec["actions"]:
                    st.markdown(_render_rec_card(a["title"], a["detail"]), unsafe_allow_html=True)

            st.markdown("**🔄 Maintain**")
            for m in rec["maintain"]:
                st.markdown(_render_rec_card(m["title"], m["detail"]), unsafe_allow_html=True)

        with c3:
            st.markdown("**💊 Suggested Riders**")
            for r in rec["riders"]:
                st.markdown(f"- {r}")

            st.markdown("**📝 Checklist**")
            for i, item in enumerate(rec["checklist"], 1):
                st.checkbox(item, key=f"chk_{i}")

        # Download Buttons: CSV & Recommendation Report TXT
        render_html("<div style='height: 1rem;'></div>")
        col_dl_pred, col_dl_rec = st.columns(2)
        with col_dl_pred:
            result_df = pd.DataFrame([pred_result])
            download_button(
                df=result_df,
                filename="applicant_prediction.csv",
                label="📥 Download this prediction as CSV",
                key="download_single_prediction",
                filetype="csv",
            )
        with col_dl_rec:
            st.download_button(
                "Download recommendation report (TXT)",
                data=_recommendations_to_text(rec, rec_features),
                file_name=f"recommendation_{tier.lower()}.txt",
                mime="text/plain",
                key="download_single_recommendation_txt",
                use_container_width=True,
            )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


# ==============================================================================
# TAB 2: BATCH UPLOAD
# ==============================================================================
with tab_batch:
    section("Batch Underwriting", "Score cohorts of applicants from CSV or Excel datasets.")

    session_dataset = get_dataset()
    dataset_name = st.session_state.get("dataset_name", "user_dataset")

    # Display active dataset banner if present in session state
    if session_dataset is not None and not session_dataset.empty:
        col_msg, col_reup = st.columns([4, 1])
        with col_msg:
            render_html(
                f"""
                <div class="glass-card" style="padding: 0.85rem 1.15rem; display: flex; align-items: center; gap: 0.75rem;">
                    <span style="width: 10px; height: 10px; border-radius: 50%; background: {THEME['LOW']}; box-shadow: 0 0 8px {THEME['LOW']};"></span>
                    <div>
                        <b style="color: {THEME['TEXT']};">Using previously uploaded dataset:</b>
                        <span style="color: {THEME['ACCENT']}; font-weight: 600; margin-left: 0.35rem;">{dataset_name}</span>
                        <span style="color: {THEME['MUTED']}; font-size: 0.85rem; margin-left: 0.5rem;">({len(session_dataset):,} rows)</span>
                    </div>
                </div>
                """
            )
        with col_reup:
            if st.button("Re-upload / Clear", key="batch_reupload_btn", use_container_width=True):
                reset_dataset()
                st.rerun()

    uploaded_file = st.file_uploader(
        "Upload applicant dataset (.csv, .xlsx, .xls)",
        type=["csv", "xlsx", "xls"],
        key="batch_file_uploader",
        help="Upload applicant files. Normalization automatically aligns column names.",
    )

    batch_df = None

    if uploaded_file is not None:
        try:
            with st.spinner("Reading and preprocessing file..."):
                raw_df = read_uploaded_file(uploaded_file)
                normalized_df = normalize_columns(raw_df)
                batch_df = coerce_types(normalized_df)
                store_dataset(batch_df, name=uploaded_file.name)
        except Exception as exc:
            st.error(f"Failed to process uploaded file: {exc}")
    elif session_dataset is not None and not session_dataset.empty:
        batch_df = session_dataset

    if batch_df is not None and not batch_df.empty:
        # Display Cohort KPI Row
        section("Cohort Overview", "Summary profile of the active batch dataset.")
        total_cells = batch_df.size or 1
        missing_count = batch_df.isna().sum().sum()
        missing_pct = (missing_count / total_cells) * 100
        roles = detect_column_roles(batch_df)

        cohort_kpis = [
            {"label": "Total Rows", "value": f"{len(batch_df):,}", "delta": "Applicants", "accent": "sky"},
            {"label": "Columns", "value": f"{len(batch_df.columns)}", "delta": "Features", "accent": "indigo"},
            {"label": "Missing Values", "value": f"{missing_pct:.1f}%", "delta": f"{missing_count} cells", "accent": "amber" if missing_pct > 0 else "emerald"},
            {"label": "Numeric Features", "value": f"{len(roles['numeric'])}", "delta": "Quantitative", "accent": "sky"},
        ]
        kpi_row(cohort_kpis, cols=4)

        # Scored Prediction Run
        section("Model Scoring", "Batch inference generated using Random Forest pipeline.")
        with st.spinner("Scoring dataset with actuarial risk model..."):
            scored_df = predict_dataframe(batch_df)
            if "reason" in scored_df.columns and scored_df["reason"].isna().all():
                scored_df = scored_df.drop(columns=["reason"])

            # Compute row-level recommendations
            recs_list = []
            for _, r_series in scored_df.iterrows():
                r_dict = r_series.to_dict()
                t = str(r_dict.get("risk_tier", "Unknown")).capitalize()
                recs_list.append(recommend(t, r_dict))

            scored_df["Recommendation"] = [r["headline"] for r in recs_list]

        # Plotly Donut Chart of Risk Tier Distribution
        col_donut, col_stats = st.columns([1, 1])

        tier_counts = scored_df["risk_tier"].value_counts()
        tier_colors = {
            "Low": THEME["LOW"],
            "Medium": THEME["MED"],
            "High": THEME["HIGH"],
            "Unknown": THEME["MUTED"],
        }
        donut_colors = [tier_colors.get(t, THEME["ACCENT"]) for t in tier_counts.index]

        with col_donut:
            fig_donut = go.Figure(
                go.Pie(
                    labels=tier_counts.index,
                    values=tier_counts.values,
                    hole=0.55,
                    marker=dict(colors=donut_colors),
                    textinfo="label+percent",
                    textfont=dict(family="Plus Jakarta Sans, sans-serif", size=12),
                    hoverinfo="label+value+percent",
                )
            )
            layout_donut = get_plotly_layout()
            layout_donut.update({
                "title": {
                    "text": "<b>Risk Tier Distribution</b>",
                    "font": {"family": "Plus Jakarta Sans, sans-serif", "size": 14, "color": THEME["TEXT"]},
                },
                "margin": dict(l=30, r=30, t=40, b=30),
                "height": 280,
                "showlegend": True,
            })
            fig_donut.update_layout(**layout_donut)
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_stats:
            low_c = tier_counts.get("Low", 0)
            med_c = tier_counts.get("Medium", 0)
            high_c = tier_counts.get("High", 0)
            total_c = len(scored_df) or 1

            render_html(
                f"""
                <div class="glass-card" style="padding: 1.25rem; height: 280px; display: flex; flex-direction: column; justify-content: space-around;">
                    <div style="font-size: 0.775rem; text-transform: uppercase; color: {THEME['MUTED']}; font-weight: 600;">Distribution Breakdown</div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: {THEME['LOW']}; font-weight: 600;">🟢 Low Risk</span>
                        <span style="color: {THEME['TEXT']}; font-weight: 700;">{low_c:,} ({low_c / total_c * 100:.1f}%)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: {THEME['MED']}; font-weight: 600;">🟡 Medium Risk</span>
                        <span style="color: {THEME['TEXT']}; font-weight: 700;">{med_c:,} ({med_c / total_c * 100:.1f}%)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: {THEME['HIGH']}; font-weight: 600;">🔴 High Risk</span>
                        <span style="color: {THEME['TEXT']}; font-weight: 700;">{high_c:,} ({high_c / total_c * 100:.1f}%)</span>
                    </div>
                    <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 0.5rem; font-size: 0.8rem; color: {THEME['MUTED']};">
                        Total Cohort Records Scored: <b style="color: {THEME['TEXT']};">{total_c:,}</b>
                    </div>
                </div>
                """
            )

        # Full Scored Results Table with column_config
        section("Scored Applicant Records", "Interactive dataset table with probability estimates and tailored recommendations.")

        col_config = {
            "risk_tier": st.column_config.TextColumn(
                "Risk Tier",
                help="Actuarial Risk Classification (Low / Medium / High)",
            ),
            "Recommendation": st.column_config.TextColumn(
                "Recommendation",
                help="Tailored actuarial recommendation headline",
            ),
            "prob_low": st.column_config.ProgressColumn(
                "P(Low)",
                help="Probability of Low Premium Tier",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
            "prob_medium": st.column_config.ProgressColumn(
                "P(Medium)",
                help="Probability of Medium Premium Tier",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
            "prob_high": st.column_config.ProgressColumn(
                "P(High)",
                help="Probability of High Premium Tier",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
        }

        st.dataframe(
            scored_df,
            column_config=col_config,
            height=500,
            use_container_width=True,
        )

        # View recommendations expander BELOW the table
        with st.expander("🔍 View recommendations", expanded=False):
            sel_idx = st.selectbox(
                "Select Applicant Row Index to Inspect",
                options=list(range(len(scored_df))),
                format_func=lambda i: (
                    f"Row {i + 1} — {scored_df.iloc[i].get('risk_tier', 'Unknown')} Risk "
                    f"({scored_df.iloc[i].get('Recommendation', '')[:45]}...)"
                ),
                key="sb_batch_recommendation_row",
            )
            sel_rec = recs_list[sel_idx]

            st.markdown(f"### {sel_rec['headline']}")
            b_c1, b_c2, b_c3 = st.columns(3)

            with b_c1:
                st.markdown("**⚠️ Attention**")
                if sel_rec["warnings"]:
                    for w in sel_rec["warnings"]:
                        st.markdown(_render_rec_card(w["title"], w["detail"]), unsafe_allow_html=True)
                else:
                    st.info("No major warnings — keep it up!")

            with b_c2:
                st.markdown("**✅ Actions**")
                if sel_rec["actions"]:
                    for a in sel_rec["actions"]:
                        st.markdown(_render_rec_card(a["title"], a["detail"]), unsafe_allow_html=True)

                st.markdown("**🔄 Maintain**")
                for m in sel_rec["maintain"]:
                    st.markdown(_render_rec_card(m["title"], m["detail"]), unsafe_allow_html=True)

            with b_c3:
                st.markdown("**💊 Suggested Riders**")
                for r in sel_rec["riders"]:
                    st.markdown(f"- {r}")

                st.markdown("**📝 Checklist**")
                for i, item in enumerate(sel_rec["checklist"], 1):
                    st.checkbox(item, key=f"batch_chk_{sel_idx}_{i}")

        # Download recommendations for all rows (CSV)
        rec_export_rows = []
        for i, (_, row) in enumerate(scored_df.iterrows()):
            r_dict = row.to_dict()
            t = str(r_dict.get("risk_tier", "Unknown")).capitalize()
            r = recs_list[i]
            p_id = (
                r_dict.get("patient_id")
                or r_dict.get("id")
                or r_dict.get("applicant_id")
                or f"Applicant_{i + 1:04d}"
            )
            top_act = (
                r["actions"][0]["title"]
                if r["actions"]
                else (r["maintain"][0]["title"] if r["maintain"] else "None")
            )
            chk_str = "; ".join(r["checklist"])
            rec_export_rows.append({
                "patient_id": p_id,
                "risk_tier": t,
                "headline": r["headline"],
                "top_action": top_act,
                "checklist": chk_str,
            })
        rec_export_df = pd.DataFrame(rec_export_rows)

        # Download Buttons
        section("Export Results", "Download full scored cohort dataset and actuarial recommendations.")
        col_dl_csv, col_dl_xlsx, col_dl_rec = st.columns(3)

        with col_dl_csv:
            download_button(
                df=scored_df,
                filename="scored_applicants.csv",
                label="📥 Download Scored CSV",
                key="batch_download_scored_csv",
                filetype="csv",
            )

        with col_dl_xlsx:
            download_button(
                df=scored_df,
                filename="scored_applicants.xlsx",
                label="📊 Download Scored Excel",
                key="batch_download_scored_xlsx",
                filetype="xlsx",
            )

        with col_dl_rec:
            st.download_button(
                "📋 Download recommendations for all rows (CSV)",
                data=rec_export_df.to_csv(index=False),
                file_name="cohort_recommendations.csv",
                mime="text/csv",
                key="batch_download_recommendations_csv",
                use_container_width=True,
            )
    else:
        empty_state(
            icon="📁",
            title="No Batch Dataset Loaded",
            message="Upload a CSV or Excel applicant dataset above to run cohort underwriting predictions.",
        )
