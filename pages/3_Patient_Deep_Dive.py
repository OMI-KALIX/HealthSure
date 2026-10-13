"""Patient Deep Dive page for HealthSure: universal clinical data ingestion, column semantic classification, and longitudinal exploration."""
from typing import Any, Dict, List, Optional
import math
import re
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from core.theme import THEME, apply_theme
from core.ui import (
    page_header,
    section,
    kpi_card,
    kpi_row,
    get_plotly_layout,
    threshold_line_chart,
    timeline_gantt,
    sidebar_footer,
    render_html,
)
from core.patient_io import (
    load_any_csv,
    classify_columns,
    classify_row_columns,
    expand_long_format,
    shape_dataset,
)
from core.patient_clinical import (
    bmi,
    aha_bp_category,
    lab_status,
    charlson_like_score,
    cardiovascular_risk_factors,
    clinical_narrative,
    drug_class,
    safe_get,
    describe_dataset,
    LAB_REF_TABLE,
)
from core.patient_ui import (
    patient_header_card,
    narrative_card,
    latest_value_card,
    diagnosis_timeline,
    empty_patient_state,
)
from core.patient_pdf import generate_patient_pdf

# 1. Page Configuration
st.set_page_config(
    page_title="Patient Deep Dive | HealthSure",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

# 2. Hero Header
page_header(
    title="🩺 Single Patient Deep Dive",
    subtitle="Universal clinical record ingestion. Upload any CSV or Excel file to inspect patient profiles, vitals, labs, conditions, and encounters without format restrictions.",
)

# 3. File Ingestion Widget
uploaded_files = st.file_uploader(
    "Upload patient clinical record (.csv, .xlsx, .xls)",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    key="patient_uploader_widget",
    help="Upload any clinical dataset. Columns are semantically classified automatically.",
)

# Ingest and shape dataset on upload
if uploaded_files:
    dfs: List[pd.DataFrame] = []
    warnings_list: List[str] = []

    for f in uploaded_files:
        try:
            loaded_df = load_any_csv(f)
            if loaded_df.empty:
                warnings_list.append(f"'{f.name}' is empty.")
            else:
                dfs.append(loaded_df)
        except Exception as exc:
            warnings_list.append(f"Could not parse '{f.name}': {exc}")

    if dfs:
        shaped = shape_dataset(dfs if len(dfs) > 1 else dfs[0])
        if warnings_list:
            shaped["_warnings"].extend(warnings_list)
        st.session_state["patient_shaped"] = shaped
    elif warnings_list:
        st.session_state["patient_shaped"] = {
            "demographics": None,
            "vitals": None,
            "labs": None,
            "diagnoses": None,
            "medications": None,
            "encounters": None,
            "_summary": {
                "rows": 0,
                "cols": 0,
                "categories_found": [],
                "categories_missing": [
                    "demographics", "vitals", "labs", "diagnoses", "medications", "encounters"
                ],
            },
            "_warnings": warnings_list,
        }

# 4. Sidebar: Dataset Summary & Reset
shaped_data = st.session_state.get("patient_shaped")
pdf_bytes: Optional[bytes] = None
pdf_filename = "HealthSure_Clinical_Report.pdf"
pdf_gen_error: Optional[str] = None

if shaped_data:
    try:
        pdf_bytes = generate_patient_pdf(shaped_data)
        demo_sub = shaped_data.get("demographics")
        p_sub = demo_sub.iloc[0] if (demo_sub is not None and isinstance(demo_sub, pd.DataFrame) and not demo_sub.empty) else None
        p_sub_name = safe_get(p_sub, "name", "full_name", "patient_name", "id", default="Patient")
        clean_sub_name = re.sub(r"[^\w\s-]", "", str(p_sub_name)).strip().replace(" ", "_")
        pdf_filename = f"HealthSure_Clinical_Report_{clean_sub_name}.pdf"
    except Exception as exc:
        pdf_bytes = None
        pdf_gen_error = str(exc)

with st.sidebar:
    section("Dataset summary")
    if pdf_gen_error:
        st.warning(f"Report generation note: {pdf_gen_error}")
    if shaped_data:
        summary = shaped_data.get("_summary", {})
        rows_count = summary.get("rows", 0)
        cols_count = summary.get("cols", 0)
        found_cats = summary.get("categories_found", [])
        missing_cats = summary.get("categories_missing", [])
        warnings_found = shaped_data.get("_warnings", [])

        st.write(f"**Total rows:** {rows_count:,}")
        st.write(f"**Total columns:** {cols_count:,}")
        st.write(f"**Categories found:** {len(found_cats)}")
        if found_cats:
            cats_display = ", ".join(c.capitalize() for c in found_cats)
            st.caption(f"Active: {cats_display}")

        if warnings_found:
            with st.expander(f"⚠️ Ingestion Warnings ({len(warnings_found)})", expanded=False):
                for w in warnings_found:
                    st.caption(f"• {w}")

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        if pdf_bytes:
            st.download_button(
                label="📥 Download Clinical PDF",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                key="btn_download_patient_pdf_sidebar",
                use_container_width=True,
            )

        if st.button("Reset dataset", key="btn_reset_patient_shaped", use_container_width=True):
            st.session_state.pop("patient_shaped", None)
            st.rerun()
    else:
        st.caption("No patient dataset loaded in session memory.")

    sidebar_footer()

# 5. Check if any dataset is loaded
if not shaped_data:
    empty_patient_state(
        "No patient record loaded yet. Upload a CSV or Excel file above to begin clinical analysis."
    )
else:
    summary = shaped_data.get("_summary", {})
    found_cats = summary.get("categories_found", [])

    # If 0 categories were recognized, show helpful guide (do NOT st.stop)
    if not found_cats:
        empty_patient_state(
            "No recognized clinical categories were detected in the uploaded file.\n\n"
            "💡 Hint: To populate clinical tabs, include columns such as 'name', 'age', 'gender', "
            "'systolic', 'diastolic', 'glucose', 'cholesterol', 'diagnosis', 'icd10', 'medication', or 'visit_type'."
        )

    # Extract clinical dataframes from shaped dict
    demo_df = shaped_data.get("demographics")
    patient_record = demo_df.iloc[0] if (demo_df is not None and isinstance(demo_df, pd.DataFrame) and not demo_df.empty) else None

    vitals_df = shaped_data.get("vitals")
    labs_df = shaped_data.get("labs")
    diag_df = shaped_data.get("diagnoses")
    meds_df = shaped_data.get("medications")
    encounters_df = shaped_data.get("encounters")

    # Compute Demographic Snapshot
    h_val = safe_get(patient_record, "height", "height_cm", "height_m", "height_in")
    w_val = safe_get(patient_record, "weight", "weight_kg", "weight_lbs")
    bmi_val, bmi_cat = bmi(h_val, w_val)

    if isinstance(diag_df, pd.DataFrame) and not diag_df.empty:
        status_col = next((c for c in diag_df.columns if "status" in str(c).lower()), None)
        active_conds = diag_df[diag_df[status_col].astype(str).str.lower() != "resolved"] if status_col else diag_df
        n_active_conds = len(active_conds)
    else:
        n_active_conds = 0

    n_meds = len(meds_df) if isinstance(meds_df, pd.DataFrame) and not meds_df.empty else 0

    # Extract latest clinical date if available
    last_visit_date = None
    for candidate_df in (encounters_df, vitals_df, diag_df):
        if isinstance(candidate_df, pd.DataFrame) and not candidate_df.empty:
            date_col = next((c for c in candidate_df.columns if any(k in str(c).lower() for k in ("date", "time"))), None)
            if date_col:
                dt_parsed = pd.to_datetime(candidate_df[date_col], errors="coerce").dropna()
                if not dt_parsed.empty:
                    last_visit_date = dt_parsed.max()
                    break

    # Action Bar: Clinical Record Summary & Primary PDF Download Button
    col_action_meta, col_action_dl = st.columns([3, 1])
    with col_action_meta:
        render_html(
            f"<div style='font-size:0.875rem; color:{THEME['MUTED']}; padding-top:0.35rem;'>"
            f"📄 <b>Clinical Profile:</b> {describe_dataset(shaped_data)}"
            f"</div>"
        )
    with col_action_dl:
        if pdf_bytes:
            st.download_button(
                label="📥 Download Clinical PDF",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                key="btn_download_patient_pdf_main",
                use_container_width=True,
            )

    render_html("<div style='height:0.5rem;'></div>")

    # 6. Render Five Clinical Tabs
    t1, t2, t3, t4, t5 = st.tabs([
        "📋 Overview",
        "❤️ Vitals",
        "🧪 Labs",
        "💊 Diagnoses & Medications",
        "🏥 Encounters",
    ])

    # ==========================================================================
    # TAB 1: OVERVIEW
    # ==========================================================================
    with t1:
        # Header Card always renders with whatever demographic fields exist
        patient_header_card(
            patient=patient_record if patient_record is not None else {},
            bmi_val=bmi_val,
            bmi_cat=bmi_cat,
            active_conditions=n_active_conds,
            n_meds=n_meds,
            last_visit=last_visit_date,
        )

        # Narrative Synthesis with safe defaults
        narrative_text = clinical_narrative(
            patient=patient_record,
            vitals=vitals_df,
            labs=labs_df,
            diagnoses=diag_df,
            medications=meds_df,
            encounters=encounters_df,
        )
        narrative_card(narrative_text)

        c1, c2, c3 = st.columns(3)

        # Column 1: Latest Vitals
        with c1:
            render_html(
                f"<div style='font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: {THEME['ACCENT']}; margin-bottom: 0.75rem;'>❤️ Latest Vitals</div>"
            )
            if isinstance(vitals_df, pd.DataFrame) and not vitals_df.empty:
                latest_v = vitals_df.iloc[-1]
                sys_col = next((c for c in vitals_df.columns if "systolic" in str(c).lower() or c == "sbp"), None)
                dia_col = next((c for c in vitals_df.columns if "diastolic" in str(c).lower() or c == "dbp"), None)

                if sys_col and dia_col and pd.notna(latest_v.get(sys_col)) and pd.notna(latest_v.get(dia_col)):
                    sys_v = latest_v[sys_col]
                    dia_v = latest_v[dia_col]
                    bp_tier = aha_bp_category(sys_v, dia_v)
                    try:
                        latest_value_card(
                            label="Blood Pressure",
                            value=f"{int(float(sys_v))}/{int(float(dia_v))}",
                            unit="mmHg",
                            status=bp_tier,
                            hint="AHA Target: < 120/80 mmHg",
                        )
                    except Exception:
                        pass

                hr_col = next((c for c in vitals_df.columns if any(k in str(c).lower() for k in ("heart", "pulse", "hr"))), None)
                if hr_col and pd.notna(latest_v.get(hr_col)):
                    try:
                        hr_v = float(latest_v[hr_col])
                        hr_status = "Normal" if 60 <= hr_v <= 100 else ("Low" if hr_v < 60 else "High")
                        latest_value_card(
                            label="Heart Rate",
                            value=f"{int(hr_v)}",
                            unit="bpm",
                            status=hr_status,
                            hint="Normal: 60 - 100 bpm",
                        )
                    except Exception:
                        pass

                spo2_col = next((c for c in vitals_df.columns if any(k in str(c).lower() for k in ("spo2", "oxygen", "o2"))), None)
                if spo2_col and pd.notna(latest_v.get(spo2_col)):
                    try:
                        spo2_v = float(latest_v[spo2_col])
                        spo2_status = "Normal" if spo2_v >= 95 else ("Critical" if spo2_v < 90 else "Low")
                        latest_value_card(
                            label="Oxygen Saturation",
                            value=f"{spo2_v:.1f}",
                            unit="%",
                            status=spo2_status,
                            hint="Normal: ≥ 95%",
                        )
                    except Exception:
                        pass
            else:
                empty_patient_state(
                    "No vital signs recorded on file.\n\n"
                    "Add columns like 'systolic', 'diastolic', 'heart_rate', or 'spo2' to view vital metrics."
                )

        # Column 2: Latest Labs
        with c2:
            render_html(
                f"<div style='font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: {THEME['ACCENT']}; margin-bottom: 0.75rem;'>🧪 Latest Labs</div>"
            )
            if isinstance(labs_df, pd.DataFrame) and not labs_df.empty:
                test_col = next((c for c in labs_df.columns if any(k in str(c).lower() for k in ("test", "analyte", "lab", "name"))), None)
                val_col = next((c for c in labs_df.columns if any(k in str(c).lower() for k in ("value", "result", "val"))), None)

                if test_col and val_col:
                    unique_tests = labs_df[test_col].dropna().unique()[:3]
                    for t_name in unique_tests:
                        t_rows = labs_df[labs_df[test_col] == t_name]
                        l_record = t_rows.iloc[-1]
                        l_val = l_record[val_col]
                        status = lab_status(t_name, l_val)
                        unit_col = next((c for c in labs_df.columns if "unit" in str(c).lower()), None)
                        unit_str = str(l_record[unit_col]) if unit_col and pd.notna(l_record.get(unit_col)) else ""
                        clean_test = str(t_name).replace("_", " ").title()

                        ref_k = next((k for k in LAB_REF_TABLE if k in str(t_name).lower()), None)
                        hint_str = f"Ref: {LAB_REF_TABLE[ref_k][0]} - {LAB_REF_TABLE[ref_k][1]}" if ref_k else None

                        latest_value_card(
                            label=clean_test,
                            value=str(l_val),
                            unit=unit_str,
                            status=status,
                            hint=hint_str,
                        )
                else:
                    # Wide columns layout check
                    rendered_any = False
                    for k, (low, high) in LAB_REF_TABLE.items():
                        matched_col = next((c for c in labs_df.columns if k in str(c).lower()), None)
                        if matched_col and pd.notna(labs_df[matched_col].iloc[-1]):
                            val = labs_df[matched_col].iloc[-1]
                            status = lab_status(k, val)
                            latest_value_card(
                                label=matched_col.replace("_", " ").title(),
                                value=str(val),
                                unit="",
                                status=status,
                                hint=f"Ref: {low} - {high}",
                            )
                            rendered_any = True
                    if not rendered_any:
                        empty_patient_state("Lab columns could not be identified.")
            else:
                empty_patient_state(
                    "No laboratory records on file.\n\n"
                    "Add columns like 'glucose', 'hba1c', 'creatinine', or 'cholesterol' to view lab analytes."
                )

        # Column 3: Risk Flags & Comorbidities
        with c3:
            render_html(
                f"<div style='font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: {THEME['ACCENT']}; margin-bottom: 0.75rem;'>⚠️ Risk & Comorbidities</div>"
            )
            if isinstance(diag_df, pd.DataFrame) and not diag_df.empty:
                charlson_score = charlson_like_score(diag_df)
                c_status = "Normal" if charlson_score == 0 else ("Low" if charlson_score == 1 else ("High" if charlson_score == 2 else "Critical"))
                latest_value_card(
                    label="Charlson Comorbidity Score",
                    value=str(charlson_score),
                    unit="pts",
                    status=c_status,
                    hint="Weighted multi-condition complexity index",
                )
                cv_factors = cardiovascular_risk_factors(patient_record, diag_df, labs_df)
            else:
                cv_factors = cardiovascular_risk_factors(patient_record, pd.DataFrame(), labs_df)
                latest_value_card(
                    label="Clinical Conditions",
                    value="0",
                    unit="dx",
                    status="Normal",
                    hint="No conditions on record.",
                )

            if cv_factors:
                pills = "".join(
                    f'<span style="display:inline-flex;align-items:center;background:{THEME["HIGH"]}22;color:{THEME["HIGH"]};border:1px solid {THEME["HIGH"]}55;padding:0.25rem 0.65rem;border-radius:9999px;font-size:0.775rem;font-weight:600;margin:0.2rem;">● {factor}</span>'
                    for factor in cv_factors
                )
            else:
                pills = f'<span style="color:{THEME["MUTED"]};font-size:0.85rem;">No major cardiovascular risk factors detected.</span>'

            render_html(
                f'<div class="glass-card" style="padding:1.15rem;margin-top:0.75rem;">'
                f'<div style="font-size:0.75rem;text-transform:uppercase;color:{THEME["MUTED"]};font-weight:600;margin-bottom:0.5rem;">'
                f'Identified Cardiovascular Factors ({len(cv_factors)})'
                f'</div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:0.2rem;">'
                f'{pills}'
                f'</div>'
                f'</div>'
            )

        # Overview Tab: Bottom Clinical Report Download Card
        render_html(
            f"<div class='glass-card' style='padding:1.25rem; margin-top:1.5rem;'>"
            f"<div style='font-size:1rem; font-weight:700; color:{THEME['TEXT']};'>📄 Comprehensive Clinical Deep Dive Report</div>"
            f"<div style='font-size:0.85rem; color:{THEME['MUTED']}; margin-top:0.25rem;'>"
            f"Download an executive multi-page ReportLab PDF synthesizing demographics, executive narrative, vitals, labs, diagnoses, medications, and healthcare utilization."
            f"</div>"
            f"</div>"
        )
        if pdf_bytes:
            st.download_button(
                label="📥 Download Clinical PDF Report (Full Record)",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                key="btn_download_patient_pdf_tab1",
                use_container_width=True,
            )

    # ==========================================================================
    # TAB 2: VITALS
    # ==========================================================================
    with t2:
        if vitals_df is None or not isinstance(vitals_df, pd.DataFrame) or vitals_df.empty:
            empty_patient_state(
                "No vitals records available in this dataset.\n\n"
                "💡 Hint: Add columns like 'systolic', 'diastolic', 'heart_rate', or 'spo2' to plot vital sign trends."
            )
        else:
            v_work = vitals_df.copy()
            date_col = next((c for c in v_work.columns if any(k in str(c).lower() for k in ("date", "time"))), None)

            if date_col and not pd.api.types.is_datetime64_any_dtype(v_work[date_col]):
                try:
                    v_work[date_col] = pd.to_datetime(v_work[date_col], errors="coerce")
                except Exception:
                    pass

            if not date_col:
                v_work["measurement_index"] = range(1, len(v_work) + 1)
                date_col = "measurement_index"

            # Sort chronologically if date exists
            try:
                v_work = v_work.sort_values(date_col)
            except Exception:
                pass

            sys_col = next((c for c in v_work.columns if "systolic" in str(c).lower() or c == "sbp"), None)
            dia_col = next((c for c in v_work.columns if "diastolic" in str(c).lower() or c == "dbp"), None)

            # Blood Pressure Longitudinal Chart
            if sys_col and dia_col:
                v_work[sys_col] = pd.to_numeric(v_work[sys_col], errors="coerce")
                v_work[dia_col] = pd.to_numeric(v_work[dia_col], errors="coerce")
                valid_bp = v_work.dropna(subset=[sys_col, dia_col])

                if not valid_bp.empty:
                    section("Blood Pressure Longitudinal Trend", "Systolic and diastolic trajectory with AHA target reference zones.")

                    fig_bp = go.Figure()
                    fig_bp.add_trace(
                        go.Scatter(
                            x=valid_bp[date_col],
                            y=valid_bp[sys_col],
                            name="Systolic BP",
                            mode="lines+markers",
                            line=dict(color=THEME["HIGH"], width=3),
                            marker=dict(size=7, color=THEME["HIGH"]),
                        )
                    )
                    fig_bp.add_trace(
                        go.Scatter(
                            x=valid_bp[date_col],
                            y=valid_bp[dia_col],
                            name="Diastolic BP",
                            mode="lines+markers",
                            line=dict(color=THEME["ACCENT"], width=3),
                            marker=dict(size=7, color=THEME["ACCENT"]),
                        )
                    )

                    fig_bp.add_hrect(
                        y0=0,
                        y1=120,
                        fillcolor="rgba(16, 185, 129, 0.08)",
                        line_width=0,
                        annotation_text="Normal SBP (< 120)",
                        annotation_position="top left",
                        annotation_font=dict(color=THEME["LOW"], size=10),
                    )
                    fig_bp.add_hrect(
                        y0=120,
                        y1=139,
                        fillcolor="rgba(245, 158, 11, 0.08)",
                        line_width=0,
                        annotation_text="Elevated / Stage 1 (120-139)",
                        annotation_position="top left",
                        annotation_font=dict(color=THEME["MED"], size=10),
                    )

                    bp_layout = get_plotly_layout()
                    bp_layout.update({
                        "title": {"text": "<b>Blood Pressure Trajectory (mmHg)</b>", "font": {"family": "Plus Jakarta Sans, sans-serif", "size": 14, "color": THEME["TEXT"]}},
                        "yaxis": {"title": {"text": "mmHg", "font": {"family": "Plus Jakarta Sans, sans-serif", "size": 11, "color": THEME["MUTED"]}}},
                        "height": 380,
                    })
                    fig_bp.update_layout(**bp_layout)
                    st.plotly_chart(fig_bp, use_container_width=True)

            # Heart Rate & SpO2
            hr_col = next((c for c in v_work.columns if any(k in str(c).lower() for k in ("heart", "pulse", "hr"))), None)
            spo2_col = next((c for c in v_work.columns if any(k in str(c).lower() for k in ("spo2", "oxygen", "o2"))), None)

            if hr_col or spo2_col:
                col_hr, col_spo2 = st.columns(2)
                with col_hr:
                    if hr_col:
                        v_work[hr_col] = pd.to_numeric(v_work[hr_col], errors="coerce")
                        valid_hr = v_work.dropna(subset=[hr_col])
                        if not valid_hr.empty:
                            threshold_line_chart(
                                df=valid_hr,
                                x=date_col,
                                y=hr_col,
                                bands=[
                                    (60, 100, THEME["LOW"], "Normal (60-100 bpm)"),
                                    (100, 150, THEME["HIGH"], "Tachycardia (>100 bpm)"),
                                ],
                                title="Heart Rate History",
                                y_label="bpm",
                            )
                with col_spo2:
                    if spo2_col:
                        v_work[spo2_col] = pd.to_numeric(v_work[spo2_col], errors="coerce")
                        valid_spo2 = v_work.dropna(subset=[spo2_col])
                        if not valid_spo2.empty:
                            threshold_line_chart(
                                df=valid_spo2,
                                x=date_col,
                                y=spo2_col,
                                bands=[
                                    (95, 100, THEME["LOW"], "Target (≥95%)"),
                                    (80, 94.9, THEME["HIGH"], "Hypoxia (<95%)"),
                                ],
                                title="Oxygen Saturation History",
                                y_label="SpO2 (%)",
                            )

            section("Vitals Dataset Table", "Tabular view of all recorded clinical measurements.")
            st.dataframe(v_work, use_container_width=True)

    # ==========================================================================
    # TAB 3: LABS
    # ==========================================================================
    with t3:
        if labs_df is None or not isinstance(labs_df, pd.DataFrame) or labs_df.empty:
            empty_patient_state(
                "No laboratory records available in this dataset.\n\n"
                "💡 Hint: Add columns like 'glucose', 'hba1c', 'creatinine', 'cholesterol', or 'hemoglobin' to track analyte trends."
            )
        else:
            l_work = labs_df.copy()
            date_col = next((c for c in l_work.columns if any(k in str(c).lower() for k in ("date", "time"))), None)

            if date_col and not pd.api.types.is_datetime64_any_dtype(l_work[date_col]):
                try:
                    l_work[date_col] = pd.to_datetime(l_work[date_col], errors="coerce")
                except Exception:
                    pass

            if not date_col:
                l_work["test_order"] = range(1, len(l_work) + 1)
                date_col = "test_order"

            test_col = next((c for c in l_work.columns if any(k in str(c).lower() for k in ("test", "analyte", "lab", "name"))), None)
            val_col = next((c for c in l_work.columns if any(k in str(c).lower() for k in ("value", "result", "val"))), None)

            if test_col and val_col:
                test_names = l_work[test_col].dropna().unique()
                selected_test = st.selectbox(
                    "Select Analyte to Inspect",
                    options=test_names,
                    key="sb_select_lab_test",
                )

                if selected_test:
                    sub_test = l_work[l_work[test_col] == selected_test].copy()
                    ref_key = next((k for k in LAB_REF_TABLE if k in str(selected_test).lower()), None)
                    bands = []
                    if ref_key:
                        r_low, r_high = LAB_REF_TABLE[ref_key]
                        bands = [(r_low, r_high, THEME["LOW"], f"Ref: {r_low}-{r_high}")]

                    sub_test[val_col] = pd.to_numeric(sub_test[val_col], errors="coerce")
                    valid_sub = sub_test.dropna(subset=[val_col])

                    if not valid_sub.empty:
                        threshold_line_chart(
                            df=valid_sub,
                            x=date_col,
                            y=val_col,
                            bands=bands,
                            title=f"{str(selected_test).replace('_', ' ').title()} Longitudinal Trajectory",
                            y_label="Value",
                        )
            else:
                # Wide format labs: columns are individual analytes
                analyte_cols = [
                    c for c in l_work.columns
                    if c not in (date_col, "patient_id", "id", "mrn", "test_order")
                    and any(k in str(c).lower() for k in LAB_REF_TABLE)
                ]
                if not analyte_cols:
                    analyte_cols = [
                        c for c in l_work.columns
                        if c not in (date_col, "patient_id", "id", "mrn", "test_order")
                        and pd.to_numeric(l_work[c], errors="coerce").notna().sum() > 0
                    ]
                if analyte_cols:
                    selected_col = st.selectbox(
                        "Select Analyte to Inspect",
                        options=analyte_cols,
                        key="sb_select_lab_wide_col",
                    )
                    if selected_col:
                        ref_key = next((k for k in LAB_REF_TABLE if k in str(selected_col).lower()), None)
                        bands = []
                        if ref_key:
                            r_low, r_high = LAB_REF_TABLE[ref_key]
                            bands = [(r_low, r_high, THEME["LOW"], f"Ref: {r_low}-{r_high}")]

                        l_work[selected_col] = pd.to_numeric(l_work[selected_col], errors="coerce")
                        valid_sub = l_work.dropna(subset=[selected_col])
                        if not valid_sub.empty:
                            threshold_line_chart(
                                df=valid_sub,
                                x=date_col,
                                y=selected_col,
                                bands=bands,
                                title=f"{str(selected_col).replace('_', ' ').title()} Longitudinal Trajectory",
                                y_label="Value",
                            )

            section("Laboratory Records Table", "Complete laboratory investigations recorded.")
            st.dataframe(l_work, use_container_width=True)

    # ==========================================================================
    # TAB 4: DIAGNOSES & MEDICATIONS
    # ==========================================================================
    with t4:
        has_diag = isinstance(diag_df, pd.DataFrame) and not diag_df.empty
        has_meds = isinstance(meds_df, pd.DataFrame) and not meds_df.empty

        if not has_diag and not has_meds:
            empty_patient_state(
                "No condition or medication records found in this dataset.\n\n"
                "💡 Hint: Add columns like 'diagnosis', 'icd10', or 'medication' to see disease history and pharmacotherapy."
            )
        else:
            sub_diag, sub_med = st.tabs(["📋 Diagnoses & Conditions", "💊 Prescribed Medications"])

            with sub_diag:
                if not has_diag:
                    empty_patient_state(
                        "No diagnosis records on file.\n\n"
                        "Add columns like 'diagnosis' or 'icd10' to display clinical conditions."
                    )
                else:
                    col_dt, col_dp = st.columns([2, 1])
                    with col_dt:
                        render_html(
                            f"<div style='font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: {THEME['ACCENT']}; margin-bottom: 0.5rem;'>📋 Clinical Diagnosis Timeline</div>"
                        )
                        diagnosis_timeline(diag_df)

                    with col_dp:
                        status_col = next((c for c in diag_df.columns if "status" in str(c).lower()), None)
                        if status_col:
                            s_counts = diag_df[status_col].astype(str).str.capitalize().value_counts()
                            fig_status = go.Figure(
                                go.Pie(
                                    labels=s_counts.index,
                                    values=s_counts.values,
                                    hole=0.5,
                                    marker=dict(colors=[THEME["MED"], THEME["LOW"], THEME["HIGH"]]),
                                    textinfo="label+percent",
                                    textfont=dict(family="Plus Jakarta Sans, sans-serif", size=11),
                                )
                            )
                            layout_pie = get_plotly_layout()
                            layout_pie.update({
                                "title": {"text": "<b>Condition Resolution Status</b>", "font": {"family": "Plus Jakarta Sans, sans-serif", "size": 13, "color": THEME["TEXT"]}},
                                "margin": dict(l=20, r=20, t=40, b=20),
                                "height": 260,
                            })
                            fig_status.update_layout(**layout_pie)
                            st.plotly_chart(fig_status, use_container_width=True)

                    st.dataframe(diag_df, use_container_width=True)

            with sub_med:
                if not has_meds:
                    empty_patient_state(
                        "No medication records on file.\n\n"
                        "Add columns like 'drug_name' or 'medication' to display active pharmacotherapy."
                    )
                else:
                    meds_work = meds_df.copy()
                    name_col = next(
                        (c for c in meds_work.columns if any(k in str(c).lower() for k in ("drug", "medication", "name"))),
                        meds_work.columns[0],
                    )
                    meds_work["drug_class"] = meds_work[name_col].apply(drug_class)

                    start_col = next((c for c in meds_work.columns if any(k in str(c).lower() for k in ("start", "date", "prescribed"))), None)
                    end_col = next((c for c in meds_work.columns if any(k in str(c).lower() for k in ("end", "stop", "expiry"))), None)

                    if start_col:
                        try:
                            meds_work[start_col] = pd.to_datetime(meds_work[start_col], errors="coerce")
                            if end_col:
                                meds_work[end_col] = pd.to_datetime(meds_work[end_col], errors="coerce")
                            else:
                                meds_work["end_date"] = meds_work[start_col] + pd.Timedelta(days=30)
                                end_col = "end_date"

                            render_html(
                                f"<div style='font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: {THEME['ACCENT']}; margin-bottom: 0.5rem;'>💊 Pharmacological Timeline</div>"
                            )
                            timeline_gantt(
                                df=meds_work.dropna(subset=[start_col, end_col]),
                                start_col=start_col,
                                end_col=end_col,
                                label_col=name_col,
                                color_col="drug_class",
                            )
                        except Exception:
                            pass

                    render_html(
                        f"<div style='font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: {THEME['ACCENT']}; margin: 1rem 0 0.5rem 0;'>📋 Prescribed Medications Table</div>"
                    )
                    st.dataframe(meds_work, use_container_width=True)

    # ==========================================================================
    # TAB 5: ENCOUNTERS
    # ==========================================================================
    with t5:
        if encounters_df is None or not isinstance(encounters_df, pd.DataFrame) or encounters_df.empty:
            empty_patient_state(
                "No hospital encounter records available in this dataset.\n\n"
                "💡 Hint: Add columns like 'encounter_type', 'visit_type', 'duration_days', or 'cost' to inspect admissions and clinical visits."
            )
        else:
            enc_work = encounters_df.copy()

            cost_col = next((c for c in enc_work.columns if "cost" in str(c).lower() or "amount" in str(c).lower()), None)
            total_cost = None
            if cost_col:
                enc_work[cost_col] = pd.to_numeric(enc_work[cost_col], errors="coerce")
                if pd.api.types.is_numeric_dtype(enc_work[cost_col]):
                    total_cost = enc_work[cost_col].sum()

            los_col = next((c for c in enc_work.columns if any(k in str(c).lower() for k in ("duration", "los", "stay"))), None)
            avg_los = None
            if los_col:
                enc_work[los_col] = pd.to_numeric(enc_work[los_col], errors="coerce")
                if pd.api.types.is_numeric_dtype(enc_work[los_col]):
                    avg_los = enc_work[los_col].mean()

            type_col = next((c for c in enc_work.columns if any(k in str(c).lower() for k in ("type", "class", "dept", "admission"))), None)
            er_count = 0
            if type_col:
                er_count = enc_work[type_col].astype(str).str.lower().str.contains("emerg|er|urgent").sum()

            enc_kpis = [
                {"label": "Total Visits", "value": f"{len(enc_work):,}", "delta": "Encounters", "accent": "sky"},
                {"label": "Total Cost", "value": f"₹{total_cost:,.0f}" if total_cost is not None else "—", "delta": "Claims", "accent": "indigo"},
                {"label": "Avg Length of Stay", "value": f"{avg_los:.1f} days" if avg_los is not None else "—", "delta": "Per Encounter", "accent": "emerald"},
                {"label": "Emergency Visits", "value": str(er_count), "delta": "Urgent Care", "accent": "rose" if er_count > 0 else "sky"},
            ]
            kpi_row(enc_kpis, cols=4)

            render_html(
                f"<div style='font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: {THEME['ACCENT']}; margin: 1.5rem 0 0.5rem 0;'>📋 Encounter History</div>"
            )
            st.dataframe(enc_work, use_container_width=True)
