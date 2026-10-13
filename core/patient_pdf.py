"""Comprehensive clinical PDF report generation using ReportLab for HealthSure Patient Deep Dive."""
from datetime import datetime
from typing import Any, Dict, List, Optional
import html
import io
import re
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from core.patient_clinical import (
    LAB_REF_TABLE,
    aha_bp_category,
    bmi,
    cardiovascular_risk_factors,
    charlson_like_score,
    clinical_narrative,
    drug_class,
    lab_status,
    safe_get,
)


def _esc(val: Any) -> str:
    """Safely format and HTML-escape string for ReportLab Paragraph rendering."""
    if val is None or pd.isna(val):
        return "—"
    return html.escape(str(val).strip())


class NumberedCanvas(canvas.Canvas):
    """Two-pass ReportLab canvas to compute total page count and draw running headers and footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 762, "HealthSure | Patient Clinical Deep Dive Report")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(36, 756, 576, 756)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 42, 576, 42)
        self.drawString(
            36, 30, "HealthSure InsurTech Suite v2.0 | Confidential EHR Record | Medical In-Confidence"
        )
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 30, page_str)
        self.restoreState()


def generate_patient_pdf(shaped_data: Dict[str, Any]) -> bytes:
    """Generate a comprehensive multi-page clinical deep dive PDF report.

    Parameters
    ----------
    shaped_data : dict
        Standard shaped clinical dataset containing demographics, vitals,
        labs, diagnoses, medications, encounters, and summary metadata.

    Returns
    -------
    bytes
        Raw PDF document bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=48,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=3,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=10,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1e293b"),
    )
    narrative_body = ParagraphStyle(
        "NarrativeBody",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#0f172a"),
    )
    th_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#ffffff"),
    )
    td_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#334155"),
    )

    elements: List[Any] = []

    # 1. Header Banner
    elements.append(Paragraph("HealthSure Clinical Deep Dive Report", title_style))
    now_str = datetime.now().strftime("%B %d, %Y | %H:%M UTC")
    elements.append(
        Paragraph(
            f"Comprehensive Electronic Health Record (EHR) Synthesis & Risk Analysis | Generated: {now_str}",
            subtitle_style,
        )
    )
    elements.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#0284c7"),
            spaceAfter=8,
        )
    )

    # 2. Demographics & Patient Profile Card
    demo_df = shaped_data.get("demographics")
    p_rec = (
        demo_df.iloc[0]
        if (demo_df is not None and isinstance(demo_df, pd.DataFrame) and not demo_df.empty)
        else None
    )

    p_name = safe_get(p_rec, "name", "full_name", "patient_name", default="Patient Record")
    p_id = safe_get(p_rec, "id", "patient_id", "mrn", default="—")
    p_age = safe_get(p_rec, "age", "age_years", default="—")
    p_gender = safe_get(p_rec, "gender", "sex", default="—")
    p_blood = safe_get(p_rec, "blood_group", "blood_type", "blood", default="—")
    p_city = safe_get(p_rec, "city", "location", "address", default="—")
    p_allergy = safe_get(p_rec, "allergies", "allergy", default="None reported")

    h_val = safe_get(p_rec, "height", "height_cm", "height_m")
    w_val = safe_get(p_rec, "weight", "weight_kg")
    bmi_val, bmi_cat = bmi(h_val, w_val)
    bmi_str = f"{bmi_val:.1f} kg/m<sup>2</sup> ({_esc(bmi_cat)})" if bmi_val is not None else "—"

    demo_data = [
        [
            Paragraph("<b>Patient Name:</b>", body_style),
            Paragraph(f"<b>{_esc(p_name)}</b>", body_style),
            Paragraph("<b>Patient ID / MRN:</b>", body_style),
            Paragraph(_esc(p_id), body_style),
        ],
        [
            Paragraph("<b>Age / Gender:</b>", body_style),
            Paragraph(f"{_esc(p_age)} yrs | {_esc(p_gender)}", body_style),
            Paragraph("<b>Blood Group:</b>", body_style),
            Paragraph(_esc(p_blood), body_style),
        ],
        [
            Paragraph("<b>Location:</b>", body_style),
            Paragraph(_esc(p_city), body_style),
            Paragraph("<b>Calculated BMI:</b>", body_style),
            Paragraph(bmi_str, body_style),
        ],
        [
            Paragraph("<b>Allergies:</b>", body_style),
            Paragraph(_esc(p_allergy), body_style),
            Paragraph("<b>Height / Weight:</b>", body_style),
            Paragraph(f"{_esc(h_val)} cm / {_esc(w_val)} kg", body_style),
        ],
    ]

    t_demo = Table(demo_data, colWidths=[105, 165, 105, 165])
    t_demo.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ])
    )
    elements.append(t_demo)
    elements.append(Spacer(1, 8))

    # 3. Clinical Executive Narrative
    vitals_df = shaped_data.get("vitals")
    labs_df = shaped_data.get("labs")
    diag_df = shaped_data.get("diagnoses")
    meds_df = shaped_data.get("medications")
    enc_df = shaped_data.get("encounters")

    narrative_text = clinical_narrative(
        p_rec,
        vitals=vitals_df,
        labs=labs_df,
        diagnoses=diag_df,
        medications=meds_df,
        encounters=enc_df,
    )
    elements.append(Paragraph("Clinical Executive Summary", section_heading))
    t_narrative = Table([[Paragraph(f"<b>Synthesis:</b> {_esc(narrative_text)}", narrative_body)]], colWidths=[540])
    t_narrative.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#86efac")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    elements.append(t_narrative)
    elements.append(Spacer(1, 8))

    # 4. Cardiovascular & Comorbidity Risk Stratification
    elements.append(Paragraph("Cardiovascular & Comorbidity Risk Assessment", section_heading))
    charlson_score = (
        charlson_like_score(diag_df) if (diag_df is not None and not diag_df.empty) else 0
    )
    c_status = (
        "Normal (0 pts)"
        if charlson_score == 0
        else (
            "Low Risk (1 pt)"
            if charlson_score == 1
            else ("High Risk (2 pts)" if charlson_score == 2 else f"Critical ({charlson_score} pts)")
        )
    )
    cv_factors = cardiovascular_risk_factors(
        p_rec,
        diag_df if (diag_df is not None and not diag_df.empty) else pd.DataFrame(),
        labs_df,
    )
    cv_str = ", ".join(cv_factors) if cv_factors else "No major cardiovascular risk factors detected."

    risk_data = [
        [
            Paragraph("<b>Charlson Comorbidity Score:</b>", body_style),
            Paragraph(f"<b>{charlson_score} pts</b> | {_esc(c_status)}", body_style),
        ],
        [
            Paragraph("<b>Identified CV Risk Factors:</b>", body_style),
            Paragraph(_esc(cv_str), body_style),
        ],
    ]
    t_risk = Table(risk_data, colWidths=[150, 390])
    t_risk.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#fca5a5")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#fecaca")),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ])
    )
    elements.append(t_risk)
    elements.append(Spacer(1, 8))

    # 5. Vital Signs Summary & History Table
    elements.append(Paragraph("Longitudinal Vital Signs Summary", section_heading))
    if vitals_df is not None and isinstance(vitals_df, pd.DataFrame) and not vitals_df.empty:
        latest_v = vitals_df.iloc[-1]
        sys_col = next((c for c in vitals_df.columns if "systolic" in str(c).lower() or c == "sbp"), None)
        dia_col = next((c for c in vitals_df.columns if "diastolic" in str(c).lower() or c == "dbp"), None)
        hr_col = next((c for c in vitals_df.columns if any(k in str(c).lower() for k in ("heart", "pulse", "hr"))), None)
        spo2_col = next((c for c in vitals_df.columns if any(k in str(c).lower() for k in ("spo2", "oxygen", "o2"))), None)
        date_col = next((c for c in vitals_df.columns if any(k in str(c).lower() for k in ("date", "time"))), None)

        bp_str = (
            f"{int(float(latest_v[sys_col]))}/{int(float(latest_v[dia_col]))} mmHg ({aha_bp_category(latest_v[sys_col], latest_v[dia_col])})"
            if (sys_col and dia_col and pd.notna(latest_v.get(sys_col)) and pd.notna(latest_v.get(dia_col)))
            else "—"
        )
        hr_str = f"{int(float(latest_v[hr_col]))} bpm" if (hr_col and pd.notna(latest_v.get(hr_col))) else "—"
        spo2_str = f"{float(latest_v[spo2_col]):.1f}%" if (spo2_col and pd.notna(latest_v.get(spo2_col))) else "—"

        v_summary = [
            [
                Paragraph("<b>Latest Blood Pressure:</b>", body_style),
                Paragraph(_esc(bp_str), body_style),
                Paragraph("<b>Latest Heart Rate:</b>", body_style),
                Paragraph(_esc(hr_str), body_style),
            ],
            [
                Paragraph("<b>Oxygen Saturation (SpO2):</b>", body_style),
                Paragraph(_esc(spo2_str), body_style),
                Paragraph("<b>Total Vitals Recorded:</b>", body_style),
                Paragraph(f"{len(vitals_df)} reading(s)", body_style),
            ],
        ]
        t_vsum = Table(v_summary, colWidths=[140, 130, 140, 130])
        t_vsum.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        elements.append(t_vsum)

        if len(vitals_df) > 1:
            elements.append(Spacer(1, 4))
            v_rows = [[
                Paragraph("<b>Timestamp / Date</b>", th_style),
                Paragraph("<b>Systolic (mmHg)</b>", th_style),
                Paragraph("<b>Diastolic (mmHg)</b>", th_style),
                Paragraph("<b>Heart Rate (bpm)</b>", th_style),
                Paragraph("<b>SpO2 (%)</b>", th_style),
            ]]
            for _, r in vitals_df.tail(6).iterrows():
                d_str = _esc(r.get(date_col, "—")) if date_col else "—"
                s_str = str(int(float(r[sys_col]))) if (sys_col and pd.notna(r.get(sys_col))) else "—"
                dia_str = str(int(float(r[dia_col]))) if (dia_col and pd.notna(r.get(dia_col))) else "—"
                h_str = str(int(float(r[hr_col]))) if (hr_col and pd.notna(r.get(hr_col))) else "—"
                sp_str = f"{float(r[spo2_col]):.1f}" if (spo2_col and pd.notna(r.get(spo2_col))) else "—"
                v_rows.append([
                    Paragraph(d_str, td_style),
                    Paragraph(s_str, td_style),
                    Paragraph(dia_str, td_style),
                    Paragraph(h_str, td_style),
                    Paragraph(sp_str, td_style),
                ])
            t_vtable = Table(v_rows, colWidths=[140, 100, 100, 100, 100])
            t_vtable.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ])
            )
            elements.append(t_vtable)
    else:
        elements.append(Paragraph("<i>No vital signs recorded on file.</i>", body_style))
    elements.append(Spacer(1, 8))

    # 6. Laboratory Investigation Panel
    elements.append(Paragraph("Laboratory Investigations & Analyte Panel", section_heading))
    if labs_df is not None and isinstance(labs_df, pd.DataFrame) and not labs_df.empty:
        test_col = next((c for c in labs_df.columns if any(k in str(c).lower() for k in ("test", "analyte", "lab", "name"))), None)
        val_col = next((c for c in labs_df.columns if any(k in str(c).lower() for k in ("value", "result", "val"))), None)
        unit_col = next((c for c in labs_df.columns if "unit" in str(c).lower()), None)

        l_rows = [[
            Paragraph("<b>Analyte / Investigation</b>", th_style),
            Paragraph("<b>Result</b>", th_style),
            Paragraph("<b>Unit</b>", th_style),
            Paragraph("<b>Reference Interval</b>", th_style),
            Paragraph("<b>Status</b>", th_style),
        ]]

        if test_col and val_col:
            for t_name in labs_df[test_col].dropna().unique()[:8]:
                t_sub = labs_df[labs_df[test_col] == t_name]
                l_rec = t_sub.iloc[-1]
                v_res = str(l_rec[val_col])
                u_str = str(l_rec.get(unit_col, "")) if unit_col and pd.notna(l_rec.get(unit_col)) else "—"
                stat = lab_status(t_name, l_rec[val_col])
                ref_k = next((k for k in LAB_REF_TABLE if k in str(t_name).lower()), None)
                ref_range = f"{LAB_REF_TABLE[ref_k][0]} - {LAB_REF_TABLE[ref_k][1]}" if ref_k else "Standard"
                l_rows.append([
                    Paragraph(_esc(str(t_name).replace("_", " ").title()), td_style),
                    Paragraph(_esc(v_res), td_style),
                    Paragraph(_esc(u_str), td_style),
                    Paragraph(_esc(ref_range), td_style),
                    Paragraph(_esc(stat), td_style),
                ])
        else:
            for k, (low, high) in LAB_REF_TABLE.items():
                m_col = next((c for c in labs_df.columns if k in str(c).lower()), None)
                if m_col and pd.notna(labs_df[m_col].iloc[-1]):
                    val_c = labs_df[m_col].iloc[-1]
                    stat = lab_status(k, val_c)
                    l_rows.append([
                        Paragraph(_esc(str(m_col).replace("_", " ").title()), td_style),
                        Paragraph(_esc(val_c), td_style),
                        Paragraph("—", td_style),
                        Paragraph(f"{low} - {high}", td_style),
                        Paragraph(_esc(stat), td_style),
                    ])

        if len(l_rows) > 1:
            t_ltable = Table(l_rows, colWidths=[150, 80, 70, 130, 110])
            t_ltable.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ])
            )
            elements.append(t_ltable)
        else:
            elements.append(Paragraph("<i>No standard laboratory analytes mapped.</i>", body_style))
    else:
        elements.append(Paragraph("<i>No laboratory records on file.</i>", body_style))
    elements.append(Spacer(1, 8))

    # 7. Diagnoses & Clinical Conditions
    elements.append(Paragraph("Diagnoses & Clinical Conditions", section_heading))
    if diag_df is not None and isinstance(diag_df, pd.DataFrame) and not diag_df.empty:
        d_name_col = next(
            (c for c in diag_df.columns if any(k in str(c).lower() for k in ("diagnosis", "name", "desc", "condition", "disease"))),
            diag_df.columns[0],
        )
        icd_col = next((c for c in diag_df.columns if "icd" in str(c).lower() or "code" in str(c).lower()), None)
        stat_col = next((c for c in diag_df.columns if "status" in str(c).lower()), None)
        date_col = next((c for c in diag_df.columns if any(k in str(c).lower() for k in ("date", "time"))), None)

        d_rows = [[
            Paragraph("<b>Condition / Diagnosis Description</b>", th_style),
            Paragraph("<b>ICD-10 Code</b>", th_style),
            Paragraph("<b>Recorded Date</b>", th_style),
            Paragraph("<b>Resolution Status</b>", th_style),
        ]]
        for _, r in diag_df.head(8).iterrows():
            cond_desc = str(r.get(d_name_col, "Clinical Condition"))
            icd_val = str(r.get(icd_col, "—")) if icd_col and pd.notna(r.get(icd_col)) else "—"
            rec_dt = str(r.get(date_col, "—")) if date_col and pd.notna(r.get(date_col)) else "—"
            c_stat = str(r.get(stat_col, "Active")).capitalize() if stat_col and pd.notna(r.get(stat_col)) else "Active"
            d_rows.append([
                Paragraph(_esc(cond_desc), td_style),
                Paragraph(_esc(icd_val), td_style),
                Paragraph(_esc(rec_dt), td_style),
                Paragraph(_esc(c_stat), td_style),
            ])
        t_dtable = Table(d_rows, colWidths=[230, 90, 110, 110])
        t_dtable.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4338ca")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ])
        )
        elements.append(t_dtable)
    else:
        elements.append(Paragraph("<i>No diagnostic conditions on file.</i>", body_style))
    elements.append(Spacer(1, 8))

    # 8. Medications & Pharmacotherapy
    elements.append(Paragraph("Prescribed Medications & Pharmacotherapy", section_heading))
    if meds_df is not None and isinstance(meds_df, pd.DataFrame) and not meds_df.empty:
        m_name_col = next((c for c in meds_df.columns if any(k in str(c).lower() for k in ("drug", "medication", "name"))), meds_df.columns[0])
        start_col = next((c for c in meds_df.columns if any(k in str(c).lower() for k in ("start", "date", "prescribed"))), None)
        end_col = next((c for c in meds_df.columns if any(k in str(c).lower() for k in ("end", "stop", "expiry"))), None)

        m_rows = [[
            Paragraph("<b>Medication Name</b>", th_style),
            Paragraph("<b>Drug Class</b>", th_style),
            Paragraph("<b>Prescription Date</b>", th_style),
            Paragraph("<b>Status / End Date</b>", th_style),
        ]]
        for _, r in meds_df.head(8).iterrows():
            m_name = str(r.get(m_name_col, "Medication"))
            d_cls = drug_class(m_name)
            s_dt = str(r.get(start_col, "—")) if start_col and pd.notna(r.get(start_col)) else "—"
            e_dt = str(r.get(end_col, "Active")) if end_col and pd.notna(r.get(end_col)) else "Active"
            m_rows.append([
                Paragraph(_esc(m_name), td_style),
                Paragraph(_esc(d_cls), td_style),
                Paragraph(_esc(s_dt), td_style),
                Paragraph(_esc(e_dt), td_style),
            ])
        t_mtable = Table(m_rows, colWidths=[180, 130, 110, 120])
        t_mtable.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0369a1")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ])
        )
        elements.append(t_mtable)
    else:
        elements.append(Paragraph("<i>No active prescribed medications on file.</i>", body_style))
    elements.append(Spacer(1, 8))

    # 9. Hospital Encounters & Health Utilization
    elements.append(Paragraph("Hospital Encounters & Healthcare Utilization", section_heading))
    if enc_df is not None and isinstance(enc_df, pd.DataFrame) and not enc_df.empty:
        cost_col = next((c for c in enc_df.columns if "cost" in str(c).lower() or "amount" in str(c).lower()), None)
        total_cost = enc_df[cost_col].sum() if (cost_col and pd.api.types.is_numeric_dtype(enc_df[cost_col])) else None
        cost_str = f"INR {total_cost:,.0f}" if total_cost is not None else "—"

        type_col = next((c for c in enc_df.columns if any(k in str(c).lower() for k in ("type", "class", "dept", "admission"))), None)
        er_count = enc_df[type_col].astype(str).str.lower().str.contains("emerg|er|urgent").sum() if type_col else 0

        los_col = next((c for c in enc_df.columns if any(k in str(c).lower() for k in ("duration", "los", "stay", "days"))), None)
        avg_los = enc_df[los_col].mean() if (los_col and pd.api.types.is_numeric_dtype(enc_df[los_col])) else None
        los_str = f"{avg_los:.1f} days" if avg_los is not None else "—"

        enc_sum = [
            [
                Paragraph("<b>Total Recorded Visits:</b>", body_style),
                Paragraph(f"{len(enc_df)} encounter(s)", body_style),
                Paragraph("<b>Total Incurred Cost:</b>", body_style),
                Paragraph(_esc(cost_str), body_style),
            ],
            [
                Paragraph("<b>Emergency / Urgent Visits:</b>", body_style),
                Paragraph(f"{er_count} visit(s)", body_style),
                Paragraph("<b>Avg Length of Stay:</b>", body_style),
                Paragraph(_esc(los_str), body_style),
            ],
        ]
        t_esum = Table(enc_sum, colWidths=[140, 130, 140, 130])
        t_esum.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        elements.append(t_esum)

        # Recent encounters table if records exist
        date_enc_col = next((c for c in enc_df.columns if any(k in str(c).lower() for k in ("date", "time", "admit", "admission"))), None)
        if len(enc_df) > 0 and (date_enc_col or type_col or cost_col):
            elements.append(Spacer(1, 4))
            e_rows = [[
                Paragraph("<b>Encounter Date</b>", th_style),
                Paragraph("<b>Visit Type / Department</b>", th_style),
                Paragraph("<b>Length of Stay</b>", th_style),
                Paragraph("<b>Cost</b>", th_style),
            ]]
            for _, r in enc_df.head(6).iterrows():
                e_date = str(r.get(date_enc_col, "—")) if date_enc_col and pd.notna(r.get(date_enc_col)) else "—"
                e_type = str(r.get(type_col, "Outpatient")).title() if type_col and pd.notna(r.get(type_col)) else "Outpatient"
                e_los = f"{r.get(los_col):.0f} days" if (los_col and pd.notna(r.get(los_col))) else "—"
                e_cost = f"INR {float(r.get(cost_col)):,.0f}" if (cost_col and pd.notna(r.get(cost_col))) else "—"
                e_rows.append([
                    Paragraph(_esc(e_date), td_style),
                    Paragraph(_esc(e_type), td_style),
                    Paragraph(_esc(e_los), td_style),
                    Paragraph(_esc(e_cost), td_style),
                ])
            t_etable = Table(e_rows, colWidths=[140, 180, 110, 110])
            t_etable.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ])
            )
            elements.append(t_etable)
    else:
        elements.append(Paragraph("<i>No hospital encounter records on file.</i>", body_style))

    # 10. Compliance & Audit Notice
    elements.append(Spacer(1, 10))
    notice_text = (
        "<b>Notice:</b> This clinical summary is generated automatically by HealthSure InsurTech Intelligence Engine. "
        "It aggregates electronic health record dimensions for clinical review, underwriting analysis, and risk stratification. "
        "This document contains protected health information (PHI) and must be handled in compliance with HIPAA and relevant data protection regulations."
    )
    t_notice = Table([[Paragraph(notice_text, td_style)]], colWidths=[540])
    t_notice.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    elements.append(t_notice)

    # Build PDF with two-pass NumberedCanvas
    doc.build(elements, canvasmaker=NumberedCanvas)
    return buf.getvalue()
