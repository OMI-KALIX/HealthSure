"""Patient profile UI layout and rendering components for HealthSure."""
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from core.theme import THEME
from core.ui import (
    get_plotly_layout,
    render_html,
    threshold_line_chart,
    timeline_gantt,
)


def _status_color(status: Optional[str]) -> str:
    """Map clinical status string to theme palette color."""
    if status is None:
        return THEME["ACCENT"]
    s = str(status).strip().lower()
    if "normal" in s:
        return THEME["LOW"]  # emerald
    if any(k in s for k in ("low", "high", "elevated", "stage 1", "stage 2", "overweight", "borderline", "warning")):
        return THEME["MED"]  # amber
    if any(k in s for k in ("critical", "crisis", "obese", "severe", "acute", "danger")):
        return THEME["HIGH"]  # red
    return THEME["ACCENT"]


def patient_header_card(
    patient: Union[pd.Series, Dict[str, Any]],
    bmi_val: Optional[float],
    bmi_cat: str,
    active_conditions: Union[int, str, List],
    n_meds: Union[int, str, List],
    last_visit: Optional[Any],
) -> None:
    """Render a comprehensive glassmorphic patient profile header card.

    Displays avatar initial, name, age, gender, blood group, BMI badge,
    residential city, known allergies, last visit date, active condition count,
    and active medication count.

    Parameters
    ----------
    patient : pd.Series or dict
        Baseline demographic record.
    bmi_val : float, optional
        Calculated BMI number.
    bmi_cat : str
        BMI category name.
    active_conditions : int, str, or list
        Count or list of active diagnoses.
    n_meds : int, str, or list
        Count or list of active medications.
    last_visit : Any, optional
        Timestamp or date string of latest clinical visit.
    """
    p_dict = patient.to_dict() if isinstance(patient, pd.Series) else (patient or {})

    # Extract demographic values with fallbacks
    name = str(
        p_dict.get("name")
        or p_dict.get("patient_name")
        or p_dict.get("patient_id")
        or p_dict.get("id")
        or "Patient Profile"
    ).strip()

    initial = name[0].upper() if name and name[0].isalpha() else "P"
    age = p_dict.get("age", "N/A")
    gender = p_dict.get("gender") or p_dict.get("sex") or "N/A"
    blood_group = p_dict.get("blood_group") or p_dict.get("blood_type") or "N/A"
    city = p_dict.get("city") or (f"Tier {p_dict['city_tier']}" if p_dict.get("city_tier") else "N/A")
    allergies = p_dict.get("allergies") or p_dict.get("allergy") or "None reported"

    n_cond = len(active_conditions) if isinstance(active_conditions, list) else active_conditions
    n_medications = len(n_meds) if isinstance(n_meds, list) else n_meds

    last_visit_str = "N/A"
    if last_visit is not None and pd.notna(last_visit):
        if hasattr(last_visit, "strftime"):
            last_visit_str = last_visit.strftime("%b %d, %Y")
        else:
            last_visit_str = str(last_visit)

    bmi_color = _status_color(bmi_cat)
    bmi_display = f"{bmi_val:.1f} • {bmi_cat}" if bmi_val is not None else str(bmi_cat)

    card_html = f"""
    <div class="glass-card" style="padding: 1.5rem; margin-bottom: 1.5rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.25rem;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="
                    width: 58px;
                    height: 58px;
                    border-radius: 50%;
                    background: linear-gradient(135deg, {THEME['INDIGO']} 0%, {THEME['ACCENT']} 100%);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.6rem;
                    font-weight: 700;
                    color: #ffffff;
                    box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.4);
                ">
                    {initial}
                </div>
                <div>
                    <h2 style="margin: 0; font-size: 1.5rem; font-weight: 700; color: {THEME['TEXT']}; letter-spacing: -0.02em;">
                        {name}
                    </h2>
                    <div style="color: {THEME['MUTED']}; font-size: 0.875rem; margin-top: 0.2rem;">
                        {age} yrs • {gender} • 📍 {city}
                    </div>
                </div>
            </div>

            <div style="
                background: {bmi_color}1f;
                color: {bmi_color};
                border: 1px solid {bmi_color}4d;
                padding: 0.4rem 0.9rem;
                border-radius: 9999px;
                font-size: 0.85rem;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 0.4rem;
            ">
                <span style="width: 7px; height: 7px; border-radius: 50%; background: {bmi_color};"></span>
                BMI: {bmi_display}
            </div>
        </div>

        <div style="
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 0.75rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 0.825rem;
        ">
            <div style="color: {THEME['MUTED']};">
                Blood Group: <b style="color: {THEME['TEXT']}; margin-left: 0.25rem;">🩸 {blood_group}</b>
            </div>
            <div style="color: {THEME['MUTED']};">
                Allergies: <b style="color: {THEME['TEXT']}; margin-left: 0.25rem;">⚠️ {allergies}</b>
            </div>
            <div style="color: {THEME['MUTED']};">
                Last Visit: <b style="color: {THEME['TEXT']}; margin-left: 0.25rem;">🗓️ {last_visit_str}</b>
            </div>
            <div style="color: {THEME['MUTED']};">
                Active Conditions: <b style="color: {THEME['ACCENT']}; margin-left: 0.25rem;">📋 {n_cond}</b>
            </div>
            <div style="color: {THEME['MUTED']};">
                Medications: <b style="color: {THEME['LOW']}; margin-left: 0.25rem;">💊 {n_medications}</b>
            </div>
        </div>
    </div>
    """
    render_html(card_html)


def narrative_card(text: str) -> None:
    """Render a soft glass card containing the synthesized clinical narrative.

    Parameters
    ----------
    text : str
        Clinical summary paragraph text.
    """
    if not text:
        return

    card_html = f"""
    <div class="glass-card" style="
        border-left: 4px solid {THEME['ACCENT']};
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        background: rgba(15, 23, 42, 0.75) !important;
    ">
        <div style="
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {THEME['ACCENT']};
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        ">
            <span>📑</span> Clinical Summary & Synthesis
        </div>
        <p style="
            color: {THEME['TEXT']};
            font-size: 0.95rem;
            line-height: 1.65;
            margin: 0;
            font-weight: 400;
        ">
            {text}
        </p>
    </div>
    """
    render_html(card_html)


def latest_value_card(
    label: str,
    value: Any,
    unit: str,
    status: str,
    hint: Optional[str] = None,
) -> None:
    """Render a clinical KPI card whose status dynamically drives highlight color.

    Color Mapping:
      - Normal   -> Emerald (THEME['LOW'])
      - Low/High -> Amber (THEME['MED'])
      - Critical -> Red (THEME['HIGH'])

    Parameters
    ----------
    label : str
        Clinical analyte or vital measurement title.
    value : Any
        Measurement value.
    unit : str
        Measurement unit (e.g. 'mmHg', '%', 'mg/dL').
    status : str
        Status name ('Normal', 'High', 'Low', 'Critical', etc.).
    hint : str, optional
        Target clinical range or guidance note.
    """
    accent_color = _status_color(status)

    hint_html = ""
    if hint:
        hint_html = f"""
        <div style="
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            margin-top: 0.75rem;
            padding-top: 0.5rem;
            font-size: 0.775rem;
            color: {THEME['MUTED']};
            display: flex;
            align-items: center;
            gap: 0.3rem;
        ">
            <span style="color: {accent_color};">ℹ</span> {hint}
        </div>
        """

    card_html = f"""
    <div class="glass-card" style="
        border-top: 3px solid {accent_color};
        padding: 1.15rem 1.25rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 120px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="
                font-size: 0.775rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: {THEME['MUTED']};
            ">{label}</span>
            <span style="
                background: {accent_color}22;
                color: {accent_color};
                border: 1px solid {accent_color}55;
                padding: 0.15rem 0.55rem;
                border-radius: 9999px;
                font-size: 0.75rem;
                font-weight: 600;
            ">{status}</span>
        </div>
        <div style="display: flex; align-items: baseline; gap: 0.35rem;">
            <span style="
                font-size: 1.75rem;
                font-weight: 700;
                color: {THEME['TEXT']};
                letter-spacing: -0.02em;
                line-height: 1.1;
            ">{value}</span>
            <span style="font-size: 0.85rem; color: {THEME['MUTED']}; font-weight: 500;">{unit}</span>
        </div>
        {hint_html}
    </div>
    """
    render_html(card_html)


def diagnosis_timeline(df: Optional[pd.DataFrame]) -> None:
    """Render a vertical HTML timeline of patient diagnoses and clinical conditions.

    Displays timeline date, condition description, ICD-10 code, and resolution status.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Diagnoses dataset.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        empty_patient_state("No diagnosis records available to display in timeline.")
        return

    # Identify relevant columns
    date_col = next((c for c in df.columns if any(k in str(c).lower() for k in ("date", "time"))), None)
    name_col = next((c for c in df.columns if any(k in str(c).lower() for k in ("diagnosis", "name", "desc", "condition"))), None)
    code_col = next((c for c in df.columns if "icd" in str(c).lower() or "code" in str(c).lower()), None)
    status_col = next((c for c in df.columns if "status" in str(c).lower()), None)

    # Sort if date column exists
    display_df = df.copy()
    if date_col:
        try:
            display_df = display_df.sort_values(date_col, ascending=False)
        except Exception:
            pass

    timeline_items = []

    for _, row in display_df.iterrows():
        # Date string
        date_str = "Recorded"
        if date_col and pd.notna(row.get(date_col)):
            val = row[date_col]
            date_str = val.strftime("%b %d, %Y") if hasattr(val, "strftime") else str(val)

        # Condition title
        condition = str(row.get(name_col) or row.get(code_col) or "Diagnosis").strip()

        # ICD-10 code
        icd_code = str(row.get(code_col, "")).strip()
        icd_badge = f'<span style="background: {THEME["ACCENT"]}22; color: {THEME["ACCENT"]}; border: 1px solid {THEME["ACCENT"]}44; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">ICD-10: {icd_code}</span>' if icd_code else ""

        # Status
        status = str(row.get(status_col, "Active")).strip().capitalize()
        status_color = THEME["LOW"] if status.lower() == "resolved" else (THEME["MED"] if status.lower() == "chronic" else THEME["HIGH"])
        status_badge = f'<span style="background: {status_color}22; color: {status_color}; border: 1px solid {status_color}44; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">{status}</span>'

        item_html = f"""
        <div style="position: relative; padding-left: 28px; padding-bottom: 1.25rem;">
            <div style="
                position: absolute;
                left: -6px;
                top: 4px;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: {status_color};
                box-shadow: 0 0 8px {status_color};
            "></div>

            <div style="
                background: {THEME['CARD']};
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 8px;
                padding: 0.85rem 1.15rem;
            ">
                <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.35rem;">
                    <span style="color: {THEME['MUTED']}; font-size: 0.8rem; font-weight: 500;">🗓️ {date_str}</span>
                    {icd_badge}
                    {status_badge}
                </div>
                <div style="color: {THEME['TEXT']}; font-weight: 600; font-size: 0.95rem;">
                    {condition}
                </div>
            </div>
        </div>
        """
        timeline_items.append(item_html)

    timeline_container = f"""
    <div style="
        border-left: 2px solid rgba(255, 255, 255, 0.12);
        margin-left: 10px;
        margin-top: 1rem;
        padding-top: 0.25rem;
    ">
        {''.join(timeline_items)}
    </div>
    """
    render_html(timeline_container)


def empty_patient_state(message: str) -> None:
    """Render a centered glass placeholder for absent patient records.

    Parameters
    ----------
    message : str
        Description or instruction for the empty state.
    """
    card_html = f"""
    <div class="glass-card" style="
        border: 1px dashed rgba(255, 255, 255, 0.16);
        border-radius: 12px;
        padding: 2.25rem 1.5rem;
        text-align: center;
        margin: 1rem 0;
    ">
        <div style="font-size: 2.5rem; margin-bottom: 0.6rem; line-height: 1;">🩺</div>
        <div style="color: {THEME['MUTED']}; font-size: 0.925rem; max-width: 480px; margin: 0 auto; line-height: 1.5;">
            {message}
        </div>
    </div>
    """
    render_html(card_html)
