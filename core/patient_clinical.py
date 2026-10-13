"""Clinical metrics, lab categorization, and cardiovascular risk rules for HealthSure."""
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import re
import pandas as pd
import streamlit as st

# Reference laboratory ranges (lower_bound, upper_bound)
LAB_REF_TABLE: Dict[str, Tuple[float, float]] = {
    "hba1c": (4.0, 5.6),
    "glucose": (70.0, 99.0),
    "creatinine": (0.6, 1.3),
    "cholesterol": (0.0, 199.0),
    "ldl": (0.0, 99.0),
    "hdl": (40.0, 60.0),
    "hemoglobin": (12.0, 17.0),
}

# Charlson-like comorbidity weight mapping
ICD10_WEIGHTS: Dict[str, int] = {
    "N18": 2,  # Chronic Kidney Disease
    "E11": 1,  # Type 2 Diabetes
    "I25": 1,  # Chronic Ischemic Heart Disease
    "I10": 0,  # Essential Hypertension
    "J45": 1,  # Asthma
    "J18": 1,  # Pneumonia
    "C": 2,    # Malignant Neoplasms (Cancer C*)
    "I50": 2,  # Heart Failure
}


def bmi(
    height_cm: Optional[Union[float, int]],
    weight_kg: Optional[Union[float, int]],
) -> Tuple[Optional[float], str]:
    """Calculate Body Mass Index and clinical classification category.

    Categories:
      - < 18.5       : Underweight
      - 18.5 - 24.9  : Normal
      - 25.0 - 29.9  : Overweight
      - >= 30.0      : Obese

    Parameters
    ----------
    height_cm : float or int, optional
        Height in centimeters.
    weight_kg : float or int, optional
        Weight in kilograms.

    Returns
    -------
    tuple of (float or None, str)
        Computed BMI rounded to 1 decimal place and classification string.
    """
    if height_cm is None or weight_kg is None:
        return (None, "Unknown")

    try:
        h = float(height_cm)
        w = float(weight_kg)
        if h <= 0 or w <= 0:
            return (None, "Unknown")

        # Convert cm to meters
        h_m = h / 100.0 if h > 3.0 else h
        bmi_val = round(w / (h_m ** 2), 1)

        if bmi_val < 18.5:
            cat = "Underweight"
        elif bmi_val < 25.0:
            cat = "Normal"
        elif bmi_val < 30.0:
            cat = "Overweight"
        else:
            cat = "Obese"

        return (bmi_val, cat)
    except Exception:
        return (None, "Unknown")


def aha_bp_category(
    systolic: Optional[Union[float, int]],
    diastolic: Optional[Union[float, int]],
) -> str:
    """Classify blood pressure reading according to American Heart Association guidelines.

    Categories:
      - Normal   : SBP < 120 and DBP < 80
      - Elevated : 120 <= SBP <= 129 and DBP < 80
      - Stage 1  : 130 <= SBP <= 139 or 80 <= DBP <= 89
      - Stage 2  : SBP >= 140 or DBP >= 90
      - Crisis   : SBP > 180 or DBP > 120
      - Unknown  : Missing or invalid input

    Parameters
    ----------
    systolic : float or int, optional
        Systolic Blood Pressure (mmHg).
    diastolic : float or int, optional
        Diastolic Blood Pressure (mmHg).

    Returns
    -------
    str
        AHA blood pressure tier.
    """
    if systolic is None or diastolic is None:
        return "Unknown"

    try:
        sys = float(systolic)
        dia = float(diastolic)
        if sys <= 0 or dia <= 0 or math.isnan(sys) or math.isnan(dia):
            return "Unknown"

        # Hypertensive Crisis takes highest priority
        if sys > 180 or dia > 120:
            return "Crisis"
        if sys >= 140 or dia >= 90:
            return "Stage 2"
        if (130 <= sys <= 139) or (80 <= dia <= 89):
            return "Stage 1"
        if (120 <= sys <= 129) and dia < 80:
            return "Elevated"
        if sys < 120 and dia < 80:
            return "Normal"

        return "Unknown"
    except Exception:
        return "Unknown"


def lab_status(test_name: Optional[str], value: Optional[Any]) -> str:
    """Determine whether a quantitative laboratory test is Normal, Low, or High.

    Parameters
    ----------
    test_name : str, optional
        Laboratory analyte name (e.g. 'hba1c', 'glucose', 'creatinine', 'cholesterol').
    value : Any, optional
        Quantitative test result.

    Returns
    -------
    str
        'Normal', 'Low', 'High', or 'Unknown'.
    """
    if test_name is None or value is None:
        return "Unknown"

    try:
        val = float(value)
        if math.isnan(val):
            return "Unknown"
    except Exception:
        return "Unknown"

    name_clean = re.sub(r"[^\w]", "", str(test_name).lower())

    # Match test against reference ranges
    for ref_key, (low, high) in LAB_REF_TABLE.items():
        if ref_key in name_clean:
            if val < low:
                return "Low"
            if val > high:
                return "High"
            return "Normal"

    return "Unknown"


def charlson_like_score(diagnoses_df: Optional[pd.DataFrame]) -> int:
    """Calculate Charlson-like Comorbidity Index based on active ICD-10 diagnosis codes.

    Weights:
      - N18 (CKD): 2
      - E11 (T2D): 1
      - I25 (CAD): 1
      - I10 (HTN): 0
      - J45 (Asthma): 1
      - J18 (Pneumonia): 1
      - C*  (Malignancy): 2
      - I50 (Heart Failure): 2

    Sums weights where diagnosis status is not 'resolved'.

    Parameters
    ----------
    diagnoses_df : pd.DataFrame, optional
        Dataset containing ICD-10 codes and optional status column.

    Returns
    -------
    int
        Cumulative comorbidity score.
    """
    if diagnoses_df is None or not isinstance(diagnoses_df, pd.DataFrame) or diagnoses_df.empty:
        return 0

    score = 0
    df = diagnoses_df.copy()

    # Find diagnosis code and status columns
    code_col = None
    for c in df.columns:
        if any(k in str(c).lower() for k in ("icd10", "icd_10", "code", "diag")):
            code_col = c
            break

    if code_col is None:
        return 0

    status_col = None
    for c in df.columns:
        if "status" in str(c).lower():
            status_col = c
            break

    scored_prefixes = set()

    for _, row in df.iterrows():
        # Skip resolved conditions
        if status_col and str(row.get(status_col, "")).strip().lower() == "resolved":
            continue

        raw_code = str(row.get(code_col, "")).strip().upper()
        clean_code = re.sub(r"[^\w]", "", raw_code)

        for prefix, weight in ICD10_WEIGHTS.items():
            if clean_code.startswith(prefix):
                # Avoid duplicate weighting for identical condition category
                if prefix not in scored_prefixes:
                    score += weight
                    scored_prefixes.add(prefix)
                break

    return score


def cardiovascular_risk_factors(
    patient: Optional[Union[pd.Series, Dict[str, Any]]],
    diagnoses: Optional[pd.DataFrame] = None,
    labs: Optional[pd.DataFrame] = None,
) -> List[str]:
    """Identify cardiovascular risk factors based on demographics, diagnoses, and lab results.

    Evaluated among:
      - 'Age >55'
      - 'Diabetes'
      - 'Hypertension'
      - 'High cholesterol'
      - 'Smoker'
      - 'Obesity (BMI>=30)'

    Parameters
    ----------
    patient : pd.Series or dict, optional
        Demographic and clinical baseline record.
    diagnoses : pd.DataFrame, optional
        Patient diagnosis history.
    labs : pd.DataFrame, optional
        Patient laboratory test history.

    Returns
    -------
    list of str
        List of identified risk factor labels.
    """
    factors: List[str] = []
    p_dict = patient.to_dict() if isinstance(patient, pd.Series) else (patient or {})

    # 1. Age > 55
    age = p_dict.get("age")
    try:
        if age is not None and float(age) > 55:
            factors.append("Age >55")
    except Exception:
        pass

    # Extract diagnosis codes and descriptions
    diag_text = ""
    if diagnoses is not None and isinstance(diagnoses, pd.DataFrame) and not diagnoses.empty:
        diag_text = " ".join(diagnoses.astype(str).values.flatten()).lower()

    # Extract lab values
    lab_text = ""
    if labs is not None and isinstance(labs, pd.DataFrame) and not labs.empty:
        lab_text = " ".join(labs.astype(str).values.flatten()).lower()

    # 2. Diabetes
    has_diabetes = False
    if any(k in diag_text for k in ("e11", "e10", "diabetes", "diabetic")):
        has_diabetes = True
    elif labs is not None and isinstance(labs, pd.DataFrame) and not labs.empty:
        for _, row in labs.iterrows():
            t_name = str(row.get("test_name", "")).lower()
            val = row.get("value") or row.get("lab_value")
            try:
                v = float(val)
                if ("hba1c" in t_name and v >= 6.5) or ("glucose" in t_name and v >= 126):
                    has_diabetes = True
                    break
            except Exception:
                pass
    if has_diabetes:
        factors.append("Diabetes")

    # 3. Hypertension
    has_htn = False
    if any(k in diag_text for k in ("i10", "hypertension", "htn")):
        has_htn = True
    else:
        sys = p_dict.get("systolic")
        dia = p_dict.get("diastolic")
        try:
            if sys is not None and float(sys) >= 140:
                has_htn = True
            elif dia is not None and float(dia) >= 90:
                has_htn = True
        except Exception:
            pass
    if has_htn:
        factors.append("Hypertension")

    # 4. High Cholesterol
    has_chol = False
    if any(k in diag_text for k in ("e78", "cholesterol", "hyperlipidemia")):
        has_chol = True
    elif labs is not None and isinstance(labs, pd.DataFrame) and not labs.empty:
        for _, row in labs.iterrows():
            t_name = str(row.get("test_name", "")).lower()
            val = row.get("value") or row.get("lab_value")
            try:
                v = float(val)
                if ("cholesterol" in t_name and v >= 200) or ("ldl" in t_name and v >= 130):
                    has_chol = True
                    break
            except Exception:
                pass
    if has_chol:
        factors.append("High cholesterol")

    # 5. Smoker
    smoker = p_dict.get("smoker")
    if smoker in (1, "1", "yes", "true", True) or str(smoker).strip().lower() in ("yes", "smoker", "1"):
        factors.append("Smoker")

    # 6. Obesity (BMI >= 30)
    bmi_val = p_dict.get("bmi")
    is_obese = False
    try:
        if bmi_val is not None and float(bmi_val) >= 30.0:
            is_obese = True
        elif p_dict.get("weight") and p_dict.get("height"):
            w = float(p_dict["weight"])
            h = float(p_dict["height"])
            h_m = h / 100.0 if h > 3.0 else h
            if (w / (h_m ** 2)) >= 30.0:
                is_obese = True
    except Exception:
        pass

    if is_obese:
        factors.append("Obesity (BMI>=30)")

    return factors


def drug_class(drug_name: Optional[str]) -> str:
    """Classify pharmacological agents into primary therapeutic classes.

    Keyword mapping:
      - metformin | glimepiride | insulin              -> Antidiabetic
      - amlodipine | losartan | telmisartan | ramipril | atenolol -> Antihypertensive
      - atorvastatin | rosuvastatin | simvastatin      -> Statin
      - aspirin | clopidogrel                          -> Antiplatelet
      - salbutamol | budesonide | montelukast          -> Respiratory
      - other                                          -> Other

    Parameters
    ----------
    drug_name : str, optional
        Name of pharmaceutical drug.

    Returns
    -------
    str
        Pharmacological drug class.
    """
    if drug_name is None:
        return "Other"

    name_lower = str(drug_name).strip().lower()

    if any(k in name_lower for k in ("metformin", "glimepiride", "insulin", "glipizide")):
        return "Antidiabetic"
    if any(k in name_lower for k in ("amlodipine", "losartan", "telmisartan", "ramipril", "atenolol", "metoprolol")):
        return "Antihypertensive"
    if any(k in name_lower for k in ("atorvastatin", "rosuvastatin", "simvastatin")):
        return "Statin"
    if any(k in name_lower for k in ("aspirin", "clopidogrel")):
        return "Antiplatelet"
    if any(k in name_lower for k in ("salbutamol", "budesonide", "montelukast", "albuterol")):
        return "Respiratory"

    return "Other"


@st.cache_data
def clinical_narrative(
    patient: Optional[Union[pd.Series, Dict[str, Any]]],
    vitals: Optional[pd.DataFrame] = None,
    labs: Optional[pd.DataFrame] = None,
    diagnoses: Optional[pd.DataFrame] = None,
    medications: Optional[pd.DataFrame] = None,
    encounters: Optional[pd.DataFrame] = None,
) -> str:
    """Generate a 3-5 sentence rule-based clinical summary paragraph.

    Skips clauses whose data is absent. Purely descriptive with no diagnostic claims.

    Parameters
    ----------
    patient : pd.Series or dict, optional
        Baseline patient demographic record.
    vitals : pd.DataFrame, optional
        Vitals time series records.
    labs : pd.DataFrame, optional
        Laboratory test history.
    diagnoses : pd.DataFrame, optional
        Clinical diagnosis history.
    medications : pd.DataFrame, optional
        Active medications list.
    encounters : pd.DataFrame, optional
        Hospital and clinic visits records.

    Returns
    -------
    str
        Synthesized clinical narrative paragraph.
    """
    p_dict = patient.to_dict() if isinstance(patient, pd.Series) else (patient or {})
    sentences: List[str] = []

    # 1. Demographics & Active Diagnoses
    age = p_dict.get("age")
    gender = p_dict.get("gender", "patient")

    active_diag_names: List[str] = []
    if diagnoses is not None and isinstance(diagnoses, pd.DataFrame) and not diagnoses.empty:
        status_col = next((c for c in diagnoses.columns if "status" in str(c).lower()), None)
        df_active = diagnoses[diagnoses[status_col].str.lower() != "resolved"] if status_col else diagnoses

        name_col = next(
            (c for c in diagnoses.columns if any(k in str(c).lower() for k in ("diagnosis", "name", "desc", "condition"))),
            next((c for c in diagnoses.columns if "icd" in str(c).lower()), None)
        )
        if name_col:
            active_diag_names = [str(x) for x in df_active[name_col].dropna().unique() if str(x).strip()]

    if age is not None:
        if active_diag_names:
            diag_str = ", ".join(active_diag_names[:3])
            sentences.append(
                f"{age}-year-old {gender} with {len(active_diag_names)} active condition(s): {diag_str}."
            )
        else:
            sentences.append(f"{age}-year-old {gender}.")
    elif active_diag_names:
        diag_str = ", ".join(active_diag_names[:3])
        sentences.append(f"Patient presenting with {len(active_diag_names)} active condition(s): {diag_str}.")

    # 2. Latest Blood Pressure & Vitals
    if vitals is not None and isinstance(vitals, pd.DataFrame) and not vitals.empty:
        sys_col = next((c for c in vitals.columns if "systolic" in str(c).lower() or c == "sbp"), None)
        dia_col = next((c for c in vitals.columns if "diastolic" in str(c).lower() or c == "dbp"), None)

        if sys_col and dia_col:
            latest_vital = vitals.iloc[-1]
            sys = latest_vital.get(sys_col)
            dia = latest_vital.get(dia_col)
            try:
                if sys is not None and dia is not None and not (math.isnan(float(sys)) or math.isnan(float(dia))):
                    aha_tier = aha_bp_category(sys, dia)
                    sentences.append(f"Latest blood pressure recorded at {int(float(sys))}/{int(float(dia))} mmHg ({aha_tier}).")
            except Exception:
                pass

    # 3. Latest Lab Result
    if labs is not None and isinstance(labs, pd.DataFrame) and not labs.empty:
        test_col = next((c for c in labs.columns if any(k in str(c).lower() for k in ("test", "analyte", "lab", "name"))), None)
        val_col = next((c for c in labs.columns if any(k in str(c).lower() for k in ("value", "result", "val"))), None)

        if test_col and val_col:
            latest_lab = labs.iloc[-1]
            t_name = str(latest_lab.get(test_col, "")).strip()
            val = latest_lab.get(val_col)
            if t_name and val is not None:
                status = lab_status(t_name, val)
                unit_col = next((c for c in labs.columns if "unit" in str(c).lower()), None)
                unit_str = f" {latest_lab[unit_col]}" if unit_col and pd.notna(latest_lab.get(unit_col)) else ""
                sentences.append(f"Latest {t_name} evaluated at {val}{unit_str} ({status}).")

    # 4. Medications
    if medications is not None and isinstance(medications, pd.DataFrame) and not medications.empty:
        med_count = len(medications)
        sentences.append(f"Currently prescribed {med_count} active medication(s).")

    # 5. Encounters / Emergency Visits
    if encounters is not None and isinstance(encounters, pd.DataFrame) and not encounters.empty:
        type_col = next((c for c in encounters.columns if any(k in str(c).lower() for k in ("type", "class", "admission", "department"))), None)
        if type_col:
            er_count = encounters[type_col].astype(str).str.lower().str.contains("emerg|er|urgent").sum()
            if er_count > 0:
                sentences.append(f"{er_count} emergency visit(s) on record.")
            else:
                sentences.append(f"{len(encounters)} clinical encounter(s) on record.")
        else:
            sentences.append(f"{len(encounters)} encounter(s) on record.")

    if not sentences:
        return "No clinical records available for narrative synthesis."

    return " ".join(sentences)


def safe_get(series_or_row: Any, *keys: str, default: Any = None) -> Any:
    """Try multiple column/key names and return the first present non-null value."""
    if series_or_row is None:
        return default

    # Direct key lookup
    for k in keys:
        if isinstance(series_or_row, dict):
            if k in series_or_row and series_or_row[k] is not None and pd.notna(series_or_row[k]):
                return series_or_row[k]
        elif isinstance(series_or_row, pd.Series):
            if k in series_or_row.index and series_or_row[k] is not None and pd.notna(series_or_row[k]):
                return series_or_row[k]

    # Fuzzy/normalized column lookup
    items = series_or_row.items() if hasattr(series_or_row, "items") else []
    for k in keys:
        clean_k = re.sub(r"[^a-zA-Z0-9]", "", str(k)).lower()
        for col_name, val in items:
            clean_col = re.sub(r"[^a-zA-Z0-9]", "", str(col_name)).lower()
            if clean_k == clean_col and val is not None and pd.notna(val):
                return val

    # Substring lookup
    for k in keys:
        clean_k = re.sub(r"[^a-zA-Z0-9]", "", str(k)).lower()
        if len(clean_k) >= 3:
            for col_name, val in items:
                clean_col = re.sub(r"[^a-zA-Z0-9]", "", str(col_name)).lower()
                if (clean_col.startswith(clean_k) or clean_col.endswith(clean_k)) and val is not None and pd.notna(val):
                    return val

    return default


def describe_dataset(shape_dict: Dict[str, Any]) -> str:
    """Return a one-line summary of the loaded clinical dataset."""
    if not isinstance(shape_dict, dict):
        return "No dataset loaded."

    parts: List[str] = []
    demo = shape_dict.get("demographics")
    if demo is not None and isinstance(demo, pd.DataFrame) and not demo.empty:
        n_pts = len(demo)
        parts.append(f"{n_pts} patient{'s' if n_pts != 1 else ''}")

    vitals = shape_dict.get("vitals")
    if vitals is not None and isinstance(vitals, pd.DataFrame) and not vitals.empty:
        n_v = len(vitals)
        parts.append(f"{n_v} vitals row{'s' if n_v != 1 else ''}")

    labs = shape_dict.get("labs")
    if labs is not None and isinstance(labs, pd.DataFrame) and not labs.empty:
        n_l = len(labs)
        parts.append(f"{n_l} lab test{'s' if n_l != 1 else ''}")

    diags = shape_dict.get("diagnoses")
    if diags is not None and isinstance(diags, pd.DataFrame) and not diags.empty:
        n_d = len(diags)
        parts.append(f"{n_d} active condition{'s' if n_d != 1 else ''}")

    meds = shape_dict.get("medications")
    if meds is not None and isinstance(meds, pd.DataFrame) and not meds.empty:
        n_m = len(meds)
        parts.append(f"{n_m} medication{'s' if n_m != 1 else ''}")

    encs = shape_dict.get("encounters")
    if encs is not None and isinstance(encs, pd.DataFrame) and not encs.empty:
        n_e = len(encs)
        parts.append(f"{n_e} encounter{'s' if n_e != 1 else ''}")

    if not parts:
        return "Loaded 0 clinical records."

    if len(parts) == 1:
        return f"Loaded {parts[0]}."
    return f"Loaded {', '.join(parts[:-1])}, and {parts[-1]}."
