"""Universal clinical data ingestion, column semantic classification, and dataset shaping for HealthSure."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import io
import re
import warnings
import pandas as pd
import streamlit as st

ALL_SEMANTIC_BUCKETS: List[str] = [
    "id", "name", "age", "gender", "blood", "height", "weight",
    "location", "contact", "allergy", "condition", "medication",
    "date", "systolic", "diastolic", "heart", "temp", "spo2",
    "resp", "lab", "encounter", "cost", "smoker", "alcohol",
    "activity", "diet", "notes"
]

BUCKET_PATTERNS: Dict[str, List[str]] = {
    "id": ["patient_id", "mrn", "id", "subject_id", "record_id"],
    "name": ["name", "full_name", "patient_name", "first_name"],
    "age": ["age", "age_years"],
    "gender": ["gender", "sex"],
    "blood": ["blood_group", "blood_type", "bloodgroup"],
    "height": ["height", "height_cm", "height_m", "height_in"],
    "weight": ["weight", "weight_kg", "weight_lbs"],
    "location": ["city", "state", "region", "address", "pincode"],
    "contact": ["phone", "mobile", "email"],
    "allergy": ["allergies", "allergy", "drug_allergies"],
    "condition": [
        "chronic_conditions", "conditions", "diagnosis", "diagnoses",
        "icd10", "disease", "condition_name"
    ],
    "medication": [
        "current_medications", "medications", "meds", "drugs",
        "drug_name", "prescription"
    ],
    "systolic": ["systolic", "systolic_bp", "bp_systolic", "sbp"],
    "diastolic": ["diastolic", "diastolic_bp", "bp_diastolic", "dbp"],
    "heart": ["heart_rate", "pulse", "hr", "pulse_rate"],
    "temp": ["temperature", "temp", "body_temp"],
    "spo2": ["spo2", "oxygen_saturation", "o2_sat"],
    "resp": ["respiratory_rate", "rr", "breaths_per_min"],
    "lab": [
        "hba1c", "glucose", "glucose_fasting", "creatinine", "cholesterol",
        "cholesterol_total", "ldl", "hdl", "hemoglobin", "hemoglobin_g_dl",
        "bun", "bilirubin", "alt", "ast", "wbc", "rbc", "platelets",
        "sodium", "potassium", "calcium", "triglycerides", "tsh", "crp",
        "esr", "inr", "albumin", "troponin", "alkaline_phosphatase", "urea",
        "lab_value", "test_name", "analyte"
    ],
    "encounter": [
        "encounter_type", "visit_type", "admission_type", "duration_days",
        "length_of_stay", "los"
    ],
    "cost": ["cost", "amount", "charge", "billed_amount", "fees"],
    "smoker": ["smoker", "smoking", "smoking_status", "tobacco"],
    "alcohol": ["alcohol", "drinking", "alcohol_use"],
    "activity": ["physical_activity", "exercise", "activity_level"],
    "diet": ["diet", "diet_type"],
    "notes": ["notes", "remarks", "comments", "observations"],
}


def load_any_csv(file: Any) -> pd.DataFrame:
    """Read CSV, XLSX, or XLS into a pandas DataFrame.

    Tries utf-8, then latin-1, then utf-16 with automatic BOM handling.
    If read fails, raises ValueError with the exact error.

    Parameters
    ----------
    file : Any
        File path, bytes stream, or Streamlit UploadedFile.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    if file is None:
        raise ValueError("No file provided to load.")

    fname = getattr(file, "name", str(file)) if not isinstance(file, (str, Path)) else str(file)
    ext = Path(fname).suffix.lower()
    is_excel = ext in (".xlsx", ".xls")

    def _rewind(f: Any) -> None:
        if hasattr(f, "seek"):
            try:
                f.seek(0)
            except Exception:
                pass

    if is_excel:
        try:
            _rewind(file)
            return pd.read_excel(file)
        except Exception as exc:
            raise ValueError(f"Failed to read Excel file '{fname}': {exc}") from exc

    # Check for BOM / binary signatures
    raw_head = b""
    try:
        _rewind(file)
        if hasattr(file, "read"):
            raw_head = file.read(4)
        elif isinstance(file, (str, Path)):
            with open(file, "rb") as bf:
                raw_head = bf.read(4)
    except Exception:
        pass
    finally:
        _rewind(file)

    if isinstance(raw_head, str):
        raw_head = raw_head.encode("utf-8", errors="ignore")

    if raw_head.startswith(b"\xff\xfe") or raw_head.startswith(b"\xfe\xff"):
        encodings = ["utf-16", "utf-8-sig", "utf-8", "latin-1"]
    elif raw_head.startswith(b"\xef\xbb\xbf"):
        encodings = ["utf-8-sig", "utf-8", "utf-16", "latin-1"]
    else:
        encodings = ["utf-8-sig", "utf-8", "utf-16", "latin-1"]

    last_err: Optional[Exception] = None

    for enc in encodings:
        _rewind(file)
        try:
            df = pd.read_csv(file, encoding=enc)
            # If latin-1 decoded null characters in column names from utf-16, discard
            if any("\x00" in str(c) for c in df.columns):
                continue
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except UnicodeError as u_err:
            last_err = u_err
            continue
        except Exception as err:
            last_err = err
            continue

    raise ValueError(f"Failed to parse CSV file '{fname}': {last_err}")


def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Classify DataFrame columns into semantic clinical categories.

    Matching is case-insensitive and ignores spaces, underscores, hyphens,
    and units in parentheses using startswith/endswith checks.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.

    Returns
    -------
    dict
        Mapping from semantic bucket names to matching column names present in df.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or len(df.columns) == 0:
        return {}

    classified: Dict[str, List[str]] = {b: [] for b in ALL_SEMANTIC_BUCKETS}

    def _clean_stem(s: str) -> str:
        s_clean = re.sub(r"\(.*?\)", "", str(s))
        s_clean = re.sub(r"\[.*?\]", "", s_clean)
        return re.sub(r"[^a-zA-Z0-9]", "", s_clean).lower()

    def _get_tokens(s: str) -> List[str]:
        s_clean = re.sub(r"\(.*?\)", "", str(s))
        s_clean = re.sub(r"\[.*?\]", "", s_clean)
        return [t.lower() for t in re.split(r"[_\s\-]+", s_clean) if t.strip()]

    for col in df.columns:
        col_str = str(col)
        c_clean = _clean_stem(col_str)
        tokens = _get_tokens(col_str)

        # 1. Datetime Check
        is_date = False
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            is_date = True
        elif any(k in tokens for k in ("date", "time", "dob", "timestamp", "datetime", "recorded")):
            is_date = True
        elif c_clean.startswith("date") or c_clean.endswith("date") or c_clean in ("dob", "timestamp", "datetime"):
            is_date = True
        else:
            # Check non-numeric sample
            non_null = df[col].dropna()
            if len(non_null) > 0 and not pd.api.types.is_numeric_dtype(non_null):
                sample = non_null.iloc[:10]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().sum() >= max(1, int(len(sample) * 0.7)):
                        is_date = True
                except Exception:
                    pass

        if is_date:
            classified["date"].append(col_str)

        # 2. Match standard semantic buckets
        for bucket, patterns in BUCKET_PATTERNS.items():
            matched_bucket = False
            for pat in patterns:
                p_clean = _clean_stem(pat)
                p_tokens = _get_tokens(pat)

                # Token exact match
                if p_clean in tokens or any(t == p_clean for t in tokens):
                    matched_bucket = True
                    break

                # Substring/stem match
                if len(p_clean) >= 3:
                    if c_clean == p_clean or c_clean.startswith(p_clean) or c_clean.endswith(p_clean):
                        matched_bucket = True
                        break
                else:
                    # Short stems (like 'id', 'hr', 'rr') require token or boundary match
                    if p_clean in tokens:
                        matched_bucket = True
                        break
                    if c_clean.endswith(p_clean) and any(c_clean.startswith(prefix) for prefix in ("patient", "record", "subject", "user", "mrn")):
                        matched_bucket = True
                        break

            if matched_bucket:
                classified[bucket].append(col_str)

    # 3. Check for Long Format (EAV: Entity-Attribute-Value) co-occurrence:
    # A category/event/type column + a field/metric/test/variable column + a value/result/reading column
    has_cat = any(any(k in _clean_stem(c) for k in ("category", "event", "type", "domain")) for c in df.columns)
    has_field = any(any(k in _clean_stem(c) for k in ("field", "metric", "test", "variable", "name", "measurement", "analyte")) for c in df.columns)
    has_value = any(any(k in _clean_stem(c) for k in ("value", "result", "reading", "val")) for c in df.columns)

    if (has_cat and has_field and has_value) or (has_field and has_value and len(df.columns) <= 6):
        classified["longformat"] = [
            str(c) for c in df.columns
            if any(k in _clean_stem(c) for k in ("event", "category", "type", "domain", "metric", "field", "test", "variable", "name", "value", "result", "val", "reading"))
        ]

    return {b: cols for b, cols in classified.items() if cols}


def classify_row_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return categorized column groupings based on co-occurring column types."""
    buckets = classify_columns(df)
    vitals_keys = {"systolic", "diastolic", "heart", "temp", "spo2", "resp"}
    labs_keys = {"lab"}
    diag_keys = {"condition"}
    med_keys = {"medication"}
    enc_keys = {"encounter", "cost"}
    common_keys = {"id", "date"}

    def _collect(target_keys: set) -> List[str]:
        cols = []
        for k in target_keys:
            cols.extend(buckets.get(k, []))
        if cols:
            for ck in common_keys:
                cols.extend(buckets.get(ck, []))
        return list(dict.fromkeys(cols))

    return {
        "vitals_cols": _collect(vitals_keys),
        "labs_cols": _collect(labs_keys),
        "diagnoses_cols": _collect(diag_keys),
        "medications_cols": _collect(med_keys),
        "encounters_cols": _collect(enc_keys),
    }


def expand_long_format(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Pivot long-format event datasets into standard clinical sub-frames.

    Never raises; skips sub-frames whose pivot fails.
    """
    result: Dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return result

    cat_col = next((c for c in df.columns if any(k in str(c).lower() for k in ("event_type", "category", "event", "type"))), None)
    field_col = next((c for c in df.columns if any(k in str(c).lower() for k in ("field", "metric", "test", "variable", "name"))), None)
    val_col = next((c for c in df.columns if any(k in str(c).lower() for k in ("value", "val", "result", "reading"))), None)

    if not cat_col or not field_col or not val_col:
        return result

    idx_cols = [c for c in df.columns if c not in (cat_col, field_col, val_col)]
    if not idx_cols:
        idx_cols = [df.index]

    for cat_val in df[cat_col].dropna().unique():
        cat_str = str(cat_val).strip()
        cat_lower = cat_str.lower()
        sub_df = df[df[cat_col] == cat_val].copy()

        role = None
        if "vital" in cat_lower:
            role = "vitals"
        elif "lab" in cat_lower:
            role = "labs"
        elif any(k in cat_lower for k in ("diag", "condition", "disease")):
            role = "diagnoses"
        elif any(k in cat_lower for k in ("med", "drug", "presc")):
            role = "medications"
        elif any(k in cat_lower for k in ("encounter", "visit", "admission")):
            role = "encounters"
        elif any(k in cat_lower for k in ("patient", "info", "demo")):
            role = "demographics"

        if not role:
            role = cat_lower

        try:
            pivoted = sub_df.pivot_table(
                index=idx_cols,
                columns=field_col,
                values=val_col,
                aggfunc="first",
            ).reset_index()
            pivoted.columns.name = None
            # Automatically coerce numeric columns in pivoted tables
            for col_name in pivoted.columns:
                if col_name not in idx_cols:
                    try:
                        conv = pd.to_numeric(pivoted[col_name], errors="coerce")
                        if conv.notna().sum() >= max(1, int(len(conv) * 0.5)):
                            pivoted[col_name] = conv
                    except Exception:
                        pass
            result[role] = pivoted
        except Exception:
            result[role] = sub_df

    return result


def shape_dataset(df: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame], Any]) -> Dict[str, Any]:
    """Master entry point. Given ANY dataframe (or list of frames), return structured clinical dict.

    Returns
    -------
    dict
      {
        "demographics": pd.DataFrame (1 row) | None,
        "vitals":       pd.DataFrame | None,
        "labs":         pd.DataFrame | None,
        "diagnoses":    pd.DataFrame | None,
        "medications":  pd.DataFrame | None,
        "encounters":   pd.DataFrame | None,
        "_summary":     dict (rows, cols, categories_found, categories_missing),
        "_warnings":    list[str]
      }
    """
    shape: Dict[str, Any] = {
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
        "_warnings": [],
    }

    if df is None:
        shape["_warnings"].append("No dataset provided.")
        return shape

    # Multi-upload handling: list or tuple of DataFrames
    if isinstance(df, (list, tuple)):
        frames = [f for f in df if isinstance(f, pd.DataFrame) and not f.empty]
        if not frames:
            shape["_warnings"].append("Uploaded file(s) are empty.")
            return shape

        for single_df in frames:
            sub_shape = shape_dataset(single_df)
            for role in ("demographics", "vitals", "labs", "diagnoses", "medications", "encounters"):
                if sub_shape.get(role) is not None and not sub_shape[role].empty:
                    if shape[role] is None:
                        shape[role] = sub_shape[role]
                    else:
                        shape[role] = pd.concat([shape[role], sub_shape[role]], ignore_index=True)
            shape["_summary"]["rows"] += sub_shape["_summary"]["rows"]
            shape["_summary"]["cols"] += sub_shape["_summary"]["cols"]
            shape["_warnings"].extend(sub_shape.get("_warnings", []))

        found = [r for r in ("demographics", "vitals", "labs", "diagnoses", "medications", "encounters") if shape[r] is not None and not shape[r].empty]
        missing = [r for r in ("demographics", "vitals", "labs", "diagnoses", "medications", "encounters") if r not in found]
        shape["_summary"]["categories_found"] = found
        shape["_summary"]["categories_missing"] = missing
        return shape

    if not isinstance(df, pd.DataFrame):
        shape["_warnings"].append("Provided data could not be converted to a DataFrame.")
        return shape

    if df.empty or len(df.columns) == 0:
        shape["_warnings"].append("File is empty.")
        return shape

    total_rows, total_cols = df.shape
    shape["_summary"]["rows"] = total_rows
    shape["_summary"]["cols"] = total_cols

    classified = classify_columns(df)

    # 1. Long-format expansion
    if "longformat" in classified:
        expanded = expand_long_format(df)
        for role in ("demographics", "vitals", "labs", "diagnoses", "medications", "encounters"):
            if role in expanded and not expanded[role].empty:
                shape[role] = expanded[role]

    else:
        # 2. Wide single-row (1 row, many columns)
        if total_rows == 1:
            shape["demographics"] = df.copy()

            lab_cols = classified.get("lab", [])
            if lab_cols:
                shape["labs"] = df[lab_cols].copy()

            vitals_cols = []
            for vk in ("systolic", "diastolic", "heart", "temp", "spo2", "resp"):
                vitals_cols.extend(classified.get(vk, []))
            if vitals_cols:
                shape["vitals"] = df[vitals_cols].copy()

            cond_cols = classified.get("condition", [])
            if cond_cols:
                shape["diagnoses"] = df[cond_cols].copy()

            med_cols = classified.get("medication", [])
            if med_cols:
                shape["medications"] = df[med_cols].copy()

        else:
            # 3. Multi-row dataset
            # Vitals
            vitals_cols = []
            for vk in ("systolic", "diastolic", "heart", "temp", "spo2", "resp"):
                vitals_cols.extend(classified.get(vk, []))
            if vitals_cols:
                cols_to_use = list(dict.fromkeys(classified.get("date", []) + classified.get("id", []) + vitals_cols))
                shape["vitals"] = df[cols_to_use].copy()

            # Labs
            lab_cols = classified.get("lab", [])
            if lab_cols:
                cols_to_use = list(dict.fromkeys(classified.get("date", []) + classified.get("id", []) + lab_cols))
                shape["labs"] = df[cols_to_use].copy()

            # Diagnoses / Conditions
            cond_cols = classified.get("condition", [])
            if cond_cols:
                cols_to_use = list(dict.fromkeys(classified.get("date", []) + classified.get("id", []) + cond_cols))
                shape["diagnoses"] = df[cols_to_use].copy()

            # Medications
            med_cols = classified.get("medication", [])
            if med_cols:
                cols_to_use = list(dict.fromkeys(classified.get("date", []) + classified.get("id", []) + med_cols))
                shape["medications"] = df[cols_to_use].copy()

            # Encounters
            enc_cols = classified.get("encounter", []) + classified.get("cost", [])
            if enc_cols:
                cols_to_use = list(dict.fromkeys(classified.get("date", []) + classified.get("id", []) + enc_cols))
                shape["encounters"] = df[cols_to_use].copy()

            # Demographics
            demo_cols = []
            for dk in ("id", "name", "age", "gender", "blood", "height", "weight", "location", "contact", "allergy", "smoker", "alcohol", "activity", "diet"):
                demo_cols.extend(classified.get(dk, []))
            if demo_cols:
                shape["demographics"] = df[demo_cols].iloc[[0]].copy()

    # Calculate categories found & missing
    found = [r for r in ("demographics", "vitals", "labs", "diagnoses", "medications", "encounters") if shape[r] is not None and not shape[r].empty]
    missing = [r for r in ("demographics", "vitals", "labs", "diagnoses", "medications", "encounters") if r not in found]

    shape["_summary"]["categories_found"] = found
    shape["_summary"]["categories_missing"] = missing

    if not found:
        shape["_warnings"].append("No recognized clinical categories found in dataset.")

    return shape
