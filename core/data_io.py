"""Data ingestion, validation, and export utilities for HealthSure."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import pandas as pd
import streamlit as st

# Canonical synonym mappings for column normalization
SYNONYM_MAP: Dict[str, str] = {
    # Income
    "annual income": "income_lpa",
    "annual_income": "income_lpa",
    "annual income (lpa)": "income_lpa",
    "annual_income_lpa": "income_lpa",
    "income (lpa)": "income_lpa",
    "income_lpa": "income_lpa",
    "income(lpa)": "income_lpa",
    "income": "income_lpa",
    # Height
    "height (m)": "height",
    "height_m": "height",
    "height(m)": "height",
    "height_cm": "height",
    "height cm": "height",
    "height(cm)": "height",
    "height": "height",
    # Weight
    "weight": "weight",
    "weight_kg": "weight",
    "weight (kg)": "weight",
    "weight(kg)": "weight",
    "weight kg": "weight",
    # Smoker
    "smoker": "smoker",
    "smoking": "smoker",
    "smoking_status": "smoker",
    "smoking status": "smoker",
    "is_smoker": "smoker",
    # City Tier
    "city_tier": "city_tier",
    "city tier": "city_tier",
    "tier": "city_tier",
    # Risk Tier / Premium Category
    "premium_category": "risk_tier",
    "premium category": "risk_tier",
    "premium_cat": "risk_tier",
    "risk_tier": "risk_tier",
    "risk tier": "risk_tier",
    "risk_category": "risk_tier",
    "risk category": "risk_tier",
}


@st.cache_data
def read_uploaded_file(uploaded_file: Any) -> pd.DataFrame:
    """Read an uploaded file (CSV, XLSX, or XLS) into a pandas DataFrame.

    Tries UTF-8 encoding first, falling back to Latin-1 for CSV files.
    Raises a clean ValueError on format or decoding failures.

    Parameters
    ----------
    uploaded_file : Any
        Streamlit UploadedFile, file-like object, or file path string.

    Returns
    -------
    pd.DataFrame
        Parsed DataFrame.

    Raises
    -------
    ValueError
        If the file cannot be parsed or format is unsupported.
    """
    if uploaded_file is None:
        raise ValueError("No file provided for upload.")

    filename = ""
    if hasattr(uploaded_file, "name"):
        filename = str(uploaded_file.name)
    elif isinstance(uploaded_file, (str, Path)):
        filename = str(uploaded_file)

    ext = Path(filename).suffix.lower() if filename else ""

    def _rewind():
        if hasattr(uploaded_file, "seek"):
            try:
                uploaded_file.seek(0)
            except Exception:
                pass

    # Excel formats
    if ext in (".xlsx", ".xls"):
        try:
            _rewind()
            return pd.read_excel(uploaded_file)
        except Exception as exc:
            clean_name = filename or "Excel file"
            raise ValueError(f"Could not read {clean_name}: {exc}") from exc

    # CSV format (or unknown extension fallback to CSV attempt)
    if ext == ".csv" or not ext:
        try:
            _rewind()
            return pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                _rewind()
                return pd.read_csv(uploaded_file, encoding="latin-1")
            except Exception as exc:
                raise ValueError(
                    f"Could not decode CSV using UTF-8 or Latin-1: {exc}"
                ) from exc
        except Exception as exc:
            # Check if Latin-1 succeeds for other read exceptions
            try:
                _rewind()
                return pd.read_csv(uploaded_file, encoding="latin-1")
            except Exception:
                clean_name = filename or "CSV file"
                raise ValueError(f"Could not read {clean_name}: {exc}") from exc

    raise ValueError(
        f"Unsupported file format '{ext}'. Please upload a CSV (.csv) or Excel (.xlsx, .xls) file."
    )


def _snake_case(text: str) -> str:
    """Convert arbitrary string into stripped, lowercase snake_case."""
    cleaned = str(text).strip()
    cleaned = re.sub(r"[^\w\s-]", " ", cleaned)
    cleaned = re.sub(r"[\s-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_").lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe column names to lowercase, stripped snake_case and canonical synonyms.

    Canonical synonyms:
      - 'annual income' / 'income (lpa)' -> 'income_lpa'
      - 'height (m)' / 'height_cm' / 'height' -> 'height'
      - 'weight' / 'weight_kg' -> 'weight'
      - 'smoker' / 'smoking' / 'smoking_status' -> 'smoker'
      - 'city_tier' / 'tier' -> 'city_tier'
      - 'premium_category' / 'risk_tier' -> 'risk_tier'

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        New DataFrame with normalized column names.
    """
    if df is None:
        raise ValueError("DataFrame cannot be None.")

    df_out = df.copy()
    new_cols: List[str] = []

    for col in df.columns:
        raw_str = str(col).strip().lower()

        # Check raw lowercase string against synonyms
        if raw_str in SYNONYM_MAP:
            new_cols.append(SYNONYM_MAP[raw_str])
            continue

        # Snake case transformation
        snake = _snake_case(col)

        # Check snake_cased version against synonyms
        if snake in SYNONYM_MAP:
            new_cols.append(SYNONYM_MAP[snake])
        else:
            new_cols.append(snake)

    df_out.columns = new_cols
    return df_out


def coerce_types(
    df: pd.DataFrame,
    warnings: Optional[List[str]] = None,
    return_warnings: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[str]]]:
    """Coerce core HealthSure dataset columns to their target types without dropping rows.

    Type specifications:
      - age: int
      - income_lpa: float
      - height: float (if height > 3.0, treated as cm and divided by 100)
      - weight: float
      - smoker: 0/1 (maps 'yes'/'no', booleans, and numeric indicators)
      - city_tier: int (extracts integer tier number)

    Parameters
    ----------
    df : pd.DataFrame
        Normalized DataFrame.
    warnings : list of str, optional
        External list to collect warning messages.
    return_warnings : bool, default False
        If True, returns (df, warnings) tuple. Otherwise returns df.

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, list of str)
        Coerced DataFrame (and optional list of warnings).
    """
    if df is None:
        raise ValueError("DataFrame cannot be None.")

    df_out = df.copy()
    warnings_list = warnings if warnings is not None else []

    # 1. age: integer
    if "age" in df_out.columns:
        orig_na = df_out["age"].isna().sum()
        numeric_age = pd.to_numeric(df_out["age"], errors="coerce")
        new_na = numeric_age.isna().sum() - orig_na
        if new_na > 0:
            warnings_list.append(
                f"Column 'age': {new_na} non-numeric value(s) encountered; coerced to missing without dropping rows."
            )
        if numeric_age.isna().any():
            df_out["age"] = numeric_age.round().astype("Int64")
        else:
            df_out["age"] = numeric_age.round().astype(int)

    # 2. income_lpa: float
    if "income_lpa" in df_out.columns:
        orig_na = df_out["income_lpa"].isna().sum()
        numeric_inc = pd.to_numeric(df_out["income_lpa"], errors="coerce")
        new_na = numeric_inc.isna().sum() - orig_na
        if new_na > 0:
            warnings_list.append(
                f"Column 'income_lpa': {new_na} non-numeric value(s) encountered; coerced to missing."
            )
        df_out["income_lpa"] = numeric_inc.astype(float)

    # 3. height: float (if height > 3.0, treat as cm, divide by 100)
    if "height" in df_out.columns:
        orig_na = df_out["height"].isna().sum()
        numeric_h = pd.to_numeric(df_out["height"], errors="coerce")
        new_na = numeric_h.isna().sum() - orig_na
        if new_na > 0:
            warnings_list.append(
                f"Column 'height': {new_na} non-numeric value(s) encountered; coerced to missing."
            )

        cm_mask = numeric_h > 3.0
        cm_count = int(cm_mask.sum())
        if cm_count > 0:
            warnings_list.append(
                f"Column 'height': {cm_count} value(s) > 3.0 were treated as centimeters and converted to meters (divided by 100)."
            )
            numeric_h = numeric_h.where(~cm_mask, numeric_h / 100.0)

        df_out["height"] = numeric_h.astype(float)

    # 4. weight: float
    if "weight" in df_out.columns:
        orig_na = df_out["weight"].isna().sum()
        numeric_w = pd.to_numeric(df_out["weight"], errors="coerce")
        new_na = numeric_w.isna().sum() - orig_na
        if new_na > 0:
            warnings_list.append(
                f"Column 'weight': {new_na} non-numeric value(s) encountered; coerced to missing."
            )
        df_out["weight"] = numeric_w.astype(float)

    # 5. smoker: 0/1 (maps 'yes'/'no', booleans, and numbers)
    if "smoker" in df_out.columns:
        orig_na = df_out["smoker"].isna().sum()

        def _map_smoker_val(val: Any) -> Optional[int]:
            if pd.isna(val):
                return None
            s = str(val).strip().lower()
            if s in ("yes", "y", "true", "1", "1.0", "smoker"):
                return 1
            if s in ("no", "n", "false", "0", "0.0", "non-smoker", "non smoker"):
                return 0
            try:
                num = float(s)
                return 1 if num > 0 else 0
            except Exception:
                return None

        mapped_smoker = df_out["smoker"].apply(_map_smoker_val)
        new_na = mapped_smoker.isna().sum() - orig_na
        if new_na > 0:
            warnings_list.append(
                f"Column 'smoker': {new_na} value(s) could not be mapped to 0/1; coerced to missing."
            )

        if mapped_smoker.isna().any():
            df_out["smoker"] = mapped_smoker.astype("Int64")
        else:
            df_out["smoker"] = mapped_smoker.astype(int)

    # 6. city_tier: integer (or derive from city if city is present)
    TIER_1_CITIES = {
        "bangalore", "chennai", "delhi", "hyderabad", "kolkata", "mumbai", "pune", "ahmedabad", "bengaluru", "new delhi"
    }
    TIER_2_CITIES = {
        "agra", "allahabad", "amritsar", "bhopal", "faridabad", "ghaziabad",
        "indore", "jaipur", "kanpur", "lucknow", "ludhiana", "meerut",
        "nagpur", "nashik", "patna", "rajkot", "ranchi", "srinagar",
        "surat", "vadodara", "varanasi", "visakhapatnam", "prayagraj"
    }

    def _derive_tier_from_city(val: Any) -> int:
        if pd.isna(val):
            return 2
        c = str(val).strip().lower()
        if c in TIER_1_CITIES or "1" in c or "metro" in c:
            return 1
        if c in TIER_2_CITIES or "2" in c:
            return 2
        return 3

    if "city_tier" not in df_out.columns and "city" in df_out.columns:
        df_out["city_tier"] = df_out["city"].apply(_derive_tier_from_city).astype(int)
        warnings_list.append("Derived 'city_tier' from 'city' column based on Indian metropolitan classifications.")
    elif "city_tier" in df_out.columns:
        orig_na = df_out["city_tier"].isna().sum()

        def _parse_tier_val(val: Any) -> Optional[int]:
            if pd.isna(val):
                return None
            if isinstance(val, (int, float)):
                return int(val)
            match = re.search(r"\d+", str(val))
            if match:
                return int(match.group())
            return _derive_tier_from_city(val)

        parsed_tier = df_out["city_tier"].apply(_parse_tier_val)
        new_na = parsed_tier.isna().sum() - orig_na
        if new_na > 0:
            warnings_list.append(
                f"Column 'city_tier': {new_na} value(s) could not be parsed as integers; coerced to missing."
            )

        if parsed_tier.isna().any():
            df_out["city_tier"] = parsed_tier.astype("Int64")
        else:
            df_out["city_tier"] = parsed_tier.astype(int)

    # Attach warnings to DataFrame metadata and Streamlit session state
    df_out.attrs["warnings"] = warnings_list
    try:
        st.session_state["dataset_warnings"] = warnings_list
    except Exception:
        pass

    if return_warnings:
        return df_out, warnings_list
    return df_out


def detect_column_roles(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect and categorize dataset columns into numeric, categorical, and datetime roles.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.

    Returns
    -------
    dict
        Dictionary with keys 'numeric', 'categorical', and 'datetime'.
    """
    if df is None or df.empty:
        return {"numeric": [], "categorical": [], "datetime": []}

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    datetime_cols: List[str] = []

    for col in df.columns:
        series = df[col]
        # Datetime check
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
        # Numeric check (excluding booleans)
        elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric_cols.append(col)
        # Categorical / string / object / boolean
        else:
            datetime_parsed = False
            if series.dtype == object and series.dropna().count() > 0:
                col_lower = str(col).lower()
                if any(k in col_lower for k in ("date", "time", "timestamp", "dob", "created")):
                    try:
                        sample = series.dropna().head(10)
                        pd.to_datetime(sample, errors="raise")
                        datetime_cols.append(col)
                        datetime_parsed = True
                    except Exception:
                        pass
            if not datetime_parsed:
                categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
    }


def store_dataset(df: pd.DataFrame, name: str = "user_dataset") -> None:
    """Store dataset and its identifier in Streamlit session state.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to store.
    name : str, default 'user_dataset'
        Label or filename for the dataset.
    """
    try:
        st.session_state["dataset_df"] = df
        st.session_state["dataset_name"] = name
    except Exception:
        pass


def get_dataset() -> Optional[pd.DataFrame]:
    """Retrieve the stored dataset from Streamlit session state.

    Returns
    -------
    pd.DataFrame or None
        Active dataset if present in session state, otherwise None.
    """
    try:
        return st.session_state.get("dataset_df", None)
    except Exception:
        return None


def reset_dataset() -> None:
    """Clear stored dataset and related metadata from Streamlit session state."""
    try:
        st.session_state.pop("dataset_df", None)
        st.session_state.pop("dataset_name", None)
        st.session_state.pop("dataset_warnings", None)
    except Exception:
        pass
