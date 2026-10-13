"""Machine learning model loading and prediction pipeline for HealthSure."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

MODEL_RELATIVE_PATH = Path("models") / "insurance_premium_model.pkl"


@st.cache_resource
def load_model() -> Pipeline:
    """Load the trained Random Forest insurance premium model pipeline.

    Loads the model once from disk and caches it in Streamlit resource cache.
    Raises a clean FileNotFoundError if the pickle file is missing.

    Returns
    -------
    Pipeline
        Trained scikit-learn Pipeline.

    Raises
    -------
    FileNotFoundError
        If 'models/insurance_premium_model.pkl' is missing.
    RuntimeError
        If unpickling fails.
    """
    # Check absolute path relative to project structure and cwd
    project_root = Path(__file__).resolve().parent.parent
    candidate_paths = [
        project_root / MODEL_RELATIVE_PATH,
        Path.cwd() / MODEL_RELATIVE_PATH,
        Path(MODEL_RELATIVE_PATH),
    ]

    model_path = None
    for p in candidate_paths:
        if p.is_file():
            model_path = p
            break

    if model_path is None:
        raise FileNotFoundError(
            f"Model file not found. Looked in {[str(p) for p in candidate_paths]}. "
            "Please ensure 'insurance_premium_model.pkl' exists in the 'models/' directory."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model from '{model_path}': {exc}"
            ) from exc


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered clinical and financial features for HealthSure.

    Adds EXACTLY:
      - bmi                 = weight / height^2
      - income_per_age      = income_lpa / age
      - weight_height_ratio = weight / height
      - income_bmi          = income_lpa * bmi
      - smoker_income       = smoker * income_lpa
      - health_risk_index   = 0.3*bmi + 0.4*age/100 + 0.3*smoker*10

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with canonical columns: weight, height, age, income_lpa, smoker.

    Returns
    -------
    pd.DataFrame
        DataFrame with the engineered feature columns added.
    """
    if df is None:
        raise ValueError("DataFrame cannot be None.")

    df_out = df.copy()

    # Convert to numeric series for calculation
    w = pd.to_numeric(df_out["weight"], errors="coerce")
    h = pd.to_numeric(df_out["height"], errors="coerce")
    age = pd.to_numeric(df_out["age"], errors="coerce")
    inc = pd.to_numeric(df_out["income_lpa"], errors="coerce")
    smoker = pd.to_numeric(df_out["smoker"], errors="coerce")

    # Feature calculations
    bmi = w / (h ** 2)
    income_per_age = inc / age
    weight_height_ratio = w / h
    income_bmi = inc * bmi
    smoker_income = smoker * inc
    health_risk_index = (0.3 * bmi) + (0.4 * age / 100.0) + (0.3 * smoker * 10.0)

    df_out["bmi"] = bmi
    df_out["income_per_age"] = income_per_age
    df_out["weight_height_ratio"] = weight_height_ratio
    df_out["income_bmi"] = income_bmi
    df_out["smoker_income"] = smoker_income
    df_out["health_risk_index"] = health_risk_index

    return df_out


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Predict risk categories and tier probabilities for a DataFrame of applicants.

    Input requires canonical columns.
    Output returns the input DataFrame augmented with:
      - risk_tier: 'Low', 'Medium', 'High' (or 'Unknown' on error)
      - prob_low: probability float (0.0 to 1.0)
      - prob_medium: probability float (0.0 to 1.0)
      - prob_high: probability float (0.0 to 1.0)
      - reason: error description if any row fails to predict

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with canonical demographic and health columns.

    Returns
    -------
    pd.DataFrame
        Augmented DataFrame with predictions and probabilities.
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")

    df_out = df.copy()
    if df_out.empty:
        df_out["risk_tier"] = []
        df_out["prob_low"] = []
        df_out["prob_medium"] = []
        df_out["prob_high"] = []
        return df_out

    # 1. Feature Engineering
    try:
        required_eng = [
            "bmi",
            "income_per_age",
            "weight_height_ratio",
            "income_bmi",
            "smoker_income",
            "health_risk_index",
        ]
        if not all(col in df_out.columns for col in required_eng):
            df_features = engineer_features(df_out)
        else:
            df_features = df_out.copy()
    except Exception as exc:
        df_out["risk_tier"] = "Unknown"
        df_out["prob_low"] = 0.0
        df_out["prob_medium"] = 0.0
        df_out["prob_high"] = 0.0
        df_out["reason"] = f"Feature engineering failed: {exc}"
        return df_out
    # Ensure categorical columns required by model exist
    TIER_1_CITIES = {
        "bangalore", "chennai", "delhi", "hyderabad", "kolkata", "mumbai", "pune", "ahmedabad", "bengaluru", "new delhi"
    }
    TIER_2_CITIES = {
        "agra", "allahabad", "amritsar", "bhopal", "faridabad", "ghaziabad",
        "indore", "jaipur", "kanpur", "lucknow", "ludhiana", "meerut",
        "nagpur", "nashik", "patna", "rajkot", "ranchi", "srinagar",
        "surat", "vadodara", "varanasi", "visakhapatnam", "prayagraj"
    }

    if "city_tier" not in df_features.columns:
        if "city" in df_features.columns:
            def _derive_tier(val: Any) -> int:
                c = str(val).strip().lower()
                if c in TIER_1_CITIES or "1" in c or "metro" in c:
                    return 1
                if c in TIER_2_CITIES or "2" in c:
                    return 2
                return 3
            df_features["city_tier"] = df_features["city"].apply(_derive_tier).astype(int)
        else:
            df_features["city_tier"] = 2
    else:
        df_features["city_tier"] = pd.to_numeric(df_features["city_tier"], errors="coerce").fillna(2).astype(int)

    if "occupation" not in df_features.columns:
        df_features["occupation"] = "Software Engineer"
    else:
        df_features["occupation"] = df_features["occupation"].fillna("Software Engineer").astype(str)

    if "smoker" not in df_features.columns:
        df_features["smoker"] = 0
    else:
        df_features["smoker"] = pd.to_numeric(df_features["smoker"], errors="coerce").fillna(0).astype(int)


    # 2. Model Loading
    try:
        model = load_model()
    except Exception as exc:
        df_out["risk_tier"] = "Unknown"
        df_out["prob_low"] = 0.0
        df_out["prob_medium"] = 0.0
        df_out["prob_high"] = 0.0
        df_out["reason"] = f"Model load failed: {exc}"
        return df_out

    # Determine class index mapping dynamically
    class_indices: Dict[str, int] = {
        str(c).title(): i for i, c in enumerate(model.classes_)
    }
    idx_low = class_indices.get("Low", 1)
    idx_med = class_indices.get("Medium", 2)
    idx_high = class_indices.get("High", 0)

    # 3. Model Inference (Batch with graceful row-by-row fallback)
    try:
        preds = model.predict(df_features)
        probs = model.predict_proba(df_features)

        df_out = df_features.copy()
        df_out["risk_tier"] = [str(p).title() for p in preds]
        df_out["prob_low"] = probs[:, idx_low]
        df_out["prob_medium"] = probs[:, idx_med]
        df_out["prob_high"] = probs[:, idx_high]
        return df_out

    except Exception:
        # Batch inference failed (e.g. unseen categories or missing values);
        # fall back to individual row inference so valid records succeed.
        risk_tiers: List[str] = []
        prob_lows: List[float] = []
        prob_meds: List[float] = []
        prob_highs: List[float] = []
        reasons: List[Optional[str]] = []
        has_error = False

        for _, row in df_features.iterrows():
            row_df = pd.DataFrame([row])
            try:
                p = model.predict(row_df)[0]
                pr = model.predict_proba(row_df)[0]
                risk_tiers.append(str(p).title())
                prob_lows.append(float(pr[idx_low]))
                prob_meds.append(float(pr[idx_med]))
                prob_highs.append(float(pr[idx_high]))
                reasons.append(None)
            except Exception as row_exc:
                has_error = True
                risk_tiers.append("Unknown")
                prob_lows.append(0.0)
                prob_meds.append(0.0)
                prob_highs.append(0.0)
                reasons.append(str(row_exc))

        df_out = df_features.copy()
        df_out["risk_tier"] = risk_tiers
        df_out["prob_low"] = prob_lows
        df_out["prob_medium"] = prob_meds
        df_out["prob_high"] = prob_highs
        if has_error:
            df_out["reason"] = reasons

        return df_out


def predict_single(record: Dict[str, Any]) -> Dict[str, Any]:
    """Predict risk tier and probabilities for a single applicant dictionary record.

    Parameters
    ----------
    record : dict
        Dictionary of applicant features.

    Returns
    -------
    dict
        Record dictionary augmented with 'risk_tier', 'prob_low',
        'prob_medium', and 'prob_high'. On error, sets risk_tier='Unknown'
        and adds 'reason'.
    """
    df_single = pd.DataFrame([record])
    res_df = predict_dataframe(df_single)
    return res_df.to_dict(orient="records")[0]
