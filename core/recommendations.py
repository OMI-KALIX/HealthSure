"""Rule-based underwriting recommendation engine for HealthSure.

Deterministic, transparent actuarial recommendations derived from the applicant's
predicted risk tier and individual biometric/socioeconomic features.
Zero external LLM / network dependencies.
"""
from typing import Any, Dict, List, Optional


def rule_engine(tier: str, features: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate deterministic underwriting rules against applicant features.

    Parameters
    ----------
    tier : str
        Predicted risk tier: "Low", "Medium", "High", or "Unknown".
    features : dict
        Applicant features (age, bmi, smoker, city_tier, income_lpa, etc.).

    Returns
    -------
    list of dict
        Matched rules, each containing:
        {id, when, title, detail, priority, category}
    """
    matched: List[Dict[str, Any]] = []
    t_clean = str(tier).strip().capitalize() if tier else "Unknown"

    age = float(features.get("age", 30))
    bmi = float(features.get("bmi", 23.0))
    smoker = int(features.get("smoker", 0))
    city_tier = int(features.get("city_tier", 2))
    income_lpa = float(features.get("income_lpa", 10.0))

    bmi_cat = (
        "underweight"
        if bmi < 18.5
        else ("optimal" if bmi < 25.0 else ("overweight" if bmi < 30.0 else "obese"))
    )

    if t_clean == "Low":
        if smoker == 0:
            matched.append({
                "id": "LOW_SMOKER_0",
                "when": "smoker == 0",
                "title": "Stay smoke-free",
                "detail": "Non-smokers save 15–25% on premiums. Kicking the habit permanently locks in this advantage.",
                "priority": "medium",
                "category": "lifestyle",
            })
        if 18.5 <= bmi <= 24.9:
            matched.append({
                "id": "LOW_BMI_NORMAL",
                "when": "18.5 <= bmi <= 24.9",
                "title": "Maintain healthy BMI",
                "detail": f"Your BMI of {bmi:.1f} is in the optimal range. Regular activity (150 min/week) keeps it here.",
                "priority": "medium",
                "category": "clinical",
            })
        if age < 40:
            matched.append({
                "id": "LOW_AGE_UNDER_40",
                "when": "age < 40",
                "title": "Lock in long-term cover",
                "detail": "Premiums rise ~3–5% per year of age. Starting early locks in today's rate.",
                "priority": "medium",
                "category": "policy",
            })
        matched.append({
            "id": "LOW_ANNUAL_CHECKUP",
            "when": "always",
            "title": "Annual health checkup",
            "detail": "Yearly screening keeps risk factors in check and may unlock loyalty discounts.",
            "priority": "medium",
            "category": "preventive",
        })

    elif t_clean == "Medium":
        if smoker == 1:
            matched.append({
                "id": "MED_SMOKER_1",
                "when": "smoker == 1",
                "title": "Smoking is the largest single risk driver",
                "detail": "Quitting for 12 months can drop you into a lower tier and reduce premiums by up to 25%.",
                "priority": "high",
                "category": "lifestyle",
            })
        if bmi >= 25.0:
            matched.append({
                "id": "MED_BMI_ELEVATED",
                "when": "bmi >= 25.0",
                "title": "Elevated BMI",
                "detail": f"Your BMI of {bmi:.1f} places you in the {bmi_cat} range. A 5–7% weight reduction meaningfully improves your risk score.",
                "priority": "high",
                "category": "clinical",
            })
        if 40 <= age <= 55 and smoker == 0:
            matched.append({
                "id": "MED_AGE_40_55_NON_SMOKER",
                "when": "40 <= age <= 55 and smoker == 0",
                "title": "Age-related risk",
                "detail": "Regular screening (BP, lipids, HbA1c) helps keep this tier stable as you age.",
                "priority": "high",
                "category": "preventive",
            })
        if city_tier == 1:
            matched.append({
                "id": "MED_CITY_TIER_1",
                "when": "city_tier == 1",
                "title": "Metro lifestyle load",
                "detail": "Tier-1 city living adds stress and pollution exposure. Prioritize sleep, diet, and checkups.",
                "priority": "high",
                "category": "lifestyle",
            })

    elif t_clean == "High":
        if smoker == 1:
            matched.append({
                "id": "HIGH_SMOKER_1",
                "when": "smoker == 1",
                "title": "Smoking: primary driver",
                "detail": "Smokers in your profile carry ~2.5x the claim probability. Cessation is the highest-leverage action.",
                "priority": "high",
                "category": "lifestyle",
            })
        if bmi >= 30.0:
            matched.append({
                "id": "HIGH_BMI_OBESE",
                "when": "bmi >= 30.0",
                "title": "Obesity range BMI",
                "detail": f"A BMI of {bmi:.1f} is in the obese range. Losing 5–10% body weight typically reclassifies applicants within 12 months.",
                "priority": "high",
                "category": "clinical",
            })
        if age >= 55:
            matched.append({
                "id": "HIGH_AGE_OVER_55",
                "when": "age >= 55",
                "title": "Age band",
                "detail": "Applicants 55+ are underwritten with a higher base rate. Focus on modifiable factors.",
                "priority": "high",
                "category": "clinical",
            })
        if income_lpa < 5.0:
            matched.append({
                "id": "HIGH_INCOME_LOW",
                "when": "income_lpa < 5.0",
                "title": "Income-to-cover ratio",
                "detail": "Consider a lower sum-assured base policy plus a top-up to keep premiums affordable.",
                "priority": "high",
                "category": "financial",
            })
        if city_tier == 1 and smoker == 1:
            matched.append({
                "id": "HIGH_METRO_SMOKER",
                "when": "city_tier == 1 and smoker == 1",
                "title": "Compounding metro + smoking risk",
                "detail": "Combined metro pollution and smoking raise respiratory and cardiac risk significantly.",
                "priority": "high",
                "category": "lifestyle",
            })

    return matched


def recommend(tier: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """Generate structured, actionable recommendations tailored to an applicant.

    Parameters
    ----------
    tier : str
        Predicted risk tier ("Low", "Medium", "High", "Unknown").
    features : dict
        Features dictionary containing age, bmi, smoker, city_tier, etc.

    Returns
    -------
    dict
        {
          "headline": str,
          "actions": list[dict],    # {title, detail, priority}
          "maintain": list[dict],   # {title, detail, priority}
          "warnings": list[dict],   # {title, detail, priority}
          "riders": list[str],      # suggested riders
          "checklist": list[str],   # next steps
        }
    """
    t_clean = str(tier).strip().capitalize() if tier else "Unknown"
    rules = rule_engine(t_clean, features)

    bmi = float(features.get("bmi", 23.0))
    smoker = int(features.get("smoker", 0))

    if t_clean == "Low":
        headline = "Excellent profile. You qualify for our best rates."
        maintain = [
            {"title": r["title"], "detail": r["detail"], "priority": r["priority"]}
            for r in rules
        ]
        actions = []
        warnings = []
        riders = [
            "Critical Illness Rider",
            "Accidental Death Benefit",
            "Hospital Cash Rider",
            "Waiver of Premium on Disability",
        ]
        checklist = [
            "Schedule your next annual health checkup",
            "Review your policy every 12 months",
            "Consider adding a critical illness rider",
            "Keep your income and address details updated",
        ]

    elif t_clean == "Medium":
        headline = "Moderate risk. A few changes can lower your premium."
        warnings = [
            {"title": r["title"], "detail": r["detail"], "priority": r["priority"]}
            for r in rules
            if r["priority"] == "high"
        ]
        actions = [
            {
                "title": "Enroll in a structured wellness program",
                "detail": "Active participation in insurer-tracked wellness programs can yield premium renewal discounts.",
                "priority": "medium",
            },
            {
                "title": "Complete a preventive health screening within 90 days",
                "detail": "Early detection of blood pressure and glucose shifts prevents tier elevation.",
                "priority": "medium",
            },
            {
                "title": "Provide recent lab reports to potentially reclassify",
                "detail": "Normal fasting glucose and lipid panels can be submitted during underwriting review.",
                "priority": "medium",
            },
        ]
        maintain = []
        if smoker == 0:
            maintain.append({
                "title": "Stay smoke-free",
                "detail": "Maintaining your non-smoking status prevents compounding risk escalation.",
                "priority": "medium",
            })
        if 18.5 <= bmi < 25.0:
            maintain.append({
                "title": "Maintain healthy weight",
                "detail": f"Your BMI of {bmi:.1f} is well-controlled. Continue regular physical activity.",
                "priority": "medium",
            })
        maintain.append({
            "title": "Routine health monitoring",
            "detail": "Track routine biomarkers bi-annually to stay within standard underwriting bands.",
            "priority": "medium",
        })
        riders = [
            "Hospital Daily Cash",
            "Critical Illness Rider",
            "Maternity / OPD Rider (if applicable)",
        ]
        checklist = [
            "Book a full-body health checkup",
            "Set a smoking-cessation goal if applicable",
            "Re-submit updated vitals in 6 months",
            "Compare rider add-ons before renewal",
        ]

    elif t_clean == "High":
        headline = "High risk profile. Let's focus on the top drivers."
        warnings = [
            {"title": r["title"], "detail": r["detail"], "priority": r["priority"]}
            for r in rules
            if r["priority"] == "high"
        ]
        actions = [
            {
                "title": "Complete a medical examination for underwriting review",
                "detail": "Comprehensive medical underwriting tests clarify true risk and prevent claim disputes.",
                "priority": "high",
            },
            {
                "title": "Discuss rider exclusions with your underwriter",
                "detail": "Specific pre-existing exclusions may be negotiable after a waiting period.",
                "priority": "high",
            },
            {
                "title": "Consider a graded benefit plan for the first 2 years",
                "detail": "Graded policies provide lower entry rates while establishing an insured history.",
                "priority": "high",
            },
        ]
        maintain = []
        if smoker == 0:
            maintain.append({
                "title": "Non-smoker status",
                "detail": "Keep up your non-smoking status to avoid severe actuarial loading.",
                "priority": "medium",
            })
        if bmi < 25.0:
            maintain.append({
                "title": "Healthy BMI",
                "detail": f"Your BMI of {bmi:.1f} is optimal; ensure cardiovascular health is monitored.",
                "priority": "medium",
            })
        maintain.append({
            "title": "Frequent physician consultations",
            "detail": "Keep regular records with your primary physician to document risk management.",
            "priority": "medium",
        })
        riders = [
            "Critical Illness Rider (may carry loading)",
            "Accidental Cover (usually available)",
            "Avoid high-sum riders until risk improves",
        ]
        checklist = [
            "Undergo the recommended medical tests",
            "Start a supervised weight / smoking program",
            "Re-apply for reclassification after 12 months",
            "Consult a licensed insurance advisor",
        ]

    else:
        headline = "Could not classify — please re-check the inputs"
        actions = []
        maintain = []
        warnings = []
        riders = []
        checklist = [
            "Verify input fields",
            "Ensure all mandatory information is provided",
        ]

    return {
        "headline": headline,
        "actions": actions,
        "maintain": maintain,
        "warnings": warnings,
        "riders": riders,
        "checklist": checklist,
    }


def recommendations_to_text(rec: Dict[str, Any], features: Dict[str, Any]) -> str:
    """Format structured recommendations into a clean plain text report."""
    lines: List[str] = [
        "=" * 72,
        "HEALTHSURE ACTUARIAL UNDERWRITING & RECOMMENDATION REPORT",
        "=" * 72,
        "",
        "APPLICANT PROFILE:",
        f"  Age: {features.get('age', '—')} years",
        f"  Gender: {features.get('gender', '—')}",
        f"  BMI: {float(features.get('bmi', 0)):.1f} kg/m²",
        f"  Smoking Status: {'Smoker' if features.get('smoker', 0) == 1 else 'Non-Smoker'}",
        f"  Annual Income: {features.get('income_lpa', '—')} LPA",
        f"  City Tier: Tier {features.get('city_tier', '—')}",
        f"  Occupation: {features.get('occupation', '—')}",
        f"  Health Risk Index: {float(features.get('health_risk_index', 0)):.2f}",
        "",
        f"HEADLINE: {rec.get('headline', '')}",
        "",
    ]

    warnings = rec.get("warnings", [])
    if warnings:
        lines.append("⚠️ ATTENTION & RISK DRIVERS:")
        for w in warnings:
            lines.append(f"  • [{w['priority'].upper()}] {w['title']}")
            lines.append(f"    {w['detail']}")
        lines.append("")

    actions = rec.get("actions", [])
    if actions:
        lines.append("✅ RECOMMENDED ACTIONS:")
        for a in actions:
            lines.append(f"  • [{a['priority'].upper()}] {a['title']}")
            lines.append(f"    {a['detail']}")
        lines.append("")

    maintain = rec.get("maintain", [])
    if maintain:
        lines.append("🔄 MAINTAIN / POSITIVE HABITS:")
        for m in maintain:
            lines.append(f"  • [{m['priority'].upper()}] {m['title']}")
            lines.append(f"    {m['detail']}")
        lines.append("")

    riders = rec.get("riders", [])
    if riders:
        lines.append("💊 SUGGESTED COVERAGE RIDERS:")
        for r in riders:
            lines.append(f"  • {r}")
        lines.append("")

    checklist = rec.get("checklist", [])
    if checklist:
        lines.append("📝 ACTION CHECKLIST:")
        for item in checklist:
            lines.append(f"  [ ] {item}")
        lines.append("")

    lines.extend([
        "=" * 72,
        "HealthSure InsurTech Suite v2.0 • Deterministic Clinical Rule Engine",
        "=" * 72,
    ])

    return "\n".join(lines)
