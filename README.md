# 🏥 HealthSure: Actuarial InsurTech & Clinical Intelligence Platform

An end-to-end Machine Learning and Healthcare Analytics web application designed for actuarial risk classification, cohort data exploration, and longitudinal patient clinical profile evaluation. Built with pure **scikit-learn** and **Streamlit**.

> **No Gemini / OpenAI / external API keys required.**  
> HealthSure runs entirely on local, deterministic machine learning models and clinical rule engines. All processing happens in-memory with zero external API calls or network latency.

---

## 📂 Project Structure

```text
HealthSure/
├── app.py                         # Application entrypoint & module navigation hub
├── Dockerfile                     # Production container specification (Python 3.11-slim)
├── requirements.txt               # Pinned lightweight dependency manifest
├── README.md                      # Documentation & deployment guidelines
├── core/                          # Modular domain logic & utilities
│   ├── __init__.py                # Package initializer
│   ├── theme.py                   # Dark glassmorphic design system tokens & CSS injection
│   ├── ui.py                      # Reusable UI cards, KPIs, badges, and Plotly templates
│   ├── data_io.py                 # File reading, column normalization, and session persistence
│   ├── model.py                   # Feature engineering & scikit-learn Pipeline inference
│   ├── patient_io.py              # Multi-file ZIP/CSV clinical record ingestion & role detection
│   ├── patient_clinical.py        # BMI, AHA blood pressure, Charlson comorbidity & narrative engine
│   ├── patient_pdf.py             # Comprehensive ReportLab multi-page clinical PDF report generator
│   ├── patient_ui.py              # Clinical summary cards, reference band charts & timelines
│   └── recommendations.py         # Deterministic rule-based underwriting recommendations engine
├── models/
│   └── insurance_premium_model.pkl # Trained Random Forest underwriting pipeline
└── pages/
    ├── 1_Insurance_Prediction.py  # Module 1: Single applicant & batch underwriting
    ├── 2_Dataset_Explorer.py      # Module 2: Generic dataset visualizer & cohort filter
    └── 3_Patient_Deep_Dive.py     # Module 3: Longitudinal multi-table clinical dossier
```

---

## 🔍 The 3 Modules Explained

### 1. 📊 Insurance Risk Prediction (`pages/1_Insurance_Prediction.py`)
Predicts actuarial premium risk categories (**Low**, **Medium**, **High**) utilizing a trained scikit-learn Random Forest classification pipeline.
* **Single Applicant Underwriting**: Interactive form capturing applicant age, physical dimensions (height, weight, live BMI readout), annual income (LPA), occupation, city tier, and smoking status. Delivers predicted risk badge, class probability distribution bar chart, calculated actuarial features, and deterministic **📋 Tailored Underwriting Recommendations** with plain text and CSV downloads.
* **Batch Underwriting**: Upload batch applicant files (`.csv`, `.xlsx`, `.xls`). Performs automatic synonym normalization, data type coercion, missingness profiling, and vectorized pipeline scoring with interactive class probability progress bars, cohort risk distribution donut chart, per-row recommendation expander, and batch recommendations CSV export.

### 2. 🔎 Dataset Explorer (`pages/2_Dataset_Explorer.py`)
Generic, schema-agnostic tabular data analysis and visual discovery tool for arbitrary datasets.
* **Dynamic Sidebar Filters**: Automatically identifies column roles (numeric, categorical, datetime) and generates real-time sliders and multiselect controls.
* **Dynamic Chart Builder**: Interactive creation of Bar, Line, Scatter, Pie, Box, and Histogram charts rendered with unified dark glassmorphism styling.
* **Statistical Summaries & Correlation**: Includes frequency distribution tables, summary statistics (`describe()`), and Pearson correlation heatmaps.

### 3. 🩺 Single Patient Deep Dive (`pages/3_Patient_Deep_Dive.py`)
Multi-table clinical record ingestion and longitudinal dossier analyzer with comprehensive PDF export.
* **Multi-Format Ingestion**: Ingests zipped archives of CSVs, batches of independent spreadsheets, or single long-format tables with automated heuristic role mapping (`patient_info`, `vitals`, `labs`, `diagnoses`, `medications`, `encounters`).
* **5 Interactive Clinical Views**:
  * **Overview & Comorbidities**: Patient banner card, rule-based clinical narrative paragraph, Charlson comorbidity index, and cardiovascular risk factor tags.
  * **Vitals Time Series**: Longitudinal systolic/diastolic blood pressure curves with AHA clinical reference bands and heart rate tracking.
  * **Laboratory Trends**: Longitudinal lab test trajectories (HbA1c, fasting glucose, lipid panel, creatinine) with normative reference bounds and status flags.
  * **Diagnoses & Medications**: Interactive diagnosis chronological timeline and active pharmaceutical regimen categorizations.
  * **Encounters & Visits**: Length-of-stay tracking, admission types, and healthcare cost analysis.
* **📄 Downloadable Clinical Report (ReportLab PDF)**:
  * Generates an executive, publication-grade clinical summary report capturing the entire Deep Dive UI across all 5 dimensions.
  * Includes running headers/footers with dynamic two-pass page numbering ("Page X of Y"), medical in-confidence notice, patient demographics card, clinical narrative synthesis, cardiovascular/comorbidity risk stratification, longitudinal vitals table, laboratory analyte panel with reference intervals, diagnoses and ICD-10 codings, pharmacological medications table, and hospital encounter/utilization metrics.
  * Accessible directly from the top action bar, the sidebar dataset panel, and the Overview dashboard card.

---

## 🚀 How to Run Locally

### Prerequisites
* Python 3.11 or Python 3.12
* `pip` package manager

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Application
```bash
streamlit run app.py
```
The application will launch and be accessible at `http://localhost:8501`.

---

## 🌐 Deploy to Render

To deploy HealthSure on [Render](https://render.com) as a Web Service:

1. **Environment**: Python 3
2. **Build Command**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Start Command**:
   ```bash
   streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

*No Gemini / OpenAI / external API keys required.* The application starts immediately without configuring external secrets or API credentials.

---

## 🐳 Docker Deployment

To build and run HealthSure inside a container:

```bash
# Build the container image
docker build -t healthsure:latest .

# Run the container
docker run -p 8501:8501 healthsure:latest
```

Open your browser at `http://localhost:8501`.

---

## 🛡️ Actuarial & Machine Learning Architecture

* **Model Framework**: Pure `scikit-learn` Pipeline (StandardScaler, OneHotEncoder, RandomForestClassifier).
* **Caching Strategy**: `@st.cache_resource` for zero-overhead model deserialization; `@st.cache_data` for uploaded file parsing and clinical narrative synthesis.
* **Privacy & Compliance**: Zero patient data egress. All computations occur within the runtime environment.
