"""Dataset Explorer page for HealthSure: dynamic charting, dataset analytics, and cohort filtering."""
import io
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from core.theme import THEME, apply_theme
from core.ui import (
    page_header,
    section,
    empty_state,
    download_button,
    get_plotly_layout,
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

# 1. Page Configuration
st.set_page_config(
    page_title="Dataset Explorer | HealthSure",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Theme
apply_theme()

# 3. Retrieve or Ingest Dataset
df = get_dataset()
dataset_name = st.session_state.get("dataset_name", "user_dataset")

# If no dataset exists in session, display upload dropzone & empty state
if df is None or df.empty:
    page_header(
        title="Dataset Explorer",
        subtitle="Upload any tabular dataset to perform visual exploration, distribution analytics, and dynamic filtering.",
    )

    uploaded_file = st.file_uploader(
        "Upload dataset (.csv, .xlsx, .xls)",
        type=["csv", "xlsx", "xls"],
        key="explorer_file_uploader",
        help="Upload CSV or Excel files. Automatic column normalization will be applied.",
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Processing dataset..."):
                raw_df = read_uploaded_file(uploaded_file)
                norm_df = normalize_columns(raw_df)
                coerced_df = coerce_types(norm_df)
                store_dataset(coerced_df, name=uploaded_file.name)
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to load dataset: {exc}")
    else:
        empty_state(
            icon="🔎",
            title="No Dataset in Session",
            message="Please upload a CSV or Excel dataset above to unlock interactive visualization tools, custom charting, and multi-variable cohort filtering.",
        )
    sidebar_footer()
    st.stop()

# ==============================================================================
# DATASET EXISTS: RENDER EXPLORER
# ==============================================================================

# Header with Dataset Metadata
page_header(
    title="Dataset Explorer",
    subtitle=f"Active Dataset: {dataset_name} • {len(df):,} rows × {len(df.columns)} columns",
)

# ------------------------------------------------------------------------------
# LEFT SIDEBAR: FILTERS
# ------------------------------------------------------------------------------
with st.sidebar:
    section("Filters", "Refine cohort records")

    col_btn_reset, col_btn_clear = st.columns(2)
    with col_btn_reset:
        if st.button("Reset filters", key="btn_reset_filters", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k.startswith("flt_"):
                    del st.session_state[k]
            st.rerun()
    with col_btn_clear:
        if st.button("Clear data", key="btn_clear_dataset", use_container_width=True):
            reset_dataset()
            for k in list(st.session_state.keys()):
                if k.startswith("flt_"):
                    del st.session_state[k]
            st.rerun()

    render_html("<div style='height: 0.5rem;'></div>")

    filtered_df = df.copy()

    # Iterate every column and render appropriate filter widget
    for col in df.columns:
        series = df[col]

        # 1. Numeric Filter (Sliders)
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            valid_series = series.dropna()
            if valid_series.empty:
                continue

            c_min = float(valid_series.min())
            c_max = float(valid_series.max())

            # Skip single-value or degenerate numeric columns
            if c_min >= c_max:
                continue

            # Integer vs Float slider
            if pd.api.types.is_integer_dtype(series):
                slider_val = st.slider(
                    col,
                    min_value=int(c_min),
                    max_value=int(c_max),
                    value=(int(c_min), int(c_max)),
                    step=1,
                    key=f"flt_{col}",
                )
            else:
                slider_val = st.slider(
                    col,
                    min_value=c_min,
                    max_value=c_max,
                    value=(c_min, c_max),
                    key=f"flt_{col}",
                )

            filtered_df = filtered_df[
                (filtered_df[col] >= slider_val[0]) & (filtered_df[col] <= slider_val[1])
            ]

        # 2. Datetime Filter (Date Range)
        elif pd.api.types.is_datetime64_any_dtype(series):
            valid_dt = series.dropna()
            if valid_dt.empty:
                continue

            min_dt = valid_dt.min().date()
            max_dt = valid_dt.max().date()

            if min_dt == max_dt:
                continue

            date_range = st.date_input(
                col,
                value=(min_dt, max_dt),
                min_value=min_dt,
                max_value=max_dt,
                key=f"flt_{col}",
            )

            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                filtered_df = filtered_df[
                    (filtered_df[col].dt.date >= date_range[0])
                    & (filtered_df[col].dt.date <= date_range[1])
                ]

        # 3. Categorical Filter (Multiselect for nunique <= 30)
        else:
            nunique = series.nunique(dropna=True)
            if 1 < nunique <= 30:
                unique_vals = sorted([str(x) for x in series.dropna().unique()])
                selected = st.multiselect(
                    col,
                    options=unique_vals,
                    default=unique_vals,
                    key=f"flt_{col}",
                )

                if selected:
                    filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)]
                else:
                    filtered_df = filtered_df.iloc[0:0]

    # Store filtered DataFrame in session state
    st.session_state["filtered_df"] = filtered_df

    sidebar_footer()

# ------------------------------------------------------------------------------
# MAIN AREA
# ------------------------------------------------------------------------------

# Empty filter state check
if filtered_df.empty:
    empty_state(
        icon="🔍",
        title="No Matching Records",
        message="The current sidebar filter criteria returned 0 records. Adjust the sliders or click 'Reset filters' to restore records.",
    )
    st.stop()

# Large Dataset Safeguard (>50k rows)
plot_df = filtered_df
if len(filtered_df) > 50000:
    st.warning(
        "⚠️ Large dataset detected (>50,000 records). Rendering a representative sample of 50,000 rows for smooth chart interactions."
    )
    plot_df = filtered_df.sample(n=50000, random_state=42)

# Build a Chart Section
section("Build a chart", "Construct dynamic visual analytics from active dataset columns.")

cols_list = list(df.columns)
c1, c2, c3, c4 = st.columns(4)

with c1:
    chart_type = st.selectbox(
        "Chart Type",
        options=["Bar", "Line", "Scatter", "Pie", "Box", "Histogram"],
        index=0,
        help="Select chart geometry",
    )

with c2:
    x_column = st.selectbox(
        "X Axis Column",
        options=cols_list,
        index=0,
        help="Select primary independent feature",
    )

with c3:
    is_y_disabled = chart_type in ("Pie", "Histogram")
    default_y_idx = min(1, len(cols_list) - 1)
    y_column = st.selectbox(
        "Y Axis Column",
        options=cols_list,
        index=default_y_idx,
        disabled=is_y_disabled,
        help="Select dependent metric (disabled for Pie and Histogram)",
    )

with c4:
    color_options = ["None"] + cols_list
    color_column_choice = st.selectbox(
        "Color Column (Optional)",
        options=color_options,
        index=0,
        help="Group or differentiate traces by color category",
    )
    color_col = None if color_column_choice == "None" else color_column_choice

# Plotly Palette
palette = [
    THEME["ACCENT"],
    THEME["INDIGO"],
    THEME["LOW"],
    THEME["MED"],
    THEME["HIGH"],
    "#ec4899",
    "#8b5cf6",
    "#06b6d4",
]

# Generate Selected Plotly Chart
try:
    if chart_type == "Bar":
        if pd.api.types.is_numeric_dtype(plot_df[y_column]):
            fig = px.bar(
                plot_df,
                x=x_column,
                y=y_column,
                color=color_col,
                color_discrete_sequence=palette,
            )
        else:
            agg_cols = [x_column] + ([color_col] if color_col else [])
            agg_df = plot_df.groupby(agg_cols).size().reset_index(name="count")
            fig = px.bar(
                agg_df,
                x=x_column,
                y="count",
                color=color_col,
                color_discrete_sequence=palette,
            )

    elif chart_type == "Line":
        fig = px.line(
            plot_df,
            x=x_column,
            y=y_column,
            color=color_col,
            color_discrete_sequence=palette,
        )

    elif chart_type == "Scatter":
        fig = px.scatter(
            plot_df,
            x=x_column,
            y=y_column,
            color=color_col,
            color_discrete_sequence=palette,
        )

    elif chart_type == "Pie":
        pie_counts = plot_df[x_column].value_counts().reset_index()
        pie_counts.columns = [x_column, "count"]
        fig = px.pie(
            pie_counts,
            names=x_column,
            values="count",
            color_discrete_sequence=palette,
            hole=0.45,
        )

    elif chart_type == "Box":
        fig = px.box(
            plot_df,
            x=x_column,
            y=y_column,
            color=color_col,
            color_discrete_sequence=palette,
        )

    elif chart_type == "Histogram":
        fig = px.histogram(
            plot_df,
            x=x_column,
            color=color_col,
            color_discrete_sequence=palette,
        )
    else:
        fig = px.histogram(plot_df, x=x_column, color_discrete_sequence=palette)

    layout = get_plotly_layout()
    chart_title = (
        f"<b>{chart_type} Chart: {x_column}</b>"
        if is_y_disabled
        else f"<b>{chart_type} Chart: {x_column} vs {y_column}</b>"
    )
    layout.update({
        "title": {
            "text": chart_title,
            "font": {"family": "Plus Jakarta Sans, sans-serif", "size": 15, "color": THEME["TEXT"]},
        },
        "margin": dict(l=40, r=30, t=50, b=40),
        "height": 450,
    })
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

except Exception as chart_err:
    st.error(f"Could not render chart: {chart_err}")

# Status strip below chart
pct_active = (len(filtered_df) / len(df)) * 100
render_html(
    f"""
    <div style="font-size: 0.85rem; color: {THEME['MUTED']}; margin: 0.5rem 0 1.25rem 0;">
        Active filter view: <b style="color: {THEME['TEXT']};">{len(filtered_df):,}</b> of
        <b style="color: {THEME['TEXT']};">{len(df):,}</b> records ({pct_active:.1f}% of cohort).
    </div>
    """
)

# Expanders: Value Counts & Summary Statistics
with st.expander(f"📊 Value counts: {x_column}", expanded=False):
    vc_df = filtered_df[x_column].value_counts(dropna=False).reset_index()
    vc_df.columns = [str(x_column), "count"]
    vc_df["percentage"] = (vc_df["count"] / len(filtered_df)) * 100
    st.dataframe(
        vc_df,
        column_config={
            "percentage": st.column_config.ProgressColumn(
                "Share (%)",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            )
        },
        use_container_width=True,
    )

with st.expander("📈 Summary statistics (Numerical & Categorical)", expanded=False):
    st.dataframe(filtered_df.describe(include="all"), use_container_width=True)

# Bottom Download Button
render_html("<div style='height: 1rem;'></div>")
download_button(
    df=filtered_df,
    filename=f"filtered_{dataset_name}.csv" if not dataset_name.endswith(".csv") else f"filtered_{dataset_name}",
    label="📥 Download Filtered Dataset as CSV",
    key="download_filtered_dataset_btn",
    filetype="csv",
)
