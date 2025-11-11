import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
from fpdf import FPDF
from datetime import datetime
import os

st.set_page_config(layout="wide")
st.title("DataLens")

# -----------------------------
# Session state placeholders
# -----------------------------
for key in [
    "cleaned_a", "cleaned_b", "cleaned_a_saved", "cleaned_b_saved",
    "cleaned_a_name", "cleaned_b_name", "cleaned_a_operations", "cleaned_b_operations",
    "compare_report"
]:
    if key not in st.session_state:
        st.session_state[key] = None
# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Upload", "Cleaning", "EDA", "Compare & Contrast", "Export", "PDF Summary"
])
# -----------------------------
# Tab 1: Upload
# -----------------------------
with tab1:
    st.header("Upload Datasets")

    uploaded_file_a = st.file_uploader("Upload Dataset A", type=["csv", "xlsx"], key="upload_a")
    uploaded_file_b = st.file_uploader("Upload Dataset B", type=["csv", "xlsx"], key="upload_b")

    if uploaded_file_a or uploaded_file_b:
        if st.button("Confirm Upload"):
            if uploaded_file_a is not None:
                if uploaded_file_a.name.endswith(".csv"):
                    st.session_state.cleaned_a = pd.read_csv(uploaded_file_a)
                else:
                    st.session_state.cleaned_a = pd.read_excel(uploaded_file_a)
                st.session_state.cleaned_a_name = uploaded_file_a.name
                st.session_state.cleaned_a_saved = None
                st.session_state.cleaned_a_operations = []

            if uploaded_file_b is not None:
                if uploaded_file_b.name.endswith(".csv"):
                    st.session_state.cleaned_b = pd.read_csv(uploaded_file_b)
                else:
                    st.session_state.cleaned_b = pd.read_excel(uploaded_file_b)
                st.session_state.cleaned_b_name = uploaded_file_b.name
                st.session_state.cleaned_b_saved = None
                st.session_state.cleaned_b_operations = []

            st.success("Files uploaded successfully!")

    # Optional: Show a preview of uploaded files
    if st.session_state.cleaned_a is not None:
        st.write(f"Preview of {st.session_state.cleaned_a_name}")
        st.dataframe(st.session_state.cleaned_a.head())

    if st.session_state.cleaned_b is not None:
        st.write(f"Preview of {st.session_state.cleaned_b_name}")
        st.dataframe(st.session_state.cleaned_b.head())         
# -----------------------------
# Tab 2: Cleaning
# -----------------------------
with tab2:
    st.header("Data Cleaning")

    # -----------------------------
    # Safe cleaning function
    # -----------------------------
    def clean_dataset(df, label, cleaning_options):
        # HARD safety check
        if df is None or not isinstance(df, pd.DataFrame):
            st.info(f"{label} is missing or invalid. Skipping cleaning.")
            return None, []

        # Debug
        st.write(f"DEBUG: Cleaning {label}, df type = {type(df)}")

        # Safe column detection
        numeric_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(include=['object']).columns

        applied_ops = []
        original_shape = df.shape

        # -----------------------------
        # Cleaning operations
        # -----------------------------
        if "Drop duplicate rows" in cleaning_options:
            before = len(df)
            df = df.drop_duplicates()
            if len(df) < before:
                applied_ops.append("Duplicate rows removed")

        if "Fill missing numeric values with median" in cleaning_options:
            for col in numeric_cols:
                na_count = df[col].isna().sum()
                if na_count > 0:
                    df[col].fillna(df[col].median(), inplace=True)
                    applied_ops.append(f"Filled {na_count} missing numeric values in {col}")

        if "Fill missing categorical values with mode" in cleaning_options:
            for col in cat_cols:
                na_count = df[col].isna().sum()
                if na_count > 0 and not df[col].mode().empty:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    applied_ops.append(f"Filled {na_count} missing categorical values in {col}")

        if "Trim whitespace from string columns" in cleaning_options:
            for col in cat_cols:
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            applied_ops.append("Trimmed whitespace from string columns")

        if "Remove columns with all nulls" in cleaning_options:
            all_null_cols = df.columns[df.isna().all()].tolist()
            if all_null_cols:
                df.drop(columns=all_null_cols, inplace=True)
                applied_ops.append(f"Removed {len(all_null_cols)} columns with all nulls")

        return df, applied_ops

    # -----------------------------
    # Datasets
    # -----------------------------
    datasets = [
        ("cleaned_a", "Dataset A", st.session_state.get('cleaned_a_name', 'Dataset A')),
        ("cleaned_b", "Dataset B", st.session_state.get('cleaned_b_name', 'Dataset B'))
    ]

    # -----------------------------
    # Check at least one dataset exists
    # -----------------------------
    if all(st.session_state.get(ds_name) is None for ds_name, _, _ in datasets):
        st.warning("Please upload at least one dataset in Tab 1 before cleaning.")
    else:
        # Preview datasets
        for ds_name, label, _ in datasets:
            df = st.session_state.get(ds_name)
            if isinstance(df, pd.DataFrame):
                st.subheader(f"{label} Preview")
                st.dataframe(df.head())
            else:
                st.info(f"{label} is not uploaded or invalid. Skipping preview.")

        # -----------------------------
        # Cleaning options
        # -----------------------------
        cleaning_options = st.multiselect(
            "Select cleaning operations to apply",
            [
                "Drop duplicate rows",
                "Fill missing numeric values with median",
                "Fill missing categorical values with mode",
                "Trim whitespace from string columns",
                "Remove columns with all nulls"
            ],
            default=["Drop duplicate rows"]
        )

        # Custom dataset names
        custom_names = {
            "cleaned_a": st.text_input("Name Dataset A (optional)", value=st.session_state.get('cleaned_a_name', 'Dataset A')),
            "cleaned_b": st.text_input("Name Dataset B (optional)", value=st.session_state.get('cleaned_b_name', 'Dataset B'))
        }

        # -----------------------------
        # Run cleaning
        # -----------------------------
        if st.button("Run Cleaning"):
            for ds_name, label, _ in datasets:
                df = st.session_state.get(ds_name)
                cleaned_df, ops = clean_dataset(df, label, cleaning_options)

                if cleaned_df is not None:
                    # Save to session state
                    st.session_state[ds_name] = cleaned_df
                    st.session_state[f"{ds_name}_name"] = custom_names[ds_name]
                    st.session_state[f"{ds_name}_operations"] = ops
                    st.session_state[f"{ds_name}_saved"] = True

                    # Display summary
                    new_shape = cleaned_df.shape
                    st.subheader(f"{label} Cleaning Summary")
                    if ops:
                        st.write("**Changes applied:**")
                        for op in ops:
                            st.markdown(f"- {op}")
                    else:
                        st.info("No changes were necessary based on selected options.")
                    st.write(f"**Original shape:** {df.shape}, **New shape:** {new_shape}")
                    st.dataframe(cleaned_df.head())
                else:
                    st.info(f"{label} skipped.")

        # -----------------------------
        # Optional debug
        # -----------------------------
        st.write("DEBUG: Session state types after cleaning")
        st.write(f"cleaned_a type = {type(st.session_state.get('cleaned_a'))}")
        st.write(f"cleaned_b type = {type(st.session_state.get('cleaned_b'))}")
     
        # -----------------------------
        # ---------- Tab 3: EDA (with chart-save for PDF) ----------
with tab3:
    st.header("Exploratory Data Analysis (EDA) — compact view")

    # ensure saved_charts exists
    if "saved_charts" not in st.session_state:
        st.session_state["saved_charts"] = []

    # Datasets and friendly names
    datasets = [
        ("cleaned_a", st.session_state.get("cleaned_a_name", "Dataset A")),
        ("cleaned_b", st.session_state.get("cleaned_b_name", "Dataset B"))
    ]

    # Available datasets only
    available = [(key, name) for key, name in datasets if isinstance(st.session_state.get(key), pd.DataFrame)]

    if not available:
        st.warning("Please upload & clean at least one dataset in Tabs 1–2 before running EDA.")
    else:
        # Choose dataset by friendly name
        display_names = [name for _, name in available]
        chosen_name = st.selectbox("Choose dataset for EDA", options=display_names)

        ds_key = next(key for key, name in available if name == chosen_name)
        df = st.session_state.get(ds_key)

        if df is None or not isinstance(df, pd.DataFrame):
            st.error(f"{chosen_name} is not available. Please return to Cleaning (Tab 2).")
            st.stop()

        # ---------- Top overview row ----------
        st.subheader(f"{chosen_name} — overview")
        col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.2], gap="small")
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", f"{df.shape[1]:,}")
        with col3:
            n_missing = int(df.isna().sum().sum())
            st.metric("Total missing", f"{n_missing:,}")
        with col4:
            n_unique_cols = sum(1 for c in df.columns if df[c].nunique(dropna=True) < df.shape[0])
            st.metric("Cols w/ unique values", f"{n_unique_cols}")

        st.markdown("---")

        # ---------- Two-column layout: left = details, right = charts & actions ----------
        left, right = st.columns([2, 1], gap="large")

        # LEFT: details as expanders
        with left:
            # Missing values (expander)
            with st.expander("Missing values (click to expand)", expanded=False):
                missing = df.isna().sum()
                miss_nonzero = missing[missing > 0].sort_values(ascending=False)
                if miss_nonzero.empty:
                    st.success("No missing values detected.")
                else:
                    st.dataframe(miss_nonzero.to_frame("n_missing"))

            # Numeric summary (expander)
            with st.expander("Numeric summary", expanded=False):
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                if numeric_cols:
                    desc = df[numeric_cols].describe().T
                    st.dataframe(desc)
                    # small quick stats row
                    min_col = desc["min"].idxmin() if "min" in desc else None
                    max_col = desc["max"].idxmax() if "max" in desc else None
                    st.write(f"Smallest min value column: **{min_col}** — Largest max value column: **{max_col}**")
                else:
                    st.info("No numeric columns available.")

            # Categorical summary (expander)
            with st.expander("Categorical summary", expanded=False):
                cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
                if cat_cols:
                    sel_cat = st.selectbox("Choose a categorical column to inspect", options=cat_cols, key=f"{ds_key}_cat_select")
                    vc = df[sel_cat].value_counts(dropna=False)
                    st.dataframe(vc)
                    if vc.shape[0] <= 50:
                        fig_cat = px.bar(x=vc.index.astype(str), y=vc.values)
                        st.plotly_chart(fig_cat, use_container_width=True)
                    else:
                        st.info("Too many categories to plot; showing table only.")
                else:
                    st.info("No categorical columns.")

            # Row/column inspector (expander)
            with st.expander("Row / column inspector", expanded=False):
                col_to_view = st.selectbox("Pick a column to show first 10 values", options=list(df.columns), key=f"{ds_key}_inspector")
                st.table(df[[col_to_view]].head(10))

            # Download column summary (compact button)
            with st.expander("Download column summary", expanded=False):
                summary = []
                for col in df.columns:
                    summary.append({
                        "column": col,
                        "dtype": str(df[col].dtype),
                        "n_unique": int(df[col].nunique(dropna=True)),
                        "n_missing": int(df[col].isna().sum())
                    })
                summary_df = pd.DataFrame(summary)
                tolink = io.BytesIO()
                summary_df.to_csv(tolink, index=False)
                tolink.seek(0)
                st.download_button("Download summary.csv", data=tolink, file_name=f"{ds_key}_summary.csv")

        # RIGHT: chart controls + saved charts
        with right:
            st.subheader("Charts")
            # chart selector
            chart_options = [
                "None",
                "Histogram (single numeric)",
                "Boxplot (single numeric)",
                "Scatter (X vs Y numeric)",
                "Correlation heatmap"
            ]
            chart_choice = st.selectbox("Choose chart", options=chart_options, index=0, key=f"{ds_key}_chart_choice")

            # load previous chart settings
            last_key = f"{ds_key}_last_chart"
            last_settings = st.session_state.get(last_key, {})

            # chart parameter controls (compact)
            chart_params = {}
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            if chart_choice == "Histogram (single numeric)":
                if numeric_cols:
                    default = last_settings.get("x_col", numeric_cols[0])
                    x_col = st.selectbox("Numeric column", options=numeric_cols, index=numeric_cols.index(default) if default in numeric_cols else 0, key=f"{ds_key}_hist_col")
                    bins = st.slider("Bins", min_value=5, max_value=200, value=last_settings.get("bins", 30), key=f"{ds_key}_bins")
                    color_col = None
                    if cat_cols:
                        color_col = st.selectbox("Color by (optional)", options=[None] + cat_cols, key=f"{ds_key}_hist_color")
                    chart_params.update({"x_col": x_col, "bins": bins, "color_col": color_col})
                else:
                    st.info("No numeric columns available.")

            elif chart_choice == "Boxplot (single numeric)":
                if numeric_cols:
                    y_col = st.selectbox("Numeric column", options=numeric_cols, key=f"{ds_key}_box_col")
                    group_col = None
                    if cat_cols:
                        group_col = st.selectbox("Group by (optional)", options=[None] + cat_cols, key=f"{ds_key}_box_group")
                    chart_params.update({"y_col": y_col, "group_col": group_col})
                else:
                    st.info("No numeric columns available.")

            elif chart_choice == "Scatter (X vs Y numeric)":
                if len(numeric_cols) >= 2:
                    x_col = st.selectbox("X axis", options=numeric_cols, key=f"{ds_key}_scatter_x")
                    y_col = st.selectbox("Y axis", options=[c for c in numeric_cols if c != x_col], key=f"{ds_key}_scatter_y")
                    color_col = None
                    if cat_cols:
                        color_col = st.selectbox("Color by (optional)", options=[None] + cat_cols, key=f"{ds_key}_scatter_color")
                    chart_params.update({"x_col": x_col, "y_col": y_col, "color_col": color_col})
                else:
                    st.info("Need 2+ numeric columns.")

            elif chart_choice == "Correlation heatmap":
                if len(numeric_cols) < 2:
                    st.info("Need 2+ numeric columns.")
                else:
                    pass  # no extra params

            # Show chart button
            if chart_choice != "None":
                if st.button("Show chart", key=f"{ds_key}_show_chart"):
                    fig = None
                    try:
                        if chart_choice == "Histogram (single numeric)":
                            x_col = chart_params.get("x_col")
                            bins = chart_params.get("bins", 30)
                            color_col = chart_params.get("color_col")
                            fig = px.histogram(df, x=x_col, color=color_col if color_col else None, nbins=bins)

                        elif chart_choice == "Boxplot (single numeric)":
                            y_col = chart_params.get("y_col")
                            group_col = chart_params.get("group_col")
                            fig = px.box(df, x=group_col if group_col else None, y=y_col)

                        elif chart_choice == "Scatter (X vs Y numeric)":
                            x_col = chart_params.get("x_col")
                            y_col = chart_params.get("y_col")
                            color_col = chart_params.get("color_col")
                            fig = px.scatter(df, x=x_col, y=y_col, color=df[color_col].astype(str) if color_col else None)

                        elif chart_choice == "Correlation heatmap":
                            corr = df.select_dtypes(include=["number"]).corr()
                            fig = px.imshow(corr, text_auto=True)

                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Could not render chart with these settings.")
                    except Exception as e:
                        st.error(f"Chart error: {e}")

                    # save last settings
                    st.session_state[last_key] = {"chart_choice": chart_choice, **chart_params}

                    # Save chart to queue for PDF
                    st.markdown("**Save this chart for the PDF summary**")
                    caption = st.text_input("Caption (optional)", value=last_settings.get("caption", ""), key=f"{ds_key}_caption")
                    if st.button("Save chart to PDF", key=f"{ds_key}_save_chart"):
                        saved = {
                            "ds_key": ds_key,
                            "ds_name": chosen_name,
                            "chart_type": chart_choice,
                            "params": chart_params,
                            "caption": caption,
                            "time": datetime.utcnow().isoformat()
                        }
                        st.session_state["saved_charts"].append(saved)
                        st.success("Chart saved for PDF summary")

            # Saved charts list (compact)
            st.markdown("---")
            st.write("Charts queued for PDF")
            if st.session_state["saved_charts"]:
                for i, c in enumerate(st.session_state["saved_charts"], 1):
                    st.write(f"{i}. **{c['ds_name']}** — {c['chart_type']} — {c.get('caption','(no caption)')}")
            else:
                st.info("No charts queued yet.")
