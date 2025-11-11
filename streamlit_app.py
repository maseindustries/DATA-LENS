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
        # tab 3
    # -----------------------------
with tab3:
    st.header("Exploratory Data Analysis (EDA) — Charts queue & nicer layout")

    # ensure saved_charts exists
    if "saved_charts" not in st.session_state:
        st.session_state["saved_charts"] = []  # list of dicts {ds_key, ds_name, chart_type, params, caption, time}

    # Datasets and friendly names
    datasets = [
        ("cleaned_a", st.session_state.get("cleaned_a_name", "Dataset A")),
        ("cleaned_b", st.session_state.get("cleaned_b_name", "Dataset B"))
    ]

    # Build list of available datasets (only DataFrames)
    available = [(key, name) for key, name in datasets if isinstance(st.session_state.get(key), pd.DataFrame)]

    if not available:
        st.warning("Please upload & clean at least one dataset in Tabs 1–2 before running EDA.")
    else:
        # Layout: left = controls + chart, right = queue
        left, right = st.columns([3, 1])

        with left:
            st.subheader("Dataset selection & overview")
            display_names = [name for _, name in available]
            chosen_name = st.selectbox("Choose dataset for EDA", options=display_names, key="eda_choose_ds")

            # find corresponding key
            ds_key = next(key for key, name in available if name == chosen_name)
            df = st.session_state.get(ds_key)

            if df is None or not isinstance(df, pd.DataFrame):
                st.error(f"{chosen_name} is not available. Please return to Cleaning (Tab 2).")
            else:
                st.markdown(f"**{chosen_name}** — rows × cols: **{df.shape[0]} × {df.shape[1]}**")
                st.write("Columns:", list(df.columns))

                # column types
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

                # quick summaries
                with st.expander("Quick summaries (missing / numeric / categorical)"):
                    missing = df.isna().sum()
                    if missing.sum() > 0:
                        st.write("Missing values (non-zero only):")
                        st.dataframe(missing[missing > 0].sort_values(ascending=False))
                    else:
                        st.info("No missing values detected.")
                    if numeric_cols:
                        st.write("Numeric summary (describe):")
                        st.dataframe(df[numeric_cols].describe().T)
                    else:
                        st.info("No numeric columns found.")
                    if cat_cols:
                        st.write("Categorical columns (value counts):")
                        sel_cat = st.selectbox("Choose a categorical column to inspect", options=cat_cols, key=f"{ds_key}_cat_inspect")
                        vc = df[sel_cat].value_counts(dropna=False)
                        st.dataframe(vc)
                    else:
                        st.info("No categorical columns found.")

                # Row / column inspector
                st.subheader("Row / column inspector")
                col_to_view = st.selectbox("Pick a column to show first 10 values", options=list(df.columns), key=f"{ds_key}_inspector")
                st.dataframe(df[[col_to_view]].head(10))

                # Column summary download
                if st.button("Download column summary CSV", key=f"{ds_key}_download_summary"):
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

                # -----------------------
                # Chart controls (always visible)
                # -----------------------
                st.markdown("---")
                st.subheader("Charts (pick options then Show chart)")
                chart_options = [
                    "None",
                    "Histogram (single numeric)",
                    "Boxplot (single numeric)",
                    "Scatter (choose X and Y numeric)",
                    "Correlation heatmap (numeric columns)"
                ]
                chart_choice = st.selectbox("Choose a chart", options=chart_options, index=0, key=f"{ds_key}_chart_choice")

                # prepare params dict and controls
                chart_params = {}
                if chart_choice == "Histogram (single numeric)":
                    if not numeric_cols:
                        st.info("No numeric columns to plot.")
                    else:
                        x_col = st.selectbox("Numeric column (histogram)", options=numeric_cols, key=f"{ds_key}_hist_x")
                        bins = st.number_input("Bins", min_value=5, max_value=500, value=30, step=1, key=f"{ds_key}_hist_bins")
                        color_col = None
                        if cat_cols:
                            color_col = st.selectbox("Color by (optional categorical)", options=[None] + cat_cols, index=0, key=f"{ds_key}_hist_color")
                        chart_params.update({"x_col": x_col, "bins": bins, "color_col": color_col})

                elif chart_choice == "Boxplot (single numeric)":
                    if not numeric_cols:
                        st.info("No numeric columns to plot.")
                    else:
                        y_col = st.selectbox("Numeric column (boxplot)", options=numeric_cols, key=f"{ds_key}_box_y")
                        group_col = None
                        if cat_cols:
                            group_col = st.selectbox("Group by (optional categorical)", options=[None] + cat_cols, index=0, key=f"{ds_key}_box_group")
                        chart_params.update({"y_col": y_col, "group_col": group_col})

                elif chart_choice == "Scatter (choose X and Y numeric)":
                    if len(numeric_cols) < 2:
                        st.info("Need at least two numeric columns for a scatter plot.")
                    else:
                        x_col = st.selectbox("X axis (numeric)", options=numeric_cols, key=f"{ds_key}_scatter_x")
                        y_col = st.selectbox("Y axis (numeric)", options=[c for c in numeric_cols if c != x_col], key=f"{ds_key}_scatter_y")
                        color_col = None
                        if cat_cols:
                            color_col = st.selectbox("Color by (optional categorical)", options=[None] + cat_cols, index=0, key=f"{ds_key}_scatter_color")
                        chart_params.update({"x_col": x_col, "y_col": y_col, "color_col": color_col})

                elif chart_choice == "Correlation heatmap (numeric columns)":
                    if len(numeric_cols) < 2:
                        st.info("Need at least two numeric columns for correlation heatmap.")
                    else:
                        chart_params.update({})

                # Show chart / Save controls
                st.write("")  # spacer
                col_show, col_save = st.columns([1, 1])
                auto_queue = st.checkbox("Auto-queue displayed chart for PDF", value=False, key=f"{ds_key}_auto_queue")
                with col_show:
                    if st.button("Show chart", key=f"{ds_key}_show_chart"):
                        fig = None
                        try:
                            if chart_choice == "Histogram (single numeric)":
                                x_col = chart_params.get("x_col")
                                bins = chart_params.get("bins", 30)
                                color_col = chart_params.get("color_col")
                                if color_col:
                                    fig = px.histogram(df, x=x_col, color=color_col, nbins=bins)
                                else:
                                    fig = px.histogram(df, x=x_col, nbins=bins)

                            elif chart_choice == "Boxplot (single numeric)":
                                y_col = chart_params.get("y_col")
                                group_col = chart_params.get("group_col")
                                if group_col:
                                    fig = px.box(df, x=group_col, y=y_col)
                                else:
                                    fig = px.box(df, y=y_col)

                            elif chart_choice == "Scatter (choose X and Y numeric)":
                                x_col = chart_params.get("x_col")
                                y_col = chart_params.get("y_col")
                                color_col = chart_params.get("color_col")
                                if color_col:
                                    fig = px.scatter(df, x=x_col, y=y_col, color=df[color_col].astype(str))
                                else:
                                    fig = px.scatter(df, x=x_col, y=y_col)

                            elif chart_choice == "Correlation heatmap (numeric columns)":
                                corr = df[numeric_cols].corr()
                                fig = px.imshow(corr, text_auto=True)

                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                # If auto_queue is checked, immediately add to saved_charts (with empty caption)
                                if auto_queue:
                                    saved = {
                                        "ds_key": ds_key,
                                        "ds_name": chosen_name,
                                        "chart_type": chart_choice,
                                        "params": chart_params,
                                        "caption": "",
                                        "time": datetime.utcnow().isoformat()
                                    }
                                    st.session_state["saved_charts"].append(saved)
                                    st.success("Displayed chart auto-queued for PDF summary")
                            else:
                                st.info("Could not create chart with the selected options.")
                        except Exception as e:
                            st.error(f"Error rendering chart: {e}")

                with col_save:
                    caption = st.text_input("Optional caption for PDF", key=f"{ds_key}_chart_caption")
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

        # RIGHT column: show queue & controls
        with right:
            st.subheader("PDF Queue")
            queue = st.session_state.get("saved_charts", [])
            if not queue:
                st.info("No charts queued yet.")
            else:
                st.write(f"Charts queued: {len(queue)}")
                for i, c in enumerate(queue, 1):
                    st.write(f"**{i}. {c['ds_name']}** — {c['chart_type']}")
                    if c.get("caption"):
                        st.caption(c["caption"])

           if queue:
    if st.button("Clear PDF queue"):
        st.session_state["saved_charts"] = []
        st.success("PDF queue cleared")

