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
    st.header("Exploratory Data Analysis (EDA) — Charts savable for PDF")

    # ensure saved_charts exists
    if "saved_charts" not in st.session_state:
        st.session_state["saved_charts"] = []  # list of dicts {ds_key, ds_name, type, params..., caption, timestamp}

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
        # Map display names -> keys so user sees saved names
        display_names = [name for _, name in available]
        chosen_name = st.selectbox("Choose dataset for EDA", options=display_names)

        # find corresponding key
        ds_key = next(key for key, name in available if name == chosen_name)
        df = st.session_state.get(ds_key)

        # Extra safety
        if df is None or not isinstance(df, pd.DataFrame):
            st.error(f"{chosen_name} is not available. Please return to Cleaning (Tab 2).")
        else:
            st.subheader(f"{chosen_name} — quick overview")
            st.write(f"Rows × Columns: **{df.shape[0]} × {df.shape[1]}**")
            st.write("Columns:", list(df.columns))

            # Missing values
            missing = df.isna().sum()
            if missing.sum() > 0:
                st.subheader("Missing values")
                st.dataframe(missing[missing > 0].sort_values(ascending=False))
            else:
                st.info("No missing values detected.")

            # Column type separation (safe)
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            # Numeric summary
            if numeric_cols:
                st.subheader("Numeric summary")
                st.dataframe(df[numeric_cols].describe().T)
            else:
                st.info("No numeric columns found.")

            # Categorical summary + inspector
            if cat_cols:
                st.subheader("Categorical columns (value counts)")
                sel_cat = st.selectbox("Choose a categorical column to inspect", options=cat_cols)
                vc = df[sel_cat].value_counts(dropna=False)
                st.dataframe(vc)
                if vc.shape[0] <= 50:
                    fig_cat = px.bar(x=vc.index.astype(str), y=vc.values)
                    st.plotly_chart(fig_cat, use_container_width=True)
                else:
                    st.info("Too many unique categories to plot; showing table only.")
            else:
                st.info("No categorical columns found.")

            # Row / column inspector
            st.subheader("Row / column inspector")
            col_to_view = st.selectbox("Pick a column to show first 10 values", options=list(df.columns))
            st.write(df[[col_to_view]].head(10))

            # Column summary download
            if st.button("Download column summary CSV"):
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
            # Charts: choose then show (with options)
            # -----------------------
            st.subheader("Charts (choose then click Show chart)")
            chart_options = [
                "None",
                "Histogram (single numeric)",
                "Boxplot (single numeric)",
                "Scatter (choose X and Y numeric)",
                "Correlation heatmap (numeric columns)"
            ]
            chart_choice = st.selectbox("Choose a chart", options=chart_options, index=0)

            # load previous chart settings for this dataset if present
            last_key = f"{ds_key}_last_chart"
            last_settings = st.session_state.get(last_key, {})

            # chart-specific controls (only show relevant controls)
            chart_params = {}
            if chart_choice == "Histogram (single numeric)":
                if not numeric_cols:
                    st.info("No numeric columns to plot.")
                else:
                    default_num = last_settings.get("x_col", numeric_cols[0])
                    x_col = st.selectbox("Numeric column (histogram)", options=numeric_cols, index=numeric_cols.index(default_num) if default_num in numeric_cols else 0)
                    bins = st.number_input("Bins", min_value=5, max_value=500, value=last_settings.get("bins", 30), step=1)
                    color_col = None
                    if cat_cols:
                        color_col = st.selectbox("Color by (optional categorical)", options=[None] + cat_cols, index=0)
                    chart_params.update({"x_col": x_col, "bins": bins, "color_col": color_col})

            elif chart_choice == "Boxplot (single numeric)":
                if not numeric_cols:
                    st.info("No numeric columns to plot.")
                else:
                    default_num = last_settings.get("y_col", numeric_cols[0])
                    y_col = st.selectbox("Numeric column (boxplot)", options=numeric_cols, index=numeric_cols.index(default_num) if default_num in numeric_cols else 0)
                    group_col = None
                    if cat_cols:
                        group_col = st.selectbox("Group by (optional categorical)", options=[None] + cat_cols, index=0)
                    chart_params.update({"y_col": y_col, "group_col": group_col})

            elif chart_choice == "Scatter (choose X and Y numeric)":
                if len(numeric_cols) < 2:
                    st.info("Need at least two numeric columns for a scatter plot.")
                else:
                    default_x = last_settings.get("x_col", numeric_cols[0])
                    x_col = st.selectbox("X axis (numeric)", options=numeric_cols, index=numeric_cols.index(default_x) if default_x in numeric_cols else 0, key=f"{ds_key}_scatter_x")
                    default_y = last_settings.get("y_col", numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])
                    y_col = st.selectbox("Y axis (numeric)", options=[c for c in numeric_cols if c != x_col], index=0, key=f"{ds_key}_scatter_y")
                    color_col = None
                    if cat_cols:
                        color_col = st.selectbox("Color by (optional categorical)", options=[None] + cat_cols, index=0)
                    chart_params.update({"x_col": x_col, "y_col": y_col, "color_col": color_col})

            elif chart_choice == "Correlation heatmap (numeric columns)":
                # no extra controls, but show note
                if len(numeric_cols) < 2:
                    st.info("Need at least two numeric columns for correlation heatmap.")
                else:
                    chart_params.update({})

            # Only render the chart when user clicks
            if chart_choice != "None":
                if st.button("Show chart"):
                    # render chart according to params
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

                        # show figure
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Could not create chart with the selected options.")
                    except Exception as e:
                        st.error(f"Error rendering chart: {e}")

                    # Save the "last settings" for this dataset
                    st.session_state[last_key] = {"chart_choice": chart_choice, **chart_params}

                    # Option to save chart config for PDF summary
                    st.markdown("**Save this chart to include in the PDF summary**")
                    caption = st.text_input("Optional short caption for PDF (appears under chart)", value=last_settings.get("caption", ""))
                    if st.button("Save chart to PDF"):
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

            # show list of saved charts (brief)
            if st.session_state["saved_charts"]:
                st.subheader("Charts queued for PDF summary")
                for i, c in enumerate(st.session_state["saved_charts"], 1):
                    st.write(f"{i}. {c['ds_name']} — {c['chart_type']} — caption: {c.get('caption','(none)')}")
