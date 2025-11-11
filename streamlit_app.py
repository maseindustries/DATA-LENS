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
tab1, tab2, tab3, tab4 = st.tabs([
    "Upload", "Cleaning", "EDA", "Compare & Contrast"
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
# Tab 3: Exploratory Data Analysis (EDA)
# -----------------------------
with tab3:
    st.title("Exploratory Data Analysis (EDA)")

    # Ensure saved_charts exists
    if "saved_charts" not in st.session_state:
        st.session_state["saved_charts"] = []

    datasets = [
        ("cleaned_a", st.session_state.get("cleaned_a_name", "Dataset A")),
        ("cleaned_b", st.session_state.get("cleaned_b_name", "Dataset B"))
    ]

    # Only keep datasets that exist
    available = [(key, name) for key, name in datasets if isinstance(st.session_state.get(key), pd.DataFrame)]
    if not available:
        st.warning("Please upload & clean at least one dataset in Tabs 1–2 before running EDA.")
    else:
        left_col, center_col, right_col = st.columns([2, 3, 1])

        # ------------------ LEFT: Dataset overview ------------------
        with left_col:
            st.subheader("Dataset selection & overview")
            chosen_name = st.selectbox("Choose dataset", [name for _, name in available], key="eda_choose_ds")
            ds_key = next(key for key, name in available if name == chosen_name)
            df = st.session_state.get(ds_key)

            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            st.markdown(f"**{chosen_name}** — rows × cols: **{df.shape[0]} × {df.shape[1]}**")
            st.write("Columns:", list(df.columns))

            # Expanders for summaries
            with st.expander("Missing values"):
                missing = df.isna().sum()
                if missing.sum() > 0:
                    st.dataframe(missing[missing > 0].sort_values(ascending=False))
                else:
                    st.info("No missing values.")

            with st.expander("Numeric summary"):
                if numeric_cols:
                    st.dataframe(df[numeric_cols].describe().T)
                else:
                    st.info("No numeric columns.")

            with st.expander("Categorical summary"):
                if cat_cols:
                    sel_cat = st.selectbox("Pick a categorical column", options=cat_cols, key=f"{ds_key}_cat")
                    st.dataframe(df[sel_cat].value_counts(dropna=False))
                else:
                    st.info("No categorical columns.")

        # ------------------ CENTER: Charts ------------------
        with center_col:
            st.subheader("Charts")
            chart_options = [
                "None",
                "Histogram (single numeric)",
                "Boxplot (single numeric)",
                "Scatter (numeric X & Y)",
                "Correlation heatmap (numeric columns)"
            ]
            chart_choice = st.selectbox("Choose chart", chart_options, key=f"{ds_key}_chart_choice")

            fig = None  # Figure object
            chart_params = {}  # For saved queue
            caption = st.text_input("Optional caption for saved chart", key=f"{ds_key}_chart_caption")

            # Generate figure dynamically
            if chart_choice == "Histogram (single numeric)" and numeric_cols:
                x_col = st.selectbox("Numeric column", numeric_cols, key=f"{ds_key}_hist_x")
                bins = st.number_input("Bins", min_value=5, max_value=500, value=30, step=1, key=f"{ds_key}_hist_bins")
                color_col = None
                if cat_cols:
                    color_col = st.selectbox("Color by (categorical)", [None]+cat_cols, key=f"{ds_key}_hist_color")
                chart_params.update({"x_col": x_col, "bins": bins, "color_col": color_col})

                fig = px.histogram(df, x=x_col, nbins=bins, color=color_col)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_choice == "Boxplot (single numeric)" and numeric_cols:
                y_col = st.selectbox("Numeric column", numeric_cols, key=f"{ds_key}_box_y")
                group_col = None
                if cat_cols:
                    group_col = st.selectbox("Group by (categorical)", [None]+cat_cols, key=f"{ds_key}_box_group")
                chart_params.update({"y_col": y_col, "group_col": group_col})

                fig = px.box(df, x=group_col, y=y_col) if group_col else px.box(df, y=y_col)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_choice == "Scatter (numeric X & Y)" and len(numeric_cols) >= 2:
                x_col = st.selectbox("X axis", numeric_cols, key=f"{ds_key}_scatter_x")
                y_col = st.selectbox("Y axis", [c for c in numeric_cols if c != x_col], key=f"{ds_key}_scatter_y")
                color_col = None
                if cat_cols:
                    color_col = st.selectbox("Color by (categorical)", [None]+cat_cols, key=f"{ds_key}_scatter_color")
                chart_params.update({"x_col": x_col, "y_col": y_col, "color_col": color_col})

                fig = px.scatter(df, x=x_col, y=y_col, color=df[color_col].astype(str) if color_col else None)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_choice == "Correlation heatmap (numeric columns)" and len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True)
                st.plotly_chart(fig, use_container_width=True)

        # ------------------ RIGHT: Saved Charts Queue ------------------
        with right_col:
            st.subheader("Saved Charts Queue")

            if st.button("Save chart", key=f"{ds_key}_save_chart"):
                if fig is not None:
                    saved = {
                        "ds_key": ds_key,
                        "ds_name": chosen_name,
                        "chart_type": chart_choice,
                        "params": chart_params,
                        "caption": caption,
                        "time": datetime.utcnow().isoformat(),
                        "figure": fig  # Store figure for optional download
                    }
                    st.session_state["saved_charts"].append(saved)
                    st.success("Chart saved")
                else:
                    st.warning("Please generate a chart first before saving")

            saved_charts = st.session_state.get("saved_charts", [])
            if saved_charts:
                for i, chart in enumerate(saved_charts, 1):
                    st.markdown(f"**{i}. {chart['ds_name']} — {chart['chart_type']}**")
                    if chart.get("caption"):
                        st.caption(chart["caption"])

                    remove_key = f"remove_{i}_{chart['ds_key']}"
                    if st.button("Remove", key=remove_key):
                        st.session_state["saved_charts"].pop(i-1)
                        st.experimental_rerun()

                    download_key = f"download_{i}_{chart['ds_key']}"
                    if st.button("Download PNG", key=download_key):
                        img_bytes = chart["figure"].to_image(format="png", width=800, height=600)
                        st.download_button("Download PNG", data=img_bytes, file_name=f"{chart['ds_name']}_{i}.png")
# -----------------------------
# Tab 4: Compare & Contrast
# -----------------------------
with tab4:
    st.header("Compare & Contrast")

    # Get datasets from session state
    A = st.session_state.get("cleaned_a")
    B = st.session_state.get("cleaned_b")
    name_a = st.session_state.get("cleaned_a_name", "Dataset A")
    name_b = st.session_state.get("cleaned_b_name", "Dataset B")

    if A is None:
        st.info("Dataset A is not available. Please upload and clean it in Tabs 1–2.")
    if B is None:
        st.info("Dataset B is not available. Some comparison features will be disabled.")

    st.markdown("---")

    if isinstance(A, pd.DataFrame) and isinstance(B, pd.DataFrame):
        st.write(f"**Datasets available:** {name_a} (A), {name_b} (B)")

        # Detect common columns for matching
        common_cols = list(set(A.columns).intersection(B.columns))
        st.write("Common columns detected:", common_cols or "(none)")

        # Auto-key detection
        auto_key = next((k for k in ["id","ID","Id","key","Key","email","Email"] if k in common_cols), None)
        use_auto = st.checkbox(f"Auto-select '{auto_key}' as join key", value=True) if auto_key else False
        selected_keys = st.multiselect(
            "Select key column(s) to match rows",
            options=common_cols,
            default=[auto_key] if (auto_key and use_auto) else (common_cols[:1] if common_cols else [])
        )

        if selected_keys:
            # Row-level merge for numeric comparisons
            numeric_common = [c for c in common_cols if pd.api.types.is_numeric_dtype(A[c]) and pd.api.types.is_numeric_dtype(B[c])]
            cat_common = [c for c in common_cols if pd.api.types.is_object_dtype(A[c]) and pd.api.types.is_object_dtype(B[c])]

            # ---------------- Numeric comparison ----------------
            if numeric_common:
                st.subheader("Numeric Column Differences")
                chosen_num = st.selectbox("Pick numeric column to compare", options=numeric_common)
                
                # Safe merge
                a_vals = A[selected_keys + [chosen_num]]
                b_vals = B[selected_keys + [chosen_num]]
                joined = pd.merge(
                    a_vals, b_vals,
                    on=selected_keys,
                    how="inner",
                    suffixes=(f"_{name_a}", f"_{name_b}")
                )
                joined["diff"] = joined[f"{chosen_num}_{name_b}"] - joined[f"{chosen_num}_{name_a}"]
                st.dataframe(joined.head(200))

            # ---------------- Categorical comparison ----------------
            if cat_common:
                st.subheader("Categorical Column Differences")
                chosen_cat = st.selectbox("Pick categorical column to compare", options=cat_common)

                a_vals = A[selected_keys + [chosen_cat]]
                b_vals = B[selected_keys + [chosen_cat]]
                joined_cat = pd.merge(
                    a_vals, b_vals,
                    on=selected_keys,
                    how="outer",
                    suffixes=(f"_{name_a}", f"_{name_b}")
                )

                # Show value differences
                joined_cat["match"] = joined_cat[f"{chosen_cat}_{name_a}"] == joined_cat[f"{chosen_cat}_{name_b}"]
                st.dataframe(joined_cat.head(200))

            st.success("Comparison complete!")
        else:
            st.info("Select at least one key column to perform comparisons.")

    st.markdown("---")
