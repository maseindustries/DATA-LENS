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

            fig = None  # figure object
            chart_params = {}  # for PDF saving
            caption = st.text_input("Optional caption for PDF", key=f"{ds_key}_chart_caption")

            # ------------------ Auto-generate figure ------------------
            if chart_choice == "Histogram (single numeric)" and numeric_cols:
                x_col = st.selectbox("Numeric column", numeric_cols, key=f"{ds_key}_hist_x")
                bins = st.number_input("Bins", min_value=5, max_value=500, value=30, step=1, key=f"{ds_key}_hist_bins")
                color_col = None
                if cat_cols:
                    color_col = st.selectbox("Color by (categorical)", [None]+cat_cols, key=f"{ds_key}_hist_color")
                chart_params.update({"x_col": x_col, "bins": bins, "color_col": color_col})

                # Generate figure immediately
                if color_col:
                    fig = px.histogram(df, x=x_col, color=color_col, nbins=bins)
                else:
                    fig = px.histogram(df, x=x_col, nbins=bins)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_choice == "Boxplot (single numeric)" and numeric_cols:
                y_col = st.selectbox("Numeric column", numeric_cols, key=f"{ds_key}_box_y")
                group_col = None
                if cat_cols:
                    group_col = st.selectbox("Group by (categorical)", [None]+cat_cols, key=f"{ds_key}_box_group")
                chart_params.update({"y_col": y_col, "group_col": group_col})

                if group_col:
                    fig = px.box(df, x=group_col, y=y_col)
                else:
                    fig = px.box(df, y=y_col)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_choice == "Scatter (numeric X & Y)" and len(numeric_cols) >= 2:
                x_col = st.selectbox("X axis", numeric_cols, key=f"{ds_key}_scatter_x")
                y_col = st.selectbox("Y axis", [c for c in numeric_cols if c != x_col], key=f"{ds_key}_scatter_y")
                color_col = None
                if cat_cols:
                    color_col = st.selectbox("Color by (categorical)", [None]+cat_cols, key=f"{ds_key}_scatter_color")
                chart_params.update({"x_col": x_col, "y_col": y_col, "color_col": color_col})

                if color_col:
                    fig = px.scatter(df, x=x_col, y=y_col, color=df[color_col].astype(str))
                else:
                    fig = px.scatter(df, x=x_col, y=y_col)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_choice == "Correlation heatmap (numeric columns)" and len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True)
                st.plotly_chart(fig, use_container_width=True)

        # ------------------ RIGHT: PDF Queue ------------------
        with right_col:
            st.subheader("PDF Queue")

            if st.button("Save chart to PDF", key=f"{ds_key}_save_chart"):
                if fig is not None:
                    saved = {
                        "ds_key": ds_key,
                        "ds_name": chosen_name,
                        "chart_type": chart_choice,
                        "params": chart_params,
                        "caption": caption,
                        "time": datetime.utcnow().isoformat(),
                        "figure": fig  # <-- store the figure for Tab 5
                    }
                    st.session_state["saved_charts"].append(saved)
                    st.success("Chart saved to PDF queue")
                else:
                    st.warning("No chart available to save. Adjust parameters first.")
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

    # Basic info
    if A is None:
        st.info("Dataset A is not available. Please upload and clean it in Tabs 1–2.")
    if B is None:
        st.info("Dataset B is not available. Some comparison features will be disabled.")

    st.markdown("---")

    # Only perform comparison if both datasets exist
    if isinstance(A, pd.DataFrame) and isinstance(B, pd.DataFrame):
        st.write(f"**Datasets available:** {name_a} (A), {name_b} (B)")

        # Matching keys
        common_cols = list(set(A.columns).intersection(set(B.columns)))
        st.write("Common columns detected:", common_cols or "(none)")

        auto_key = None
        for cand in ["id", "ID", "Id", "key", "Key", "email", "Email"]:
            if cand in common_cols:
                auto_key = cand
                break

        use_auto = False
        if auto_key:
            use_auto = st.checkbox(f"Auto-select '{auto_key}' as join key", value=True)

        selected_keys = st.multiselect(
            "Select key column(s) to match rows",
            options=common_cols,
            default=[auto_key] if (auto_key and use_auto) else (common_cols[:1] if common_cols else [])
        )

        if selected_keys:
            dupA = A.duplicated(subset=selected_keys, keep=False).sum()
            dupB = B.duplicated(subset=selected_keys, keep=False).sum()
            st.write(f"Key duplicates: {name_a}: {dupA}/{A.shape[0]}, {name_b}: {dupB}/{B.shape[0]}")

            # Merge and compare
            merged = A.merge(B, on=selected_keys, how="outer", indicator=True, suffixes=("_A", "_B"))
            only_a = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
            only_b = merged[merged["_merge"] == "right_only"].drop(columns=["_merge"])
            both = merged[merged["_merge"] == "both"].drop(columns=["_merge"])

            st.markdown("### Summary Metrics")
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Only in {name_a}", f"{only_a.shape[0]:,}")
            c2.metric(f"Only in {name_b}", f"{only_b.shape[0]:,}")
            c3.metric("Matched (both)", f"{both.shape[0]:,}")

            st.markdown("---")

            # Column comparison
            cols_a = set(A.columns)
            cols_b = set(B.columns)
            only_cols_a = sorted(list(cols_a - cols_b))
            only_cols_b = sorted(list(cols_b - cols_a))
            common = sorted(list(cols_a & cols_b))
            c1, c2 = st.columns(2)
            with c1:
                st.subheader(f"Columns only in {name_a} ({len(only_cols_a)})")
                st.write(only_cols_a or "None")
            with c2:
                st.subheader(f"Columns only in {name_b} ({len(only_cols_b)})")
                st.write(only_cols_b or "None")

            st.markdown("---")

            # Save report for PDF
            st.session_state["compare_report"] = {
                "name_a": name_a,
                "name_b": name_b,
                "selected_keys": selected_keys,
                "counts": {"only_a": only_a.shape[0], "only_b": only_b.shape[0], "both": both.shape[0]},
                "only_cols_a": only_cols_a,
                "only_cols_b": only_cols_b,
                "numeric_comparison": [c for c in common if pd.api.types.is_numeric_dtype(A[c]) and pd.api.types.is_numeric_dtype(B[c])],
                "timestamp": datetime.utcnow().isoformat()
            }
            st.success("Compare completed and saved for export/PDF.")
        else:
            st.info("Select at least one key column to perform row-level comparisons.")
    st.markdown("---")
# -----------------------------
# Tab 5: PDF Summary Report
# -----------------------------
with tab5:
    st.header("PDF Summary Report")

    # Optional custom report title
    report_title = st.text_input("Report Title", value="DataLens PDF Summary Report")

    # Get datasets
    cleaned_a = st.session_state.get("cleaned_a")
    cleaned_b = st.session_state.get("cleaned_b")
    name_a = st.session_state.get("cleaned_a_name", "Dataset A")
    name_b = st.session_state.get("cleaned_b_name", "Dataset B")

    st.info("This PDF will include dataset summaries, charts (if generated), and comparisons.")

    # Optional notes for executive summary
    notes = st.text_area("Optional notes / observations for the report", value="")

    if st.button("Generate PDF Summary"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # -----------------------------
        # Title Page / Overview
        # -----------------------------
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, report_title, ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.ln(5)
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(0, 10, f"Dataset A: {name_a}", ln=True)
        pdf.cell(0, 10, f"Dataset B: {name_b if cleaned_b is not None else 'Not uploaded'}", ln=True)
        pdf.ln(10)
        if notes:
            pdf.multi_cell(0, 8, f"Notes: {notes}")

        # -----------------------------
        # Function: Dataset Summary
        # -----------------------------
        def add_dataset_summary(df, dataset_name):
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, f"{dataset_name} Summary", ln=True)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Rows: {df.shape[0]}, Columns: {df.shape[1]}", ln=True)

            # Percent missing and duplicates
            missing_pct = df.isna().mean().mean() * 100
            duplicates = df.duplicated().sum()
            pdf.cell(0, 10, f"% Missing: {missing_pct:.2f}%, Duplicates: {duplicates}", ln=True)

            numeric_cols = df.select_dtypes(include=['number']).columns
            cat_cols = df.select_dtypes(include=['object']).columns
            pdf.cell(0, 10, f"Numeric columns: {len(numeric_cols)}, Categorical columns: {len(cat_cols)}", ln=True)
            pdf.ln(5)

        # -----------------------------
        # Add Dataset Summaries
        # -----------------------------
        if cleaned_a is not None:
            add_dataset_summary(cleaned_a, name_a)

        if cleaned_b is not None:
            add_dataset_summary(cleaned_b, name_b)

        # -----------------------------
        # Add Saved Charts (Safe Handling)
        # -----------------------------
        saved_charts = st.session_state.get("saved_charts", [])
        if saved_charts:
            for chart in saved_charts:
                fig = chart.get("figure")
                pdf.add_page()
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, f"{chart['ds_name']} - {chart['chart_type']}", ln=True)
                if chart.get("caption"):
                    pdf.set_font("Arial", "", 12)
                    pdf.multi_cell(0, 8, f"Caption: {chart['caption']}")

                if fig is not None:
                    try:
                        # Attempt to export chart as PNG (may fail on cloud)
                        img_bytes = fig.to_image(format="png")
                        pdf.image(io.BytesIO(img_bytes), x=10, y=None, w=180)
                    except Exception:
                        pdf.set_font("Arial", "", 12)
                        pdf.cell(0, 10, "Chart not available (environment limitation)", ln=True)
                else:
                    pdf.set_font("Arial", "", 12)
                    pdf.cell(0, 10, "Please generate the chart first before saving.", ln=True)

        # -----------------------------
        # Compare & Contrast Section
        # -----------------------------
        if cleaned_a is not None and cleaned_b is not None:
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Compare & Contrast Summary", ln=True)
            pdf.set_font("Arial", "", 12)

            shared_cols = [c for c in cleaned_a.columns if c in cleaned_b.columns]
            unique_a = [c for c in cleaned_a.columns if c not in cleaned_b.columns]
            unique_b = [c for c in cleaned_b.columns if c not in cleaned_a.columns]

            pdf.cell(0, 10, f"Shared Columns ({len(shared_cols)}): {', '.join(shared_cols)}", ln=True)
            pdf.cell(0, 10, f"Unique to {name_a}: {', '.join(unique_a) if unique_a else 'None'}", ln=True)
            pdf.cell(0, 10, f"Unique to {name_b}: {', '.join(unique_b) if unique_b else 'None'}", ln=True)

            # Basic numeric differences
            numeric_common = [c for c in shared_cols if pd.api.types.is_numeric_dtype(cleaned_a[c]) and pd.api.types.is_numeric_dtype(cleaned_b[c])]
            if numeric_common:
                pdf.ln(5)
                pdf.cell(0, 10, "Numeric differences for shared columns:", ln=True)
                for col in numeric_common:
                    mean_a = cleaned_a[col].mean()
                    mean_b = cleaned_b[col].mean()
                    pdf.cell(0, 8, f"{col} | {name_a} mean: {mean_a:.2f}, {name_b} mean: {mean_b:.2f}, diff: {mean_b - mean_a:.2f}", ln=True)

        # -----------------------------
        # Output PDF (In-memory safe)
        # -----------------------------
        pdf_bytes = pdf.output(dest='S').encode('latin1')  # Correct for BytesIO
        pdf_buffer = io.BytesIO(pdf_bytes)
        st.download_button("Download PDF Summary", data=pdf_buffer, file_name="data_summary.pdf")
