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
    st.title("Exploratory Data Analysis (EDA)")

    # Ensure saved_charts exists
    if "saved_charts" not in st.session_state:
        st.session_state["saved_charts"] = []

    # Datasets and friendly names
    datasets = [
        ("cleaned_a", st.session_state.get("cleaned_a_name", "Dataset A")),
        ("cleaned_b", st.session_state.get("cleaned_b_name", "Dataset B"))
    ]

    available = [(key, name) for key, name in datasets if isinstance(st.session_state.get(key), pd.DataFrame)]
    if not available:
        st.warning("Please upload & clean at least one dataset in Tabs 1–2 before running EDA.")
    else:
        # Column layout: left = overview, center = chart, right = queue
        left_col, center_col, right_col = st.columns([2, 3, 1])

        # ------------------ LEFT: Dataset overview ------------------
        with left_col:
            st.subheader("Dataset selection & overview")
            display_names = [name for _, name in available]
            chosen_name = st.selectbox("Choose dataset", options=display_names, key="eda_choose_ds")
            ds_key = next(key for key, name in available if name == chosen_name)
            df = st.session_state.get(ds_key)

            if df is None:
                st.error("Dataset not available. Please return to Cleaning (Tab 2).")
            else:
                st.markdown(f"**{chosen_name}** — rows × cols: **{df.shape[0]} × {df.shape[1]}**")
                st.write("Columns:", list(df.columns))

                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

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
                        vc = df[sel_cat].value_counts(dropna=False)
                        st.dataframe(vc)
                    else:
                        st.info("No categorical columns.")

                with st.expander("Row / Column Inspector"):
                    col_to_view = st.selectbox("Pick a column to inspect", options=df.columns, key=f"{ds_key}_inspect_col")
                    st.dataframe(df[[col_to_view]].head(10))

                if st.button("Download column summary CSV", key=f"{ds_key}_download_summary"):
                    summary = [{"column": c, "dtype": str(df[c].dtype),
                                "n_unique": int(df[c].nunique(dropna=True)),
                                "n_missing": int(df[c].isna().sum())} for c in df.columns]
                    summary_df = pd.DataFrame(summary)
                    buffer = io.BytesIO()
                    summary_df.to_csv(buffer, index=False)
                    buffer.seek(0)
                    st.download_button("Download CSV", data=buffer, file_name=f"{ds_key}_summary.csv")

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
            chart_choice = st.selectbox("Choose chart", options=chart_options, key=f"{ds_key}_chart_choice")
            chart_params = {}

            # Chart-specific controls
            if chart_choice == "Histogram (single numeric)" and numeric_cols:
                x_col = st.selectbox("Numeric column", options=numeric_cols, key=f"{ds_key}_hist_x")
                bins = st.number_input("Bins", min_value=5, max_value=500, value=30, step=1, key=f"{ds_key}_hist_bins")
                color_col = None
                if cat_cols:
                    color_col = st.selectbox("Color by (categorical)", options=[None]+cat_cols, key=f"{ds_key}_hist_color")
                chart_params.update({"x_col": x_col, "bins": bins, "color_col": color_col})

            elif chart_choice == "Boxplot (single numeric)" and numeric_cols:
                y_col = st.selectbox("Numeric column", options=numeric_cols, key=f"{ds_key}_box_y")
                group_col = None
                if cat_cols:
                    group_col = st.selectbox("Group by (categorical)", options=[None]+cat_cols, key=f"{ds_key}_box_group")
                chart_params.update({"y_col": y_col, "group_col": group_col})

            elif chart_choice == "Scatter (numeric X & Y)" and len(numeric_cols)>=2:
                x_col = st.selectbox("X axis", options=numeric_cols, key=f"{ds_key}_scatter_x")
                y_col = st.selectbox("Y axis", options=[c for c in numeric_cols if c != x_col], key=f"{ds_key}_scatter_y")
                color_col = None
                if cat_cols:
                    color_col = st.selectbox("Color by (categorical)", options=[None]+cat_cols, key=f"{ds_key}_scatter_color")
                chart_params.update({"x_col": x_col, "y_col": y_col, "color_col": color_col})

            elif chart_choice == "Correlation heatmap (numeric columns)" and len(numeric_cols)>=2:
                chart_params.update({})

            # Show chart & save to PDF queue
            st.write("")
            col_show, col_save = st.columns([1,1])
            caption = st.text_input("Optional caption for PDF", key=f"{ds_key}_chart_caption")
            with col_show:
                if st.button("Show chart", key=f"{ds_key}_show_chart"):
                    fig = None
                    try:
                        if chart_choice == "Histogram (single numeric)":
                            if chart_params.get("color_col"):
                                fig = px.histogram(df, x=chart_params["x_col"], color=chart_params["color_col"], nbins=chart_params["bins"])
                            else:
                                fig = px.histogram(df, x=chart_params["x_col"], nbins=chart_params["bins"])
                        elif chart_choice == "Boxplot (single numeric)":
                            if chart_params.get("group_col"):
                                fig = px.box(df, x=chart_params["group_col"], y=chart_params["y_col"])
                            else:
                                fig = px.box(df, y=chart_params["y_col"])
                        elif chart_choice == "Scatter (numeric X & Y)":
                            if chart_params.get("color_col"):
                                fig = px.scatter(df, x=chart_params["x_col"], y=chart_params["y_col"], color=df[chart_params["color_col"]].astype(str))
                            else:
                                fig = px.scatter(df, x=chart_params["x_col"], y=chart_params["y_col"])
                        elif chart_choice == "Correlation heatmap (numeric columns)":
                            corr = df[numeric_cols].corr()
                            fig = px.imshow(corr, text_auto=True)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error rendering chart: {e}")

            with col_save:
                if st.button("Save chart to PDF", key=f"{ds_key}_save_chart"):
                    saved = {"ds_key": ds_key, "ds_name": chosen_name,
                             "chart_type": chart_choice, "params": chart_params,
                             "caption": caption, "time": datetime.utcnow().isoformat()}
                    st.session_state["saved_charts"].append(saved)
                    st.success("Chart saved to PDF queue")

        # ------------------ RIGHT: PDF Queue ------------------
        with right_col:
            st.subheader("PDF Queue")
            queue = st.session_state.get("saved_charts", [])
            if not queue:
                st.info("No charts queued yet.")
            else:
                for i, c in enumerate(queue,1):
                    st.markdown(f"**{i}. {c['ds_name']}** — {c['chart_type']}")
                    if c.get("caption"):
                        st.caption(c["caption"])
                    remove_key = f"remove_{i}_{ds_key}"
                    if st.button("Remove", key=remove_key):
                        st.session_state["saved_charts"].pop(i-1)
                        st.experimental_rerun()

            if queue:
                if st.button("Clear PDF queue", key=f"{ds_key}_clear_queue"):
                    st.session_state["saved_charts"] = []
                    st.success("PDF queue cleared")
# ------------------ Tab 4 ------------------
with tab4:
    st.header("Compare & Contrast")

    # Safety: get cleaned datasets
    A = st.session_state.get("cleaned_a")
    B = st.session_state.get("cleaned_b")
    name_a = st.session_state.get("cleaned_a_name", "Dataset A")
    name_b = st.session_state.get("cleaned_b_name", "Dataset B")

    # Only proceed if both datasets exist
    if not isinstance(A, pd.DataFrame) or not isinstance(B, pd.DataFrame):
        st.info("Please upload and clean a second dataset to use this feature.")
        st.stop()  # Stop rendering further
    else:
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

        selected_keys = st.multiselect("Select key column(s) to match rows", options=common_cols,
                                       default=[auto_key] if (auto_key and use_auto) else (common_cols[:1] if common_cols else []))
        if not selected_keys:
            st.warning("Select at least one key column to perform row-level comparisons.")
            st.stop()

        dupA = A.duplicated(subset=selected_keys, keep=False).sum()
        dupB = B.duplicated(subset=selected_keys, keep=False).sum()
        st.write(f"Key duplicates: {name_a}: {dupA}/{A.shape[0]}, {name_b}: {dupB}/{B.shape[0]}")

        # Merge and compare
        merged = A.merge(B, on=selected_keys, how="outer", indicator=True, suffixes=("_A", "_B"))
        only_a = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
        only_b = merged[merged["_merge"] == "right_only"].drop(columns=["_merge"])
        both = merged[merged["_merge"] == "both"].drop(columns=["_merge"])

        # Summary metrics
        st.markdown("### Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Only in {name_a}", f"{only_a.shape[0]:,}")
        c2.metric(f"Only in {name_b}", f"{only_b.shape[0]:,}")
        c3.metric("Matched (both)", f"{both.shape[0]:,}")

        # Expanders with previews and export
        for label, df_part, fname in [
            (f"Rows only in {name_a}", only_a, f"only_in_{name_a}.csv"),
            (f"Rows only in {name_b}", only_b, f"only_in_{name_b}.csv"),
            ("Rows in both", both, "matched_rows.csv")
        ]:
            with st.expander(f"Preview: {label} ({df_part.shape[0]})", expanded=False):
                if not df_part.empty:
                    st.dataframe(df_part.head(200))
                    buf = io.BytesIO()
                    df_part.to_csv(buf, index=False)
                    buf.seek(0)
                    st.download_button(f"Download {fname}", data=buf, file_name=fname)
                else:
                    st.info(f"No rows for {label}.")

        # Column comparison
        st.markdown("### Column presence comparison")
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

        # Numeric differences
        st.markdown("### Numeric differences for common numeric columns")
        numeric_common = [c for c in common if pd.api.types.is_numeric_dtype(A[c]) and pd.api.types.is_numeric_dtype(B[c])]
        if numeric_common:
            stats = []
            for col in numeric_common:
                a_series = A.set_index(selected_keys)[col] if selected_keys else A[col]
                b_series = B.set_index(selected_keys)[col] if selected_keys else B[col]
                joined = a_series.to_frame("A").join(b_series.to_frame("B"), how="inner").dropna()
                if not joined.empty:
                    diff = joined["A"] - joined["B"]
                    stats.append({
                        "column": col,
                        "n_compared": int(joined.shape[0]),
                        "mean_diff": float(diff.mean()),
                        "median_diff": float(diff.median()),
                        "std_diff": float(diff.std())
                    })
            if stats:
                st.dataframe(pd.DataFrame(stats).set_index("column"))
            else:
                st.info("No overlapping numeric values to compare.")
        else:
            st.info("No numeric columns in common.")

        # Save report for PDF/export
        st.session_state["compare_report"] = {
            "name_a": name_a,
            "name_b": name_b,
            "selected_keys": selected_keys,
            "counts": {"only_a": only_a.shape[0], "only_b": only_b.shape[0], "both": both.shape[0]},
            "only_cols_a": only_cols_a,
            "only_cols_b": only_cols_b,
            "numeric_comparison": numeric_common,
            "timestamp": datetime.utcnow().isoformat()
        }
        st.success("Compare completed and saved for export/PDF.")
