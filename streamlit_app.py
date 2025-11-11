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

    # Helper: validate dataset
    def valid_df(key):
        df = st.session_state.get(key)
        return df is not None and isinstance(df, pd.DataFrame)

    a_exists = valid_df("cleaned_a")
    b_exists = valid_df("cleaned_b")

    if not a_exists and not b_exists:
        st.warning("No cleaned datasets available. Please upload & clean datasets in Tabs 1–2 first.")
    else:
        st.write("Datasets available:", 
                 f"Dataset A = {'yes' if a_exists else 'no'}", 
                 f"Dataset B = {'yes' if b_exists else 'no'}")

        df_a = st.session_state.get("cleaned_a") if a_exists else None
        df_b = st.session_state.get("cleaned_b") if b_exists else None

        # Choose a primary comparison mode
        mode = st.radio("Comparison mode", options=[
            "Columns & Schema", 
            "Row-level differences (left/right)", 
            "Common columns value-count diffs", 
            "Numeric columns stats comparison",
            "Generate full comparison report"
        ])

        # ---------- Columns & Schema ----------
        if mode == "Columns & Schema":
            st.subheader("Columns & Schema comparison")
            cols_a = set(df_a.columns) if df_a is not None else set()
            cols_b = set(df_b.columns) if df_b is not None else set()

            st.write(f"Columns in A: {len(cols_a)}; Columns in B: {len(cols_b)}")
            only_a = sorted(list(cols_a - cols_b))
            only_b = sorted(list(cols_b - cols_a))
            common = sorted(list(cols_a & cols_b))

            st.markdown("**Columns only in A**")
            st.write(only_a or "None")
            st.markdown("**Columns only in B**")
            st.write(only_b or "None")
            st.markdown("**Common columns**")
            st.write(common or "None")

            # show dtypes for common columns if present
            if common:
                st.subheader("Data types for common columns")
                dtypes = []
                for c in common:
                    dtype_a = str(df_a[c].dtype) if df_a is not None else "N/A"
                    dtype_b = str(df_b[c].dtype) if df_b is not None else "N/A"
                    dtypes.append({"column": c, "dtype_a": dtype_a, "dtype_b": dtype_b})
                st.dataframe(pd.DataFrame(dtypes))

        # ---------- Row-level differences ----------
        elif mode == "Row-level differences (left/right)":
            st.subheader("Row-level differences")
            # require at least one dataset
            if df_a is None:
                st.info("Dataset A missing — show rows only in B")
            if df_b is None:
                st.info("Dataset B missing — show rows only in A")

            # Ask for key columns to compare rows by (if not provided, compare full rows)
            st.write("Choose key columns to determine row identity (if none chosen, full-row equality is used).")
            possible_keys = list(set(df_a.columns if df_a is not None else []) | set(df_b.columns if df_b is not None else []))
            key_cols = st.multiselect("Key columns (order matters)", options=possible_keys)

            def rows_only(left, right, keys):
                # left rows not present in right
                if keys:
                    # merge on keys to identify matches
                    merged = left.merge(right[keys].drop_duplicates(), on=keys, how="left", indicator=True)
                    only = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
                    return only
                else:
                    # compare full rows by hashing
                    left_hash = left.apply(lambda row: hash(tuple(row.values)), axis=1)
                    right_hashes = set(right.apply(lambda r: hash(tuple(r.values)), axis=1))
                    mask = left_hash.apply(lambda h: h not in right_hashes)
                    return left[mask]

            if df_a is not None and df_b is not None:
                only_a = rows_only(df_a, df_b, key_cols)
                only_b = rows_only(df_b, df_a, key_cols)

                st.markdown(f"Rows only in A: {only_a.shape[0]}")
                st.dataframe(only_a.head(50))
                st.markdown(f"Rows only in B: {only_b.shape[0]}")
                st.dataframe(only_b.head(50))

                # Download buttons
                buf_a = io.BytesIO()
                only_a.to_csv(buf_a, index=False)
                buf_a.seek(0)
                st.download_button("Download rows only in A (CSV)", data=buf_a, file_name="only_in_A.csv")

                buf_b = io.BytesIO()
                only_b.to_csv(buf_b, index=False)
                buf_b.seek(0)
                st.download_button("Download rows only in B (CSV)", data=buf_b, file_name="only_in_B.csv")
            else:
                # If only one exists, show that dataset
                only = df_a if df_b is None else df_b
                st.markdown(f"Rows in the available dataset: {only.shape[0]}")
                st.dataframe(only.head(100))
                buf = io.BytesIO()
                only.to_csv(buf, index=False)
                buf.seek(0)
                st.download_button("Download dataset (CSV)", data=buf, file_name="dataset_only.csv")

        # ---------- Common columns value-count diffs ----------
        elif mode == "Common columns value-count diffs":
            st.subheader("Value-count diffs for common categorical columns")
            if df_a is None or df_b is None:
                st.info("Both datasets required for this mode.")
            else:
                common = list(set(df_a.columns) & set(df_b.columns))
                if not common:
                    st.info("No common columns to compare.")
                else:
                    cat_common = [c for c in common if df_a[c].dtype == "object" or df_b[c].dtype == "object"]
                    if not cat_common:
                        st.info("No common categorical columns detected. You can still compare numeric stats in another mode.")
                    else:
                        col_choice = st.selectbox("Choose a common categorical column", options=sorted(cat_common))
                        vc_a = df_a[col_choice].fillna("<NA>").value_counts().rename("count_a")
                        vc_b = df_b[col_choice].fillna("<NA>").value_counts().rename("count_b")
                        vc = pd.concat([vc_a, vc_b], axis=1).fillna(0).astype(int)
                        vc["diff_a_minus_b"] = vc["count_a"] - vc["count_b"]
                        st.dataframe(vc.sort_values("diff_a_minus_b", ascending=False))

                        # Download option
                        buf = io.BytesIO()
                        vc.to_csv(buf)
                        buf.seek(0)
                        st.download_button("Download value-count diff (CSV)", data=buf, file_name=f"value_count_diff_{col_choice}.csv")

        # ---------- Numeric columns stats comparison ----------
        elif mode == "Numeric columns stats comparison":
            st.subheader("Numeric stats comparison for shared numeric columns")
            if df_a is None or df_b is None:
                st.info("Both datasets required for numeric stats comparison.")
            else:
                num_common = [c for c in set(df_a.columns) & set(df_b.columns)
                              if pd.api.types.is_numeric_dtype(df_a[c]) and pd.api.types.is_numeric_dtype(df_b[c])]
                if not num_common:
                    st.info("No shared numeric columns detected.")
                else:
                    sel = st.multiselect("Numeric columns to compare", options=sorted(num_common), default=num_common[:5])
                    stats = []
                    for c in sel:
                        a = df_a[c].dropna()
                        b = df_b[c].dropna()
                        stats.append({
                            "column": c,
                            "a_count": int(a.count()),
                            "b_count": int(b.count()),
                            "a_mean": float(a.mean()) if not a.empty else None,
                            "b_mean": float(b.mean()) if not b.empty else None,
                            "a_std": float(a.std()) if not a.empty else None,
                            "b_std": float(b.std()) if not b.empty else None,
                            "mean_diff": (float(a.mean()) - float(b.mean())) if (not a.empty and not b.empty) else None
                        })
                    stat_df = pd.DataFrame(stats)
                    st.dataframe(stat_df)

                    # Download stats CSV
                    buf = io.BytesIO()
                    stat_df.to_csv(buf, index=False)
                    buf.seek(0)
                    st.download_button("Download numeric stats CSV", data=buf, file_name="numeric_stats_comparison.csv")

        # ---------- Generate full comparison report ----------
        elif mode == "Generate full comparison report":
            st.subheader("Full comparison report")
            # Build a report dict of results for export & later tabs
            report = {}
            cols_a = set(df_a.columns) if df_a is not None else set()
            cols_b = set(df_b.columns) if df_b is not None else set()
            report["only_in_a_columns"] = sorted(list(cols_a - cols_b))
            report["only_in_b_columns"] = sorted(list(cols_b - cols_a))
            report["common_columns"] = sorted(list(cols_a & cols_b))

            # Row diffs (full-row based)
            if df_a is not None and df_b is not None:
                # rows only in A
                only_a = df_a.merge(df_b.drop_duplicates(), how="left", indicator=True)
                only_a = only_a[only_a["_merge"] == "left_only"].drop(columns=["_merge"])
                only_b = df_b.merge(df_a.drop_duplicates(), how="left", indicator=True)
                only_b = only_b[only_b["_merge"] == "left_only"].drop(columns=["_merge"])
                report["only_in_a_rows"] = only_a
                report["only_in_b_rows"] = only_b
            else:
                report["only_in_a_rows"] = df_a if df_b is None else pd.DataFrame()
                report["only_in_b_rows"] = df_b if df_a is None else pd.DataFrame()

            # numeric comparison summary for common numeric columns
            if df_a is not None and df_b is not None:
                num_common = [c for c in report["common_columns"]
                              if pd.api.types.is_numeric_dtype(df_a[c]) and pd.api.types.is_numeric_dtype(df_b[c])]
                num_stats = []
                for c in num_common:
                    a = df_a[c].dropna()
                    b = df_b[c].dropna()
                    num_stats.append({
                        "column": c,
                        "a_count": int(a.count()),
                        "b_count": int(b.count()),
                        "a_mean": float(a.mean()) if not a.empty else None,
                        "b_mean": float(b.mean()) if not b.empty else None,
                        "mean_diff": (float(a.mean()) - float(b.mean())) if (not a.empty and not b.empty) else None
                    })
                report["numeric_stats"] = pd.DataFrame(num_stats)
            else:
                report["numeric_stats"] = pd.DataFrame()

            # Save to session state for later exports
            st.session_state["compare_report"] = report
            st.success("Comparison report generated and saved to session_state['compare_report'].")

            # Show quick previews
            st.markdown("**Columns only in A**")
            st.write(report["only_in_a_columns"])
            st.markdown("**Columns only in B**")
            st.write(report["only_in_b_columns"])
            st.markdown("**Numeric stats preview**")
            st.dataframe(report["numeric_stats"].head())

            # Export: bundle into an Excel workbook
            if st.button("Export full report to Excel"):
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="openpyxl") as writer:
                    pd.DataFrame({"only_in_a_columns": report["only_in_a_columns"]}).to_excel(writer, sheet_name="only_in_A_cols", index=False)
                    pd.DataFrame({"only_in_b_columns": report["only_in_b_columns"]}).to_excel(writer, sheet_name="only_in_B_cols", index=False)
                    report["numeric_stats"].to_excel(writer, sheet_name="numeric_stats", index=False)
                    # put row diffs if present
                    if not report["only_in_a_rows"].empty:
                        report["only_in_a_rows"].to_excel(writer, sheet_name="only_in_A_rows", index=False)
                    if not report["only_in_b_rows"].empty:
                        report["only_in_b_rows"].to_excel(writer, sheet_name="only_in_B_rows", index=False)
                out.seek(0)
                st.download_button("Download full report (Excel)", data=out, file_name="comparison_report.xlsx")

            # Also allow CSVs for row diffs if they exist
            if "only_in_a_rows" in report and not report["only_in_a_rows"].empty:
                buf = io.BytesIO()
                report["only_in_a_rows"].to_csv(buf, index=False)
                buf.seek(0)
                st.download_button("Download rows only in A (CSV)", data=buf, file_name="only_in_A_rows.csv")
            if "only_in_b_rows" in report and not report["only_in_b_rows"].empty:
                buf = io.BytesIO()
                report["only_in_b_rows"].to_csv(buf, index=False)
                buf.seek(0)
                st.download_button("Download rows only in B (CSV)", data=buf, file_name="only_in_B_rows.csv")

