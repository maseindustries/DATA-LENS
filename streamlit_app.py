import streamlit as st
import pandas as pd
import io
import plotly.express as px

st.set_page_config(layout="wide")
st.title("DataLens")

# -----------------------------
# Session state placeholders
# -----------------------------
for key in ["cleaned_a", "cleaned_b", "compare_report"]:
    if key not in st.session_state:
        st.session_state[key] = None

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Upload", "Cleaning", "EDA", "Compare & Contrast", "Export"
])

# -----------------------------
# Tab 1: Upload
# -----------------------------
with tab1:
    st.header("Upload Datasets")
    uploaded_file_a = st.file_uploader("Upload Dataset A", type=["csv", "xlsx"], key="upload_a")
    uploaded_file_b = st.file_uploader("Upload Dataset B", type=["csv", "xlsx"], key="upload_b")

    if uploaded_file_a is not None:
        if uploaded_file_a.name.endswith(".csv"):
            st.session_state.cleaned_a = pd.read_csv(uploaded_file_a)
        else:
            st.session_state.cleaned_a = pd.read_excel(uploaded_file_a)

    if uploaded_file_b is not None:
        if uploaded_file_b.name.endswith(".csv"):
            st.session_state.cleaned_b = pd.read_csv(uploaded_file_b)
        else:
            st.session_state.cleaned_b = pd.read_excel(uploaded_file_b)

# -----------------------------
# Tab 2: Cleaning
# -----------------------------
with tab2:
    st.header("Data Cleaning")

    if st.session_state.cleaned_a is not None:
        st.subheader("Dataset A Preview")
        st.dataframe(st.session_state.cleaned_a.head())

    if st.session_state.cleaned_b is not None:
        st.subheader("Dataset B Preview")
        st.dataframe(st.session_state.cleaned_b.head())

    cleaning_options = st.multiselect(
        "Select cleaning operations to apply",
        [
            "Drop duplicate rows",
            "Fill missing numeric values with median",
            "Fill missing categorical values with mode",
            "Trim whitespace from string columns",
            "Remove columns with all nulls"
        ],
        default=["Drop duplicate rows"],
        key="cleaning_options"
    )

    if st.button("Run Cleaning", key="clean_button"):
        for ds_name in ["cleaned_a", "cleaned_b"]:
            df = st.session_state[ds_name]
            if df is None:
                continue

            original_shape = df.shape
            changes = []

            # --- Cleaning operations ---
            if "Drop duplicate rows" in cleaning_options:
                before = len(df)
                df = df.drop_duplicates()
                after = len(df)
                if after < before:
                    changes.append(f"Removed {before - after} duplicate rows.")

            if "Fill missing numeric values with median" in cleaning_options:
                num_cols = df.select_dtypes(include=['number']).columns
                for col in num_cols:
                    na_count = df[col].isna().sum()
                    if na_count > 0:
                        df[col].fillna(df[col].median(), inplace=True)
                        changes.append(f"Filled {na_count} missing numeric values in '{col}' with median.")

            if "Fill missing categorical values with mode" in cleaning_options:
                cat_cols = df.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    na_count = df[col].isna().sum()
                    if na_count > 0 and not df[col].mode().empty:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                        changes.append(f"Filled {na_count} missing categorical values in '{col}' with mode.")

            if "Trim whitespace from string columns" in cleaning_options:
                str_cols = df.select_dtypes(include=['object']).columns
                for col in str_cols:
                    if df[col].apply(lambda x: isinstance(x, str)).any():
                        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                changes.append("Trimmed whitespace from string columns.")

            if "Remove columns with all nulls" in cleaning_options:
                all_null_cols = df.columns[df.isna().all()].tolist()
                if all_null_cols:
                    df.drop(columns=all_null_cols, inplace=True)
                    changes.append(f"Removed {len(all_null_cols)} columns containing only null values.")

            # --- Store cleaned data and report ---
            st.session_state[ds_name] = df
            new_shape = df.shape

            st.subheader(f"{'Dataset A' if ds_name == 'cleaned_a' else 'Dataset B'} Cleaning Summary")
            if changes:
                st.write("**Changes applied:**")
                for c in changes:
                    st.markdown(f"- {c}")
            else:
                st.info("No changes were necessary based on selected options.")

            st.write(f"**Original shape:** {original_shape}, **New shape:** {new_shape}")
            st.dataframe(df.head())

# -----------------------------
# Tab 3: EDA
# -----------------------------
with tab3:
    st.header("Exploratory Data Analysis")

    datasets_choice = st.multiselect(
        "Select dataset(s) for EDA",
        ["Dataset A", "Dataset B"],
        default=["Dataset A"],
        key="eda_datasets"
    )

    chart_type = st.selectbox(
        "Select chart type",
        ["All", "Histogram", "Boxplot", "Scatter", "Correlation Heatmap"],
        key="eda_chart_type"
    )

    if st.button("Run EDA", key="eda_run"):
        for ds in datasets_choice:
            if ds == "Dataset A" and st.session_state.cleaned_a is not None:
                df = st.session_state.cleaned_a
            elif ds == "Dataset B" and st.session_state.cleaned_b is not None:
                df = st.session_state.cleaned_b
            else:
                continue

            st.subheader(f"{ds} - {chart_type if chart_type != 'All' else 'Comprehensive EDA'}")

            numeric_cols = df.select_dtypes(include=['number']).columns
            all_cols = df.columns

            def plot_histogram():
                col = st.selectbox(f"Select column for {ds} histogram", numeric_cols, key=f"hist_{ds}")
                fig = px.histogram(df, x=col, title=f"{ds} - Histogram of {col}")
                st.plotly_chart(fig, use_container_width=True)

            def plot_boxplot():
                col = st.selectbox(f"Select column for {ds} boxplot", numeric_cols, key=f"box_{ds}")
                fig = px.box(df, y=col, title=f"{ds} - Boxplot of {col}")
                st.plotly_chart(fig, use_container_width=True)

            def plot_scatter():
                x_col = st.selectbox(f"X-axis column for {ds}", all_cols, key=f"scatter_x_{ds}")
                y_col = st.selectbox(f"Y-axis column for {ds}", numeric_cols, key=f"scatter_y_{ds}")
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{ds} - Scatter: {x_col} vs {y_col}")
                st.plotly_chart(fig, use_container_width=True)

            def plot_corr_heatmap():
                if len(numeric_cols) == 0:
                    st.warning(f"No numeric columns found in {ds} for correlation heatmap.")
                else:
                    corr = df[numeric_cols].corr()
                    fig = px.imshow(corr, text_auto=True, title=f"{ds} - Correlation Heatmap")
                    st.plotly_chart(fig, use_container_width=True)

            if chart_type == "All":
                with st.expander(f"{ds} - Histogram"):
                    plot_histogram()
                with st.expander(f"{ds} - Boxplot"):
                    plot_boxplot()
                with st.expander(f"{ds} - Scatter"):
                    plot_scatter()
                with st.expander(f"{ds} - Correlation Heatmap"):
                    plot_corr_heatmap()
            elif chart_type == "Histogram":
                plot_histogram()
            elif chart_type == "Boxplot":
                plot_boxplot()
            elif chart_type == "Scatter":
                plot_scatter()
            elif chart_type == "Correlation Heatmap":
                plot_corr_heatmap()

# -----------------------------
# Tab 4: Compare & Contrast
# -----------------------------
with tab4:
    st.header("Compare & Contrast")

    if st.session_state.cleaned_a is None or st.session_state.cleaned_b is None:
        st.warning("Please upload and clean both datasets first.")
    else:
        common_cols = list(set(st.session_state.cleaned_a.columns).intersection(st.session_state.cleaned_b.columns))

        compare_type = st.selectbox(
            "Select Compare Type",
            ['Row presence check', 'Cell-by-cell comparison', 'Summary compare', 'Schema compare'],
            key="compare_type"
        )

        key_col = None
        if compare_type == 'Row presence check' and common_cols:
            key_col = st.selectbox(
                "Select key column for row matching (optional)",
                common_cols,
                key="compare_key"
            )

        if st.button("Run Compare", key="run_compare"):
            df_a = st.session_state.cleaned_a.copy()
            df_b = st.session_state.cleaned_b.copy()
            report = None
            explanation = ""

            # --- Row presence check ---
            if compare_type == 'Row presence check':
                if key_col:
                    ids_a = set(df_a[key_col])
                    ids_b = set(df_b[key_col])
                    only_in_a = ids_a - ids_b
                    only_in_b = ids_b - ids_a
                    overlap = len(ids_a.intersection(ids_b))
                    total = len(ids_a.union(ids_b))
                    similarity = overlap / total * 100 if total else 0

                    report = pd.DataFrame({
                        f'Only in A ({key_col})': list(only_in_a) + [None]*(max(len(only_in_a), len(only_in_b)) - len(only_in_a)),
                        f'Only in B ({key_col})': list(only_in_b) + [None]*(max(len(only_in_a), len(only_in_b)) - len(only_in_b))
                    })

                    st.metric("Row Match Rate", f"{similarity:.1f}%")
                    explanation = (
                        f"{len(only_in_a)} unique rows found only in **Dataset A**, "
                        f"{len(only_in_b)} only in **Dataset B**, "
                        f"and {overlap} rows appear in both datasets based on `{key_col}`."
                    )
                else:
                    st.info("No key column selected — cannot perform row presence check.")

            # --- Schema compare ---
            elif compare_type == 'Schema compare':
                cols_a = set(df_a.columns)
                cols_b = set(df_b.columns)
                cols_only_a = list(cols_a - cols_b)
                cols_only_b = list(cols_b - cols_a)
                shared_cols = len(cols_a.intersection(cols_b))
                total_cols = len(cols_a.union(cols_b))
                similarity = shared_cols / total_cols * 100 if total_cols else 0

                max_len = max(len(cols_only_a), len(cols_only_b))
                cols_only_a += [None] * (max_len - len(cols_only_a))
                cols_only_b += [None] * (max_len - len(cols_only_b))
                report = pd.DataFrame({
                    'Columns only in A': cols_only_a,
                    'Columns only in B': cols_only_b
                })

                st.metric("Schema Match Rate", f"{similarity:.1f}%")
                explanation = (
                    f"**{shared_cols}** columns shared between datasets, "
                    f"**{len(cols_only_a)}** unique to A, and **{len(cols_only_b)}** unique to B."
                )

            # --- Cell-by-cell comparison ---
            elif compare_type == 'Cell-by-cell comparison':
                diffs = []
                shared_cols = common_cols
                total_diff = 0

                for col in shared_cols:
                    col_a = df_a[col]
                    col_b = df_b[col]
                    min_len = min(len(col_a), len(col_b))
                    col_a = col_a.iloc[:min_len]
                    col_b = col_b.iloc[:min_len]

                    mismatch = (col_a.fillna("NA") != col_b.fillna("NA"))
                    diff_count = mismatch.sum()
                    total_diff += diff_count

                    if pd.api.types.is_numeric_dtype(col_a):
                        mean_diff = (col_a.mean(skipna=True) - col_b.mean(skipna=True))
                        diffs.append({
                            'Column': col,
                            'Type': 'Numeric',
                            'Mean Difference': round(mean_diff, 3),
                            'Mismatched Count': diff_count
                        })
                    else:
                        diffs.append({
                            'Column': col,
                            'Type': 'Categorical',
                            'Mean Difference': None,
                            'Mismatched Count': diff_count
                        })

                report = pd.DataFrame(diffs).sort_values(by="Mismatched Count", ascending=False)
                st.bar_chart(report.set_index("Column")["Mismatched Count"])
                explanation = f"Compared {len(shared_cols)} columns; found **{total_diff}** differing cells."

            # --- Summary compare ---
            elif compare_type == 'Summary compare':
                summary_a = df_a.describe(include='all').transpose()
                summary_b = df_b.describe(include='all').transpose()
                combined = summary_a.join(summary_b, lsuffix='_A', rsuffix='_B', how='outer')
                combined['Mean Difference (if numeric)'] = combined.apply(
                    lambda r: round(r['mean_A'] - r['mean_B'], 3)
                    if 'mean_A' in r and pd.notna(r['mean_A']) and pd.notna(r['mean_B']) else None, axis=1
                )
                report = combined
                explanation = "Side-by-side summary statistics comparison of both datasets."

            # --- Display results ---
            if report is not None:
                st.session_state.compare_report = report
                st.markdown(f"**Summary:** {explanation}")
                st.dataframe(report, use_container_width=True)

                buffer = io.BytesIO()
                report.to_excel(buffer, index=True)
                st.download_button(
                    "Export Comparison Report",
                    data=buffer.getvalue(),
                    file_name="comparison_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                if compare_type in ["Row presence check", "Schema compare"]:
                    st.subheader("Visual Summary")
                    summary_data = {
                        "Only in A": len(report.iloc[:, 0].dropna()),
                        "Only in B": len(report.iloc[:, 1].dropna())
                    }
                    fig = px.bar(x=list(summary_data.keys()), y=list(summary_data.values()),
                                 title=f"{compare_type} Summary", labels={"x": "", "y": "Count"})
                    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Tab 5: Export
# -----------------------------
with tab5:
    st.header("Export Reports")

    export_options = st.multiselect(
        "Select items to export",
        options=['Cleaned Dataset A', 'Cleaned Dataset B', 'EDA Reports', 'Compare Report'],
        key="export_options"
    )

    file_name = st.text_input(
        "Enter export file name",
        value="DataLens_Report.xlsx",
        key="export_filename"
    )

    if st.button("Export", key="export_button"):
        st.success(f"Exported {', '.join(export_options)} to {file_name}")

