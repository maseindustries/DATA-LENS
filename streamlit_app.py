import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
from fpdf import FPDF
from datetime import datetime
import os

st.set_page_config(layout="wide")
st.title("DataLens Pro")

# -----------------------------
# Session state placeholders
# -----------------------------
for key in [
    "dataset_a_original","dataset_b_original",
    "dataset_a_cleaned","dataset_b_cleaned",
    "dataset_a_cleaned_duplicates","dataset_b_cleaned_duplicates",
    "dataset_a_cleaned_outliers","dataset_b_cleaned_outliers",
    "compare_report"
]:
    if key not in st.session_state:
        st.session_state[key] = None
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Upload", "Cleaning", "EDA", "Compare & Contrast", "Export", "PDF Report"
])

with tab1:
    st.header("Upload Datasets")

    uploaded_file_a = st.file_uploader("Upload Dataset A", type=["csv", "xlsx"], key="upload_a")
    uploaded_file_b = st.file_uploader("Upload Dataset B (optional)", type=["csv", "xlsx"], key="upload_b")

    def load_file(file):
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)

    if uploaded_file_a:
        st.session_state.dataset_a_original = load_file(uploaded_file_a)
        st.session_state.dataset_a_cleaned = st.session_state.dataset_a_original.copy()
    if uploaded_file_b:
        st.session_state.dataset_b_original = load_file(uploaded_file_b)
        st.session_state.dataset_b_cleaned = st.session_state.dataset_b_original.copy()

    # Basic summary
    for ds_label, key in [("Dataset A", "dataset_a_cleaned"), ("Dataset B", "dataset_b_cleaned")]:
        df = st.session_state.get(key)
        if df is not None:
            st.subheader(f"{ds_label} Summary")
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            st.write("Columns and types:")
            st.dataframe(pd.DataFrame(df.dtypes, columns=['Type']))
            st.write("Missing values per column:")
            st.dataframe(df.isna().sum())
            st.write(f"Duplicate rows: {df.duplicated().sum()}")
with tab2:
    st.header("Data Cleaning")

    for ds_label, cleaned_key in [("Dataset A", "dataset_a_cleaned"), ("Dataset B", "dataset_b_cleaned")]:
        df = st.session_state.get(cleaned_key)
        if df is not None:
            st.subheader(f"{ds_label} Preview")
            st.dataframe(df.head())

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

    if st.button("Run Cleaning"):
        for ds_label, cleaned_key in [("Dataset A", "dataset_a_cleaned"), ("Dataset B", "dataset_b_cleaned")]:
            df = st.session_state.get(cleaned_key)
            if df is None:
                continue

            original_shape = df.shape
            changes = []

            # Cleaning operations
            if "Drop duplicate rows" in cleaning_options:
                dup_before = df.duplicated().sum()
                df = df.drop_duplicates()
                dup_after = df.duplicated().sum()
                if dup_before > dup_after:
                    changes.append(f"Removed {dup_before - dup_after} duplicate rows.")

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
                    df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                changes.append("Trimmed whitespace from string columns.")

            if "Remove columns with all nulls" in cleaning_options:
                null_cols = df.columns[df.isna().all()].tolist()
                if null_cols:
                    df.drop(columns=null_cols, inplace=True)
                    changes.append(f"Removed {len(null_cols)} columns with all null values.")

            # Save duplicates/outliers for later use
            st.session_state[f"{cleaned_key}_duplicates"] = df[df.duplicated(keep=False)]

            def detect_outliers(df_local):
                numeric_cols = df_local.select_dtypes(include=['number']).columns
                outliers = pd.DataFrame()
                for col in numeric_cols:
                    Q1 = df_local[col].quantile(0.25)
                    Q3 = df_local[col].quantile(0.75)
                    IQR = Q3 - Q1
                    mask = (df_local[col] < Q1 - 1.5*IQR) | (df_local[col] > Q3 + 1.5*IQR)
                    outliers = pd.concat([outliers, df_local[mask]])
                return outliers.drop_duplicates()

            st.session_state[f"{cleaned_key}_outliers"] = detect_outliers(df)

            st.session_state[cleaned_key] = df
            new_shape = df.shape

            # Cleaning summary
            st.subheader(f"{ds_label} Cleaning Summary")
            if changes:
                st.write("**Changes applied:**")
                for c in changes:
                    st.markdown(f"- {c}")
            else:
                st.info("No changes were necessary based on selected options.")
            st.write(f"**Original shape:** {original_shape}, **New shape:** {new_shape}")
            st.dataframe(df.head())
with tab3:
    st.header("Exploratory Data Analysis")

    datasets_choice = st.multiselect(
        "Select dataset(s) for EDA",
        ["Dataset A", "Dataset B"],
        default=["Dataset A"]
    )

    chart_type = st.selectbox(
        "Select chart type",
        ["All", "Histogram", "Boxplot", "Scatter", "Correlation Heatmap"]
    )

    top_n = st.number_input("Top N insights for categorical columns", min_value=1, max_value=20, value=5)

    for ds in datasets_choice:
        df = st.session_state.dataset_a_cleaned if ds == "Dataset A" else st.session_state.dataset_b_cleaned
        if df is None:
            continue

        st.subheader(f"{ds} - {chart_type if chart_type != 'All' else 'Comprehensive EDA'}")
        numeric_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(include=['object']).columns
        all_cols = df.columns

        # Top N insights for categorical columns
        if len(cat_cols) > 0:
            st.write("**Top N Categories**")
            top_tables = {col: df[col].value_counts().head(top_n) for col in cat_cols}
            for col, table in top_tables.items():
                st.write(f"{col}:")
                st.dataframe(table)

        # Helper plotting functions
        def plot_histogram(col):
            fig = px.histogram(df, x=col, title=f"{ds} Histogram: {col}")
            st.plotly_chart(fig, use_container_width=True)

        def plot_boxplot(col):
            fig = px.box(df, y=col, title=f"{ds} Boxplot: {col}")
            st.plotly_chart(fig, use_container_width=True)

        def plot_scatter(x_col, y_col):
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{ds} Scatter: {x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)

        def plot_corr_heatmap():
            if len(numeric_cols) == 0:
                st.warning(f"No numeric columns in {ds}")
            else:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True, title=f"{ds} Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)

        # Reactive column selections
        if chart_type == "All":
            with st.expander(f"{ds} - Histogram"):
                col = st.selectbox(f"{ds} Histogram Column", numeric_cols, key=f"hist_{ds}")
                plot_histogram(col)
            with st.expander(f"{ds} - Boxplot"):
                col = st.selectbox(f"{ds} Boxplot Column", numeric_cols, key=f"box_{ds}")
                plot_boxplot(col)
            with st.expander(f"{ds} - Scatter"):
                x_col = st.selectbox(f"{ds} Scatter X", all_cols, key=f"scatter_x_{ds}")
                y_col = st.selectbox(f"{ds} Scatter Y", numeric_cols, key=f"scatter_y_{ds}")
                plot_scatter(x_col, y_col)
            with st.expander(f"{ds} - Correlation Heatmap"):
                plot_corr_heatmap()
        elif chart_type == "Histogram":
            col = st.selectbox(f"{ds} Histogram Column", numeric_cols, key=f"hist_{ds}")
            plot_histogram(col)
        elif chart_type == "Boxplot":
            col = st.selectbox(f"{ds} Boxplot Column", numeric_cols, key=f"box_{ds}")
            plot_boxplot(col)
        elif chart_type == "Scatter":
            x_col = st.selectbox(f"{ds} Scatter X", all_cols, key=f"scatter_x_{ds}")
            y_col = st.selectbox(f"{ds} Scatter Y", numeric_cols, key=f"scatter_y_{ds}")
            plot_scatter(x_col, y_col)
        elif chart_type == "Correlation Heatmap":
            plot_corr_heatmap()
with tab4:
    st.header("Compare & Contrast")
    df_a = st.session_state.dataset_a_cleaned
    df_b = st.session_state.dataset_b_cleaned

    if df_a is None and df_b is None:
        st.warning("Upload at least one dataset first.")
    else:
        compare_type = st.selectbox(
            "Select Compare Type",
            ['Row presence check', 'Cell-by-cell comparison', 'Summary compare', 'Schema compare']
        )

        key_col = None
        if compare_type == 'Row presence check' and df_a is not None and df_b is not None:
            common_cols = list(set(df_a.columns).intersection(df_b.columns))
            if common_cols:
                key_col = st.selectbox("Select key column for row matching (optional)", common_cols)

        if st.button("Run Compare"):
            report = None
            explanation = ""
            # Row presence
            if compare_type == 'Row presence check' and df_a is not None and df_b is not None and key_col:
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
                explanation = f"{len(only_in_a)} unique rows only in A, {len(only_in_b)} only in B, {overlap} rows appear in both."
                st.metric("Row Match Rate", f"{similarity:.1f}%")

            # Schema compare (single file OK)
            elif compare_type == 'Schema compare':
                cols_a = set(df_a.columns) if df_a is not None else set()
                cols_b = set(df_b.columns) if df_b is not None else set()
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
                explanation = f"{shared_cols} columns shared, {len(cols_only_a)} unique to A, {len(cols_only_b)} unique to B."
                st.metric("Schema Match Rate", f"{similarity:.1f}%")

            # Summary compare
            elif compare_type == 'Summary compare':
                summary_a = df_a.describe(include='all').transpose() if df_a is not None else pd.DataFrame()
                summary_b = df_b.describe(include='all').transpose() if df_b is not None else pd.DataFrame()
                combined = summary_a.join(summary_b, lsuffix='_A', rsuffix='_B', how='outer')
                combined['Mean Difference (if numeric)'] = combined.apply(
                    lambda r: round(r['mean_A'] - r['mean_B'], 3)
                    if 'mean_A' in r and pd.notna(r['mean_A']) and pd.notna(r['mean_B']) else None, axis=1
                )
                report = combined
                explanation = "Side-by-side summary statistics comparison."

            # Cell-by-cell
            elif compare_type == 'Cell-by-cell comparison' and df_a is not None and df_b is not None:
                common_cols = list(set(df_a.columns).intersection(df_b.columns))
                diffs = []
                total_diff = 0
                for col in common_cols:
                    col_a = df_a[col].iloc[:min(len(df_a), len(df_b))]
                    col_b = df_b[col].iloc[:min(len(df_a), len(df_b))]
                    mismatch = (col_a.fillna("NA") != col_b.fillna("NA"))
                    diff_count = mismatch.sum()
                    total_diff += diff_count
                    if pd.api.types.is_numeric_dtype(col_a):
                        mean_diff = round(col_a.mean() - col_b.mean(), 3)
                        diffs.append({'Column': col, 'Type': 'Numeric', 'Mean Difference': mean_diff, 'Mismatched Count': diff_count})
                    else:
                        diffs.append({'Column': col, 'Type': 'Categorical', 'Mean Difference': None, 'Mismatched Count': diff_count})
                report = pd.DataFrame(diffs).sort_values(by="Mismatched Count", ascending=False)
                st.bar_chart(report.set_index("Column")["Mismatched Count"])
                explanation = f"Compared {len(common_cols)} columns; found {total_diff} differing cells."

            if report is not None:
                st.session_state.compare_report = report
                st.markdown(f"**Summary:** {explanation}")
                st.dataframe(report, use_container_width=True)
with tab5:
    st.header("Export Reports")
    export_options = st.multiselect(
        "Select items to export to Excel",
        ['Cleaned Dataset A', 'Cleaned Dataset B', 'Duplicates', 'Outliers']
    )
    file_name = st.text_input("Export file name", value="DataLens_Report.xlsx")

    if st.button("Export Selected"):
        if not export_options:
            st.warning("Select at least one item to export.")
        else:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                if 'Cleaned Dataset A' in export_options and st.session_state.dataset_a_cleaned is not None:
                    st.session_state.dataset_a_cleaned.to_excel(writer, sheet_name='Cleaned_A', index=False)
                if 'Cleaned Dataset B' in export_options and st.session_state.dataset_b_cleaned is not None:
                    st.session_state.dataset_b_cleaned.to_excel(writer, sheet_name='Cleaned_B', index=False)
                if 'Duplicates' in export_options:
                    for ds_key, label in [('dataset_a_cleaned', 'A'), ('dataset_b_cleaned', 'B')]:
                        df = st.session_state.get(ds_key)
                        duplicates = st.session_state.get(f"{ds_key}_duplicates")
                        if duplicates is not None and not duplicates.empty:
                            duplicates.to_excel(writer, sheet_name=f'Duplicates_{label}', index=False)
                if 'Outliers' in export_options:
                    for ds_key, label in [('dataset_a_cleaned', 'A'), ('dataset_b_cleaned', 'B')]:
                        outliers = st.session_state.get(f"{ds_key}_outliers")
                        if outliers is not None and not outliers.empty:
                            outliers.to_excel(writer, sheet_name=f'Outliers_{label}', index=False)

            st.download_button("Download Excel Report", data=buffer.getvalue(),
                               file_name=file_name,
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.success(f"Exported: {', '.join(export_options)}")
with tab6:
    st.header("Generate PDF Report")

    dataset_choice = st.selectbox("Select Dataset for PDF Report", ["Dataset A", "Dataset B"])
    df = st.session_state.dataset_a_cleaned if dataset_choice == "Dataset A" else st.session_state.dataset_b_cleaned

    if df is None:
        st.warning("Please upload and clean the dataset first.")
    else:
        pdf_sections = st.multiselect(
            "Select sections to include in PDF",
            [
                "Data Overview",
                "Descriptive Statistics",
                "Outlier Summary",
                "Top N Categories",
                "Charts",
                "Correlation Matrix"
            ],
            default=["Data Overview"]
        )

        top_n = st.number_input("Top N categories for categorical columns", min_value=1, max_value=20, value=5)

        chart_types = st.multiselect(
            "Select charts to include",
            ["Histogram", "Boxplot", "Scatter", "Correlation Heatmap"],
            default=["Histogram", "Boxplot"]
        )

        if st.button("Generate PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, f"{dataset_choice} - DataLens Report", ln=True, align="C")
            pdf.ln(10)

            # --- Data Overview ---
            if "Data Overview" in pdf_sections:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Data Overview", ln=True)
                pdf.set_font("Arial", '', 10)
                pdf.multi_cell(0, 5, f"Number of rows: {df.shape[0]}\nNumber of columns: {df.shape[1]}")
                missing = df.isna().sum()
                pdf.multi_cell(0, 5, f"Missing values per column:\n{missing.to_string()}")
                duplicates = df.duplicated().sum()
                pdf.cell(0, 5, f"Duplicate rows: {duplicates}", ln=True)
                pdf.ln(5)

            # --- Descriptive Statistics ---
            if "Descriptive Statistics" in pdf_sections:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Descriptive Statistics", ln=True)
                stats = df.describe(include='all').transpose().head(20)
                pdf.set_font("Arial", '', 8)
                pdf.multi_cell(0, 5, stats.to_string())
                pdf.ln(5)

            # --- Top N Categories ---
            if "Top N Categories" in pdf_sections:
                cat_cols = df.select_dtypes(include=['object']).columns
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, f"Top {top_n} Categories", ln=True)
                pdf.set_font("Arial", '', 8)
                for col in cat_cols:
                    pdf.multi_cell(0, 5, f"{col}:\n{df[col].value_counts().head(top_n).to_string()}")
                    pdf.ln(2)

            # --- Outliers ---
            if "Outlier Summary" in pdf_sections:
                numeric_cols = df.select_dtypes(include=['number']).columns
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Outlier Summary", ln=True)
                pdf.set_font("Arial", '', 8)
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                    pdf.multi_cell(0, 5, f"{col}:\n{outliers[col].to_string()}")
                    pdf.ln(2)

            # --- Charts ---
            if "Charts" in pdf_sections:
                import matplotlib.pyplot as plt
                import tempfile

                for chart in chart_types:
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, chart, ln=True)
                    if chart == "Histogram":
                        for col in df.select_dtypes(include=['number']).columns:
                            fig = px.histogram(df, x=col)
                            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                            fig.write_image(tmpfile.name)
                            pdf.image(tmpfile.name, w=180)
                            tmpfile.close()
                    elif chart == "Boxplot":
                        for col in df.select_dtypes(include=['number']).columns:
                            fig = px.box(df, y=col)
                            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                            fig.write_image(tmpfile.name)
                            pdf.image(tmpfile.name, w=180)
                            tmpfile.close()
                    # Scatter/Correlation can be added similarly

            # Download PDF
            pdf_file = f"{dataset_choice}_DataLens_Report.pdf"
            pdf.output(pdf_file)
            with open(pdf_file, "rb") as f:
                st.download_button("Download PDF", f.read(), file_name=pdf_file, mime="application/pdf")
