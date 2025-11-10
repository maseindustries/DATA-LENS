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
    "cleaned_a_name", "cleaned_b_name", "compare_report"
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
    for ds_name, label in [("cleaned_a", "Dataset A"), ("cleaned_b", "Dataset B")]:
        df = st.session_state.get(ds_name)
        if df is not None:
            st.subheader(f"{label} Preview")
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

    custom_name_a = st.text_input("Name Dataset A (optional)", value="Dataset A Cleaned")
    custom_name_b = st.text_input("Name Dataset B (optional)", value="Dataset B Cleaned")

    if st.button("Run Cleaning"):
        for ds_name, label in [("cleaned_a", "Dataset A"), ("cleaned_b", "Dataset B")]:
            df = st.session_state.get(ds_name)
            if df is None:
                continue

            original_shape = df.shape
            changes = []

            # Cleaning operations
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
                    df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                changes.append("Trimmed whitespace from string columns.")

            if "Remove columns with all nulls" in cleaning_options:
                all_null_cols = df.columns[df.isna().all()].tolist()
                if all_null_cols:
                    df.drop(columns=all_null_cols, inplace=True)
                    changes.append(f"Removed {len(all_null_cols)} columns containing only null values.")

            st.session_state[ds_name] = df
            new_shape = df.shape

            st.subheader(f"{label} Cleaning Summary")
            if changes:
                st.write("**Changes applied:**")
                for c in changes:
                    st.markdown(f"- {c}")
            else:
                st.info("No changes were necessary based on selected options.")
            st.write(f"**Original shape:** {original_shape}, **New shape:** {new_shape}")
            st.dataframe(df.head())

    if st.button("Save Changes"):
        if st.session_state.cleaned_a is not None:
            st.session_state['cleaned_a_saved'] = st.session_state.cleaned_a.copy()
            st.session_state['cleaned_a_name'] = custom_name_a
        if st.session_state.cleaned_b is not None:
            st.session_state['cleaned_b_saved'] = st.session_state.cleaned_b.copy()
            st.session_state['cleaned_b_name'] = custom_name_b
        st.success("Changes saved! All future analysis will use these named datasets.")

    if st.button("Reset Changes"):
        st.session_state['cleaned_a_saved'] = None
        st.session_state['cleaned_b_saved'] = None
        st.success("Saved changes reset. Using original cleaned preview data.")

# -----------------------------
# Tab 3: EDA
# -----------------------------
with tab3:
    st.header("Exploratory Data Analysis")
    datasets_choice = st.multiselect(
        "Select dataset(s) for EDA",
        [
            st.session_state.get('cleaned_a_name') or "Dataset A",
            st.session_state.get('cleaned_b_name') or "Dataset B"
        ],
        default=[st.session_state.get('cleaned_a_name') or "Dataset A"]
    )

    chart_type = st.selectbox(
        "Select chart type",
        ["All", "Histogram", "Boxplot", "Scatter", "Correlation Heatmap"]
    )

    for ds in datasets_choice:
        df = None
        if ds == (st.session_state.get('cleaned_a_name') or "Dataset A"):
            df = st.session_state.get('cleaned_a_saved') or st.session_state.get('cleaned_a')
        elif ds == (st.session_state.get('cleaned_b_name') or "Dataset B"):
            df = st.session_state.get('cleaned_b_saved') or st.session_state.get('cleaned_b')
        if df is None:
            continue

        st.subheader(f"{ds} - {chart_type if chart_type != 'All' else 'Comprehensive EDA'}")
        numeric_cols = df.select_dtypes(include=['number']).columns
        all_cols = df.columns

        col1, col2, col3 = st.columns(3)

        with col1:
            if chart_type in ["All", "Histogram"]:
                hist_col = st.selectbox(f"Histogram Column ({ds})", numeric_cols, key=f"hist_{ds}")
        with col2:
            if chart_type in ["All", "Boxplot"]:
                box_col = st.selectbox(f"Boxplot Column ({ds})", numeric_cols, key=f"box_{ds}")
        with col3:
            if chart_type in ["All", "Scatter"]:
                scatter_x = st.selectbox(f"Scatter X ({ds})", all_cols, key=f"scatter_x_{ds}")
                scatter_y = st.selectbox(f"Scatter Y ({ds})", numeric_cols, key=f"scatter_y_{ds}")

        if st.button(f"Generate {ds} EDA"):
            if chart_type in ["All", "Histogram"]:
                fig = px.histogram(df, x=hist_col, title=f"{ds} Histogram: {hist_col}")
                st.plotly_chart(fig, use_container_width=True)
            if chart_type in ["All", "Boxplot"]:
                fig = px.box(df, y=box_col, title=f"{ds} Boxplot: {box_col}")
                st.plotly_chart(fig, use_container_width=True)
            if chart_type in ["All", "Scatter"]:
                fig = px.scatter(df, x=scatter_x, y=scatter_y, title=f"{ds} Scatter: {scatter_x} vs {scatter_y}")
                st.plotly_chart(fig, use_container_width=True)
            if chart_type in ["All", "Correlation Heatmap"]:
                if len(numeric_cols) > 0:
                    corr = df[numeric_cols].corr()
                    fig = px.imshow(corr, text_auto=True, title=f"{ds} Correlation Heatmap")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No numeric columns for correlation in {ds}.")
# -----------------------------
# Tab 4: Compare & Contrast
# -----------------------------
with tab4:
    st.header("Compare & Contrast")

    df_a = st.session_state['cleaned_a_saved'] if st.session_state.get('cleaned_a_saved') is not None else st.session_state.get('cleaned_a')
    df_b = st.session_state['cleaned_b_saved'] if st.session_state.get('cleaned_b_saved') is not None else st.session_state.get('cleaned_b')

    if df_a is None and df_b is None:
        st.warning("Upload and clean at least one dataset first.")
    else:
        common_cols = list(set(df_a.columns if df_a is not None else []).intersection(df_b.columns if df_b is not None else []))
        compare_type = st.selectbox(
            "Select Compare Type",
            ['Row presence check', 'Cell-by-cell comparison', 'Summary compare', 'Schema compare']
        )

        key_col = None
        if compare_type == 'Row presence check' and common_cols:
            key_col = st.selectbox("Select key column for row matching (optional)", common_cols)

        if st.button("Run Compare"):
            report = None
            explanation = ""

            # Row presence check
            if compare_type == 'Row presence check' and key_col and df_a is not None and df_b is not None:
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
                explanation = f"{len(only_in_a)} unique rows only in A, {len(only_in_b)} only in B, {overlap} rows appear in both."

            # Schema compare
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

                st.metric("Schema Match Rate", f"{similarity:.1f}%")
                explanation = f"{shared_cols} columns shared, {len(cols_only_a)} unique to A, {len(cols_only_b)} unique to B."

            # Cell-by-cell comparison
            elif compare_type == 'Cell-by-cell comparison' and df_a is not None and df_b is not None:
                numeric_cols = df_a.select_dtypes(include=['number']).columns.intersection(df_b.select_dtypes(include=['number']).columns)
                diffs = []
                total_diff = 0
                for col in numeric_cols.union(common_cols):
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

            # Summary compare
            elif compare_type == 'Summary compare':
                if df_a is not None:
                    summary_a = df_a.describe(include='all').transpose()
                else:
                    summary_a = pd.DataFrame()
                if df_b is not None:
                    summary_b = df_b.describe(include='all').transpose()
                else:
                    summary_b = pd.DataFrame()
                combined = summary_a.join(summary_b, lsuffix='_A', rsuffix='_B', how='outer')
                combined['Mean Difference (if numeric)'] = combined.apply(
                    lambda r: round(r.get('mean_A', 0) - r.get('mean_B', 0), 3) if pd.notna(r.get('mean_A')) and pd.notna(r.get('mean_B')) else None,
                    axis=1
                )
                report = combined
                explanation = "Side-by-side summary statistics comparison."

            # Display
            if report is not None:
                st.session_state.compare_report = report
                st.markdown(f"**Summary:** {explanation}")
                st.dataframe(report, use_container_width=True)


# -----------------------------
# Tab 5: Export
# -----------------------------
with tab5:
    st.header("Export Reports")
    export_options = st.multiselect(
        "Select items to export to Excel",
        ['Cleaned Dataset A', 'Cleaned Dataset B', 'Duplicates', 'Outliers']
    )
    file_name = st.text_input("Export file name", value="DataLens_Report.xlsx")

    def detect_outliers(df):
        numeric_cols = df.select_dtypes(include=['number']).columns
        if numeric_cols.empty:
            return pd.DataFrame()
        outlier_rows = pd.DataFrame()
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            mask = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
            outlier_rows = pd.concat([outlier_rows, df[mask]])
        return outlier_rows.drop_duplicates()

    if st.button("Export Selected"):
        if not export_options:
            st.warning("Select at least one item to export.")
        else:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                for ds_name, label in [('cleaned_a_saved','A'), ('cleaned_b_saved','B'), ('cleaned_a','A'), ('cleaned_b','B')]:
                    df = st.session_state.get(ds_name)
                    if df is None:
                        continue
                    if f'Cleaned Dataset {label}' in export_options:
                        df.to_excel(writer, sheet_name=f'Cleaned_{label}', index=False)
                    if 'Duplicates' in export_options:
                        duplicates = df[df.duplicated(keep=False)]
                        if not duplicates.empty:
                            duplicates.to_excel(writer, sheet_name=f'Duplicates_{label}', index=False)
                    if 'Outliers' in export_options:
                        outliers = detect_outliers(df)
                        if not outliers.empty:
                            outliers.to_excel(writer, sheet_name=f'Outliers_{label}', index=False)
            st.download_button("Download Excel Report", data=buffer.getvalue(),
                               file_name=file_name,
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.success(f"Exported: {', '.join(export_options)}")


# -----------------------------
# Tab 6: PDF Report
# -----------------------------
with tab6:
    st.header("Generate PDF Report")

    dataset_choice = st.selectbox(
        "Select Dataset for PDF Report",
        ["Dataset A", "Dataset B"]
    )
    df = st.session_state['cleaned_a_saved'] if dataset_choice=="Dataset A" and st.session_state.get('cleaned_a_saved') is not None else (
        st.session_state['cleaned_b_saved'] if dataset_choice=="Dataset B" and st.session_state.get('cleaned_b_saved') is not None else (
        st.session_state['cleaned_a'] if dataset_choice=="Dataset A" else st.session_state['cleaned_b'])
    )

    if df is None:
        st.warning("Please upload and clean the dataset first.")
    else:
        pdf_sections = st.multiselect(
            "Select sections to include in PDF",
            [
                "Data Overview (rows, columns, missing, duplicates)",
                "Descriptive Statistics",
                "Outlier Summary",
                "Top N Categories",
                "Correlation Matrix",
                "Selected Charts"
            ],
            default=["Data Overview (rows, columns, missing, duplicates)"]
        )

        top_n = st.number_input("Top N categories for categorical columns", min_value=1, max_value=20, value=5)

        if st.button("Generate PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, f"{dataset_choice} - DataLens Report", ln=True, align="C")
            pdf.ln(10)

            # Data Overview
            if "Data Overview (rows, columns, missing, duplicates)" in pdf_sections:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Data Overview", ln=True)
                pdf.set_font("Arial", '', 10)
                pdf.multi_cell(0, 5, f"Number of rows: {df.shape[0]}\nNumber of columns: {df.shape[1]}")
                missing = df.isna().sum()
                pdf.multi_cell(0, 5, f"Missing values per column:\n{missing.to_string()}")
                duplicates = df.duplicated().sum()
                pdf.cell(0, 5, f"Duplicate rows: {duplicates}", ln=True)
                pdf.ln(5)

            # Descriptive Statistics
            if "Descriptive Statistics" in pdf_sections:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Descriptive Statistics", ln=True)
                stats = df.describe(include='all').transpose().head(20)
                pdf.set_font("Arial", '', 8)
                pdf.multi_cell(0, 5, stats.to_string())
                pdf.ln(5)

            # Outlier Summary
            if "Outlier Summary" in pdf_sections:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Outlier Summary", ln=True)
                outliers = detect_outliers(df)
                pdf.set_font("Arial", '', 10)
                if not outliers.empty:
                    pdf.multi_cell(0, 5, f"Found {len(outliers)} outlier rows.")
                else:
                    pdf.multi_cell(0, 5, "No outliers detected.")
                pdf.ln(5)

            # Top N Categories
            if "Top N Categories" in pdf_sections:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, f"Top {top_n} Categories per Column", ln=True)
                pdf.set_font("Arial", '', 10)
                cat_cols = df.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    top_vals = df[col].value_counts().head(top_n)
                    pdf.multi_cell(0, 5, f"{col}:\n{top_vals.to_string()}\n")
                pdf.ln(5)

            # Correlation Matrix
            if "Correlation Matrix" in pdf_sections:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    corr = df[numeric_cols].corr()
                    fig = px.imshow(corr, text_auto=True)
                    chart_file = f"corr_matrix_{dataset_choice}.png"
                    fig.write_image(chart_file)
                    pdf.image(chart_file, w=180)
                    os.remove(chart_file)
                pdf.ln(5)

            # Insights
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Insights", ln=True)
            pdf.set_font("Arial", '', 10)
            insights_text = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
            if df.select_dtypes(include=['object']).shape[1] > 0:
                top_missing = df.isna().sum().sort_values(ascending=False).head(5)
                insights_text += "Columns with most missing values — " + "; ".join([f"{c}: {v}" for c,v in top_missing.items()]) + ". "
            duplicates = df.duplicated().sum()
            insights_text += f"There are {duplicates} internal duplicate rows. "
            pdf.multi_cell(0, 5, insights_text)

            try:
                out_bytes = pdf.output(dest='S').encode('latin-1')
                st.download_button("Download PDF Report", data=out_bytes, file_name=f"{dataset_choice}_DataLens_Report.pdf")
            except Exception as e:
                st.error(f"Error generating PDF: {e}")
