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

    if uploaded_file_a is not None:
        if uploaded_file_a.name.endswith(".csv"):
            st.session_state.cleaned_a = pd.read_csv(uploaded_file_a)
        else:
            st.session_state['cleaned_a'] = pd.read_excel(uploaded_file_a)
        st.session_state['cleaned_a_name'] = st.session_state.get('cleaned_a_name', 'Dataset A')
        st.session_state['cleaned_a_saved'] = None
        st.session_state['cleaned_a_operations'] = []

    if uploaded_file_b is not None:
        if uploaded_file_b.name.endswith(".csv"):
            st.session_state.cleaned_b = pd.read_csv(uploaded_file_b)
        else:
            st.session_state.cleaned_b = pd.read_excel(uploaded_file_b)
            st.session_state['cleaned_b_name'] = st.session_state.get('cleaned_b_name', 'Dataset B')
        st.session_state['cleaned_b_saved'] = None
        st.session_state['cleaned_b_operations'] = []
            
# -----------------------------
# Tab 2: Cleaning
# -----------------------------
with tab2:
    st.header("Data Cleaning")

    # Preview datasets
    for ds_name, label in [("cleaned_a", "Dataset A"), ("cleaned_b", "Dataset B")]:
        df = st.session_state.get(ds_name)
        if df is not None:
            st.subheader(f"{label} Preview")
            st.dataframe(df.head())

    # Cleaning options
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

    # Custom names
    custom_name_a = st.text_input("Name Dataset A (optional)", value=st.session_state.get('cleaned_a_name', 'Dataset A'))
    custom_name_b = st.text_input("Name Dataset B (optional)", value=st.session_state.get('cleaned_b_name', 'Dataset B'))

    if st.button("Run Cleaning"):
        for ds_name, label, custom_name in [
            ("cleaned_a", "Dataset A", custom_name_a),
            ("cleaned_b", "Dataset B", custom_name_b)
        ]:
            df = st.session_state.get(ds_name)
            if df is None:
                continue

            original_shape = df.shape
            applied_ops = []

            # Cleaning operations
            if "Drop duplicate rows" in cleaning_options:
                before = len(df)
                df = df.drop_duplicates()
                after = len(df)
                if after < before:
                    applied_ops.append("Duplicate rows removed")

            if "Fill missing numeric values with median" in cleaning_options:
                num_cols = df.select_dtypes(include=['number']).columns
                for col in num_cols:
                    na_count = df[col].isna().sum()
                    if na_count > 0:
                        df[col].fillna(df[col].median(), inplace=True)
                        applied_ops.append(f"Filled {na_count} missing numeric values in {col}")

            if "Fill missing categorical values with mode" in cleaning_options:
                cat_cols = df.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    na_count = df[col].isna().sum()
                    if na_count > 0 and not df[col].mode().empty:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                        applied_ops.append(f"Filled {na_count} missing categorical values in {col}")

            if "Trim whitespace from string columns" in cleaning_options:
                str_cols = df.select_dtypes(include=['object']).columns
                for col in str_cols:
                    df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                applied_ops.append("Trimmed whitespace from string columns")

            if "Remove columns with all nulls" in cleaning_options:
                all_null_cols = df.columns[df.isna().all()].tolist()
                if all_null_cols:
                    df.drop(columns=all_null_cols, inplace=True)
                    applied_ops.append(f"Removed {len(all_null_cols)} columns with all nulls")

            # Save cleaned dataset
            st.session_state[ds_name] = df
            st.session_state[f"{ds_name}_name"] = custom_name
            st.session_state[f"{ds_name}_operations"] = applied_ops

            # Display cleaning summary
            new_shape = df.shape
            st.subheader(f"{label} Cleaning Summary")
            if applied_ops:
                st.write("**Changes applied:**")
                for op in applied_ops:
                    st.markdown(f"- {op}")
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
        # Safely retrieve the DataFrame
        if ds == st.session_state.get('cleaned_a_name', 'Dataset A'):
            df = st.session_state['cleaned_a_saved'] if st.session_state.get('cleaned_a_saved') is not None else st.session_state.get('cleaned_a')
        else:
            df = st.session_state['cleaned_b_saved'] if st.session_state.get('cleaned_b_saved') is not None else st.session_state.get('cleaned_b')

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
        
# Tab 4: Compare & Contrast
# -----------------------------
with tab4:
    st.header("Compare & Contrast")

    # Safely get datasets
    df_a = st.session_state['cleaned_a_saved'] if st.session_state.get('cleaned_a_saved') is not None else st.session_state.get('cleaned_a')
    df_b = st.session_state['cleaned_b_saved'] if st.session_state.get('cleaned_b_saved') is not None else st.session_state.get('cleaned_b')

    name_a = st.session_state.get('cleaned_a_name', 'Dataset A')
    name_b = st.session_state.get('cleaned_b_name', 'Dataset B')

    if df_a is None and df_b is None:
        st.warning("Upload and clean at least one dataset first.")
    else:
        # Determine common columns safely
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
            if compare_type == 'Row presence check' and df_a is not None and df_b is not None and key_col:
                ids_a = set(df_a[key_col])
                ids_b = set(df_b[key_col])
                only_in_a = ids_a - ids_b
                only_in_b = ids_b - ids_a
                overlap = len(ids_a.intersection(ids_b))
                total = len(ids_a.union(ids_b))
                similarity = overlap / total * 100 if total else 0

                max_len = max(len(only_in_a), len(only_in_b))
                only_in_a_list = list(only_in_a) + [None]*(max_len - len(only_in_a))
                only_in_b_list = list(only_in_b) + [None]*(max_len - len(only_in_b))
                report = pd.DataFrame({
                    f'Only in {name_a} ({key_col})': only_in_a_list,
                    f'Only in {name_b} ({key_col})': only_in_b_list
                })

                st.metric("Row Match Rate", f"{similarity:.1f}%")
                explanation = f"{len(only_in_a)} unique rows only in {name_a}, {len(only_in_b)} only in {name_b}, {overlap} rows appear in both."

            # Schema compare
            elif compare_type == 'Schema compare':
                cols_a = set(df_a.columns if df_a is not None else [])
                cols_b = set(df_b.columns if df_b is not None else [])
                cols_only_a = list(cols_a - cols_b)
                cols_only_b = list(cols_b - cols_a)
                shared_cols = len(cols_a.intersection(cols_b))
                total_cols = len(cols_a.union(cols_b))
                similarity = shared_cols / total_cols * 100 if total_cols else 0

                max_len = max(len(cols_only_a), len(cols_only_b))
                cols_only_a += [None] * (max_len - len(cols_only_a))
                cols_only_b += [None] * (max_len - len(cols_only_b))
                report = pd.DataFrame({
                    'Columns only in '+name_a: cols_only_a,
                    'Columns only in '+name_b: cols_only_b
                })
                st.metric("Schema Match Rate", f"{similarity:.1f}%")
                explanation = f"{shared_cols} columns shared, {len(cols_only_a)} unique to {name_a}, {len(cols_only_b)} unique to {name_b}."

            # Cell-by-cell comparison
            elif compare_type == 'Cell-by-cell comparison' and df_a is not None and df_b is not None:
                diffs = []
                total_diff = 0
                shared_cols = common_cols
                for col in shared_cols:
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
                explanation = f"Compared {len(shared_cols)} columns; found {total_diff} differing cells."

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

    # Safely get cleaned operations
    cleaned_a_ops = st.session_state.get('cleaned_a_operations')
    if not isinstance(cleaned_a_ops, list):
        cleaned_a_ops = []

    cleaned_b_ops = st.session_state.get('cleaned_b_operations')
    if not isinstance(cleaned_b_ops, list):
        cleaned_b_ops = []

    # Dataset names
    name_a = st.session_state.get('cleaned_a_name', 'Dataset A')
    name_b = st.session_state.get('cleaned_b_name', 'Dataset B')

    # Create export options
    export_options_list = [
        f"{name_a} - {op}" for op in cleaned_a_ops
    ] + [
        f"{name_b} - {op}" for op in cleaned_b_ops
    ] + ["Duplicates", "Outliers"]

    export_options = st.multiselect(
        "Select items to export to Excel",
        options=export_options_list,
        default=[]
    )

    file_name = st.text_input("Export file name", value="DataLens_Report.xlsx")

    # Outlier detection function
    def detect_outliers(df):
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
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

                # Export cleaning operations for Dataset A
                df_a = st.session_state.get('cleaned_a')
                for op in cleaned_a_ops:
                    label = f"{name_a} - {op}"
                    if label in export_options and df_a is not None:
                        df_a.to_excel(writer, sheet_name=label[:31], index=False)

                # Export cleaning operations for Dataset B
                df_b = st.session_state.get('cleaned_b')
                for op in cleaned_b_ops:
                    label = f"{name_b} - {op}"
                    if label in export_options and df_b is not None:
                        df_b.to_excel(writer, sheet_name=label[:31], index=False)

                # Export duplicates
                if "Duplicates" in export_options:
                    for name, df in [(name_a, df_a), (name_b, df_b)]:
                        if df is not None:
                            duplicates = df[df.duplicated(keep=False)]
                            if not duplicates.empty:
                                duplicates.to_excel(writer, sheet_name=f"Duplicates_{name[:25]}", index=False)

                # Export outliers
                if "Outliers" in export_options:
                    for name, df in [(name_a, df_a), (name_b, df_b)]:
                        if df is not None:
                            outliers = detect_outliers(df)
                            if not outliers.empty:
                                outliers.to_excel(writer, sheet_name=f"Outliers_{name[:25]}", index=False)

            st.download_button(
                "Download Excel Report",
                data=buffer.getvalue(),
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success(f"Exported: {', '.join(export_options)}")
# -----------------------------
# Tab 6: PDF Report
# -----------------------------
with tab6:
    st.header("Generate PDF Report")

    # Select dataset safely
    name_a = st.session_state.get('cleaned_a_name', 'Dataset A')
    name_b = st.session_state.get('cleaned_b_name', 'Dataset B')

    dataset_choice = st.selectbox(
        "Select Dataset for PDF Report",
        [name_a, name_b]
    )
    df = st.session_state.get('cleaned_a') if dataset_choice == name_a else st.session_state.get('cleaned_b')

    if df is None:
        st.warning("Please upload and clean the dataset first.")
    else:
        # Sections to include
        pdf_sections = st.multiselect(
            "Select sections to include in PDF",
            [
                "Data Overview (rows, columns, missing, duplicates)",
                "Descriptive Statistics",
                "Outlier Summary",
                "Cleaning Operations",
                "Top N Categories",
                "Charts",
                "Insights Paragraph"
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
                numeric_cols = df.select_dtypes(include=['number']).columns
                outliers = pd.DataFrame()
                if len(numeric_cols) > 0:
                    for col in numeric_cols:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        mask = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
                        outliers = pd.concat([outliers, df[mask]])
                    outliers = outliers.drop_duplicates()
                pdf.set_font("Arial", '', 10)
                pdf.multi_cell(0, 5, f"Found {len(outliers)} outlier rows")
                pdf.ln(5)

            # Cleaning Operations
            if "Cleaning Operations" in pdf_sections:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Applied Cleaning Operations", ln=True)
                pdf.set_font("Arial", '', 10)
                ops = st.session_state.get('cleaned_a_operations' if dataset_choice == name_a else 'cleaned_b_operations', [])
                if ops:
                    for op in ops:
                        pdf.multi_cell(0, 5, f"- {op}")
                else:
                    pdf.multi_cell(0, 5, "No cleaning operations applied.")
                pdf.ln(5)

            # Top N Categories
            if "Top N Categories" in pdf_sections:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, f"Top {top_n} Categories", ln=True)
                pdf.set_font("Arial", '', 10)
                cat_cols = df.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    top_vals = df[col].value_counts().head(top_n)
                    pdf.multi_cell(0, 5, f"{col}:\n{top_vals.to_string()}")
                    pdf.ln(2)

            # Charts placeholder
            if "Charts" in pdf_sections:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Charts included", ln=True)
                pdf.set_font("Arial", '', 10)
                pdf.multi_cell(0, 5, "Charts from EDA would be inserted here.")
                pdf.ln(5)

            # Insights Paragraph
            if "Insights Paragraph" in pdf_sections:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Insights", ln=True)
                pdf.set_font("Arial", '', 10)
                insights = f"The dataset {dataset_choice} has {df.shape[0]} rows and {df.shape[1]} columns. "
                missing_summary = ", ".join([f"{col}: {val}" for col, val in df.isna().sum().items() if val > 0])
                if missing_summary:
                    insights += f"Columns with missing values — {missing_summary}. "
                duplicates_count = df.duplicated().sum()
                if duplicates_count > 0:
                    insights += f"There are {duplicates_count} duplicate rows. "
                if len(cat_cols) > 0:
                    sample_col = cat_cols[0]
                    top_val = df[sample_col].value_counts().idxmax()
                    insights += f"Top value in {sample_col}: {top_val}."
                pdf.multi_cell(0, 5, insights)
                pdf.ln(5)

            # Output PDF
            out_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("Download PDF", data=out_bytes, file_name=f"{dataset_choice}_Report.pdf")

