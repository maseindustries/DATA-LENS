import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------- Page Config -----------------
st.set_page_config(layout="wide")
st.title("DataLens")

# ----------------- Session State -----------------
for key in ["uploaded_file_a", "uploaded_file_b", "cleaned_a", "cleaned_b", "compare_report", "last_plotly_figure", "original_a", "original_b"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ----------------- Tabs -----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Upload", "Cleaning", "EDA", "Compare & Contrast", "Export", "PDF Report"
])

# ----------------- Tab 1: Upload -----------------
with tab1:
    st.header("Upload Datasets")

    uploaded_file_a = st.file_uploader("Upload Dataset A", type=["csv", "xlsx"], key="upload_a")
    uploaded_file_b = st.file_uploader("Upload Dataset B", type=["csv", "xlsx"], key="upload_b")

    if uploaded_file_a is not None:
        st.session_state.uploaded_file_a = uploaded_file_a
        if uploaded_file_a.name.endswith(".csv"):
            df_a = pd.read_csv(uploaded_file_a)
        else:
            df_a = pd.read_excel(uploaded_file_a)
        st.session_state.cleaned_a = df_a.copy()
        st.session_state.original_a = df_a.copy()
        st.success(f"Loaded {uploaded_file_a.name}")

    if uploaded_file_b is not None:
        st.session_state.uploaded_file_b = uploaded_file_b
        if uploaded_file_b.name.endswith(".csv"):
            df_b = pd.read_csv(uploaded_file_b)
        else:
            df_b = pd.read_excel(uploaded_file_b)
        st.session_state.cleaned_b = df_b.copy()
        st.session_state.original_b = df_b.copy()
        st.success(f"Loaded {uploaded_file_b.name}")

# ----------------- Tab 2: Cleaning -----------------
with tab2:
    st.header("Data Cleaning")

    for ds_name, label in [("cleaned_a", "Dataset A"), ("cleaned_b", "Dataset B")]:
        df = st.session_state.get(ds_name)
        if df is not None:
            st.subheader(f"{label} Preview")
            st.dataframe(df.head(), use_container_width=True)

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

    col_rename = st.text_input("Optional: Rename this dataset for reference", value="")

    if st.button("Run Cleaning"):
        for ds_name, label in [("cleaned_a", "Dataset A"), ("cleaned_b", "Dataset B")]:
            df = st.session_state.get(ds_name)
            if df is None:
                continue

            original_shape = df.shape
            changes = []

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
            st.dataframe(df.head(), use_container_width=True)

    if st.button("Reset Cleaning"):
        if st.session_state.original_a is not None:
            st.session_state.cleaned_a = st.session_state.original_a.copy()
        if st.session_state.original_b is not None:
            st.session_state.cleaned_b = st.session_state.original_b.copy()
        st.success("Datasets reset to original uploaded state.")

# ----------------- Tab 3: EDA -----------------
with tab3:
    st.header("Exploratory Data Analysis")
    datasets_choice = st.multiselect(
        "Select dataset(s) for EDA",
        ["Dataset A", "Dataset B"],
        default=["Dataset A"]
    )

    chart_type = st.selectbox(
        "Select chart type",
        ["Histogram", "Boxplot", "Scatter", "Correlation Heatmap"]
    )

    def plot_histogram(df, ds):
        numeric_cols = df.select_dtypes(include=['number']).columns
        col = st.selectbox(f"{ds} Histogram Column", numeric_cols, key=f"hist_{ds}")
        fig = px.histogram(df, x=col, title=f"{ds} Histogram: {col}")
        st.plotly_chart(fig, use_container_width=True)
        st.session_state["last_plotly_figure"] = fig

    def plot_boxplot(df, ds):
        numeric_cols = df.select_dtypes(include=['number']).columns
        col = st.selectbox(f"{ds} Boxplot Column", numeric_cols, key=f"box_{ds}")
        fig = px.box(df, y=col, title=f"{ds} Boxplot: {col}")
        st.plotly_chart(fig, use_container_width=True)
        st.session_state["last_plotly_figure"] = fig

    def plot_scatter(df, ds):
        numeric_cols = df.select_dtypes(include=['number']).columns
        all_cols = df.columns
        x_col = st.selectbox(f"{ds} Scatter X", all_cols, key=f"scatter_x_{ds}")
        y_col = st.selectbox(f"{ds} Scatter Y", numeric_cols, key=f"scatter_y_{ds}")
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{ds} Scatter: {x_col} vs {y_col}")
        st.plotly_chart(fig, use_container_width=True)
        st.session_state["last_plotly_figure"] = fig

    def plot_corr_heatmap(df, ds):
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            st.warning(f"No numeric columns in {ds}")
        else:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, title=f"{ds} Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
            st.session_state["last_plotly_figure"] = fig

    for ds in datasets_choice:
        df = st.session_state.cleaned_a if ds == "Dataset A" else st.session_state.cleaned_b
        if df is None:
            continue

        st.subheader(f"{ds} - {chart_type}")
        if chart_type == "Histogram":
            plot_histogram(df, ds)
        elif chart_type == "Boxplot":
            plot_boxplot(df, ds)
        elif chart_type == "Scatter":
            plot_scatter(df, ds)
        elif chart_type == "Correlation Heatmap":
            plot_corr_heatmap(df, ds)

        # Insights paragraph
        st.subheader(f"{ds} Insights")
        n_rows, n_cols = df.shape
        missing = df.isna().sum()
        duplicates = df.duplicated().sum()
        top_cat_summary = []
        for col in df.select_dtypes(include=['object']).columns:
            val = df[col].value_counts().head(1)
            if not val.empty:
                top_cat_summary.append(f"{col} → {val.index[0]} ({val.iloc[0]})")
        corr = df.select_dtypes(include=['number']).corr()
        high_corr = ""
        if not corr.empty:
            corr_unstacked = corr.abs().unstack()
            corr_unstacked = corr_unstacked[corr_unstacked < 1]
            if not corr_unstacked.empty:
                top_corr = corr_unstacked.sort_values(ascending=False).head(1)
                high_corr = f"Highest numeric correlation: {top_corr.index[0][0]} vs {top_corr.index[0][1]} = {top_corr.iloc[0]:.2f}"
        insight_text = f"The dataset has {n_rows} rows and {n_cols} columns.\nColumns with most missing values: {missing[missing>0].to_dict()}\nDuplicate rows: {duplicates}\nTop categories: {', '.join(top_cat_summary)}\n{high_corr}"
        st.text(insight_text)
# ----------------- Tab 4: Compare & Contrast -----------------
with tab4:
    st.header("Compare & Contrast")
    if st.session_state.cleaned_a is None and st.session_state.cleaned_b is None:
        st.warning("Upload at least one dataset first.")
    else:
        df_a = st.session_state.cleaned_a
        df_b = st.session_state.cleaned_b
        common_cols = list(set(df_a.columns).intersection(df_b.columns)) if df_a is not None and df_b is not None else []

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
            elif compare_type == 'Schema compare' and df_a is not None and df_b is not None:
                cols_a = set(df_a.columns)
                cols_b = set(df_b.columns)
                cols_only_a = list(cols_a - cols_b)
                cols_only_b = list(cols_b - cols_a)
                shared_cols = len(cols_a.intersection(cols_b))
                total_cols = len(cols_a.union(cols_b))
                similarity = shared_cols / total_cols * 100 if total_cols else 0

                max_len = max(len(cols_only_a), len(cols_only_b))
                cols_only_a += [None]*(max_len-len(cols_only_a))
                cols_only_b += [None]*(max_len-len(cols_only_b))
                report = pd.DataFrame({'Columns only in A': cols_only_a, 'Columns only in B': cols_only_b})
                st.metric("Schema Match Rate", f"{similarity:.1f}%")
                explanation = f"{shared_cols} columns shared, {len(cols_only_a)} unique to A, {len(cols_only_b)} unique to B."

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

            if report is not None:
                st.session_state.compare_report = report
                st.markdown(f"**Summary:** {explanation}")
                st.dataframe(report, use_container_width=True)
# ----------------- Tab 5: Export -----------------
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
                if 'Cleaned Dataset A' in export_options and st.session_state.cleaned_a is not None:
                    st.session_state.cleaned_a.to_excel(writer, sheet_name='Cleaned_A', index=False)
                if 'Cleaned Dataset B' in export_options and st.session_state.cleaned_b is not None:
                    st.session_state.cleaned_b.to_excel(writer, sheet_name='Cleaned_B', index=False)
                if 'Duplicates' in export_options:
                    for ds_name in ['cleaned_a', 'cleaned_b']:
                        df = st.session_state.get(ds_name)
                        if df is not None:
                            duplicates = df[df.duplicated(keep=False)]
                            if not duplicates.empty:
                                duplicates.to_excel(writer, sheet_name=f'Duplicates_{ds_name[-1].upper()}', index=False)
                if 'Outliers' in export_options:
                    for ds_name in ['cleaned_a', 'cleaned_b']:
                        df = st.session_state.get(ds_name)
                        if df is not None:
                            outliers = detect_outliers(df)
                            if not outliers.empty:
                                outliers.to_excel(writer, sheet_name=f'Outliers_{ds_name[-1].upper()}', index=False)
            st.download_button("Download Excel Report", data=buffer.getvalue(),
                               file_name=file_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.success(f"Exported: {', '.join(export_options)}")
# ----------------- Tab 6: PDF Report -----------------
with tab6:
    st.header("Generate PDF Report")

    dataset_choice = st.selectbox(
        "Select Dataset for PDF Report",
        ["Dataset A", "Dataset B"]
    )
    df = st.session_state.cleaned_a if dataset_choice == "Dataset A" else st.session_state.cleaned_b
    uploaded_file = st.session_state.uploaded_file_a if dataset_choice == "Dataset A" else st.session_state.uploaded_file_b
    dataset_name = uploaded_file.name if uploaded_file else dataset_choice

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
                "Correlation Matrix",
                "Charts",
                "Insights Paragraph"
            ],
            default=["Data Overview", "Insights Paragraph"]
        )
        top_n = st.number_input("Top N categories for categorical columns", min_value=1, max_value=20, value=5)

        if st.button("Generate PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, f"{dataset_name} - DataLens Report", ln=True, align="C")
            pdf.ln(10)

            # Data Overview
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

            # Descriptive Statistics
            if "Descriptive Statistics" in pdf_sections:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Descriptive Statistics", ln=True)
                stats = df.describe(include='all').transpose().head(20)
                pdf.set_font("Arial", '', 8)
                pdf.multi_cell(0, 5, stats.to_string())
                pdf.ln(5)

            # Top N categories
            if "Top N Categories" in pdf_sections:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, f"Top {top_n} Categories", ln=True)
                pdf.set_font("Arial", '', 10)
                for col in df.select_dtypes(include=['object']).columns:
                    top_vals = df[col].value_counts().head(top_n)
                    pdf.multi_cell(0, 5, f"{col}:\n{top_vals.to_string()}")
                pdf.ln(5)

            # Insights paragraph
            if "Insights Paragraph" in pdf_sections:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Insights", ln=True)
                pdf.set_font("Arial", '', 10)
                n_rows, n_cols = df.shape
                missing_cols = missing[missing>0].to_dict()
                top_cat_summary = []
                for col in df.select_dtypes(include=['object']).columns:
                    val = df[col].value_counts().head(1)
                    if not val.empty:
                        top_cat_summary.append(f"{col} → {val.index[0]} ({val.iloc[0]})")
                corr = df.select_dtypes(include=['number']).corr()
                high_corr = ""
                if not corr.empty:
                    corr_unstacked = corr.abs().unstack()
                    corr_unstacked = corr_unstacked[corr_unstacked < 1]
                    if not corr_unstacked.empty:
                        top_corr = corr_unstacked.sort_values(ascending=False).head(1)
                        high_corr = f"Highest numeric correlation: {top_corr.index[0][0]} vs {top_corr.index[0][1]} = {top_corr.iloc[0]:.2f}"
                insight_text = f"The dataset has {n_rows} rows and {n_cols} columns.\nColumns with most missing values: {missing_cols}\nDuplicate rows: {duplicates}\nTop categories: {', '.join(top_cat_summary)}\n{high_corr}"
                pdf.multi_cell(0, 5, insight_text)

            # Charts
            if "Charts" in pdf_sections and st.session_state.get("last_plotly_figure") is not None:
                fig = st.session_state["last_plotly_figure"]
                chart_file = "temp_chart.png"
                try:
                    fig.write_image(chart_file)
                    pdf.image(chart_file, w=180)
                    os.remove(chart_file)
                except Exception as e:
                    st.warning(f"Could not export chart: {e}")

            out_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("Download PDF Report", data=out_bytes, file_name=f"{dataset_name}_report.pdf", mime="application/pdf")
            st.success("PDF generated successfully!")
