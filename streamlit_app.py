import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import tempfile

# -----------------------------
# Session state initialization
# -----------------------------
if 'raw_a' not in st.session_state:
    st.session_state.raw_a = None
if 'raw_b' not in st.session_state:
    st.session_state.raw_b = None
if 'cleaned_a' not in st.session_state:
    st.session_state.cleaned_a = None
if 'cleaned_b' not in st.session_state:
    st.session_state.cleaned_b = None
if 'compare_report' not in st.session_state:
    st.session_state.compare_report = None

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Upload", "Clean", "Visualize", "Compare & Contrast", "Export", "PDF Summary"
])

# -----------------------------
# Tab 1: Upload
# -----------------------------
with tab1:
    st.header("Upload Datasets")
    file_a = st.file_uploader("Upload Dataset A", type=['csv', 'xlsx'], key="file_a")
    file_b = st.file_uploader("Upload Dataset B", type=['csv', 'xlsx'], key="file_b")

    if file_a:
        if file_a.name.endswith('.csv'):
            st.session_state.raw_a = pd.read_csv(file_a)
        else:
            st.session_state.raw_a = pd.read_excel(file_a)
        st.success(f"Dataset A loaded: {st.session_state.raw_a.shape[0]} rows, {st.session_state.raw_a.shape[1]} cols")

    if file_b:
        if file_b.name.endswith('.csv'):
            st.session_state.raw_b = pd.read_csv(file_b)
        else:
            st.session_state.raw_b = pd.read_excel(file_b)
        st.success(f"Dataset B loaded: {st.session_state.raw_b.shape[0]} rows, {st.session_state.raw_b.shape[1]} cols")

# -----------------------------
# Tab 2: Clean
# -----------------------------
with tab2:
    st.header("Clean Datasets")
    def clean_dataset(df):
        df = df.copy()
        df.drop_duplicates(inplace=True)
        # Fill missing numeric with median, categorical with mode
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Missing", inplace=True)
        return df

    if st.session_state.raw_a is not None:
        if st.button("Clean Dataset A"):
            st.session_state.cleaned_a = clean_dataset(st.session_state.raw_a)
            st.success("Dataset A cleaned")
    if st.session_state.raw_b is not None:
        if st.button("Clean Dataset B"):
            st.session_state.cleaned_b = clean_dataset(st.session_state.raw_b)
            st.success("Dataset B cleaned")

# -----------------------------
# Tab 3: Visualize
# -----------------------------
with tab3:
    st.header("Visualize Data")
    dataset_choice = st.selectbox("Select dataset", ["Dataset A", "Dataset B"])
    df = st.session_state.cleaned_a if dataset_choice=="Dataset A" else st.session_state.cleaned_b
    if df is not None:
        st.dataframe(df.head())
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select numeric column for histogram", numeric_cols)
            plt.hist(df[col], bins=20)
            st.pyplot(plt.gcf())
            plt.clf()
    else:
        st.warning("Please clean the dataset first.")

# -----------------------------
# Tab 4: Compare & Contrast
# -----------------------------
with tab4:
    st.header("Compare & Contrast")
    if st.session_state.cleaned_a is None or st.session_state.cleaned_b is None:
        st.warning("Upload and clean both datasets first.")
    else:
        common_cols = list(set(st.session_state.cleaned_a.columns).intersection(st.session_state.cleaned_b.columns))
        compare_type = st.selectbox(
            "Select Compare Type",
            ['Row presence check', 'Cell-by-cell comparison', 'Summary compare', 'Schema compare']
        )

        key_col = None
        if compare_type == 'Row presence check' and common_cols:
            key_col = st.selectbox("Select key column for row matching (optional)", common_cols)

        if st.button("Run Compare"):
            df_a = st.session_state.cleaned_a.copy()
            df_b = st.session_state.cleaned_b.copy()
            report = None
            explanation = ""

            if compare_type == 'Row presence check' and key_col:
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
                explanation = f"{shared_cols} columns shared, {len(cols_only_a)} unique to A, {len(cols_only_b)} unique to B."

            elif compare_type == 'Cell-by-cell comparison':
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

            elif compare_type == 'Summary compare':
                summary_a = df_a.describe(include='all').transpose()
                summary_b = df_b.describe(include='all').transpose()
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
                if 'Cleaned Dataset A' in export_options and st.session_state.cleaned_a is not None:
                    st.session_state.cleaned_a.to_excel(writer, sheet_name='Cleaned_A', index=False)
                if 'Cleaned Dataset B' in export_options and st.session_state.cleaned_b is not None:
                    st.session_state.cleaned_b.to_excel(writer, sheet_name='Cleaned_B', index=False)
                if 'Duplicates' in export_options:
                    for ds_name in ['cleaned_a', 'cleaned_b']:
                        df_dup = st.session_state.get(ds_name)
                        if df_dup is not None:
                            duplicates = df_dup[df_dup.duplicated(keep=False)]
                            if not duplicates.empty:
                                duplicates.to_excel(writer, sheet_name=f'Duplicates_{ds_name[-1].upper()}', index=False)
                if 'Outliers' in export_options:
                    for ds_name in ['cleaned_a', 'cleaned_b']:
                        df_out = st.session_state.get(ds_name)
                        if df_out is not None:
                            outliers = detect_outliers(df_out)
                            if not outliers.empty:
                                outliers.to_excel(writer, sheet_name=f'Outliers_{ds_name[-1].upper()}', index=False)
            st.download_button("Download Excel Report", data=buffer.getvalue(), file_name=file_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.success(f"Exported: {', '.join(export_options)}")

# -----------------------------
# Tab 6: PDF Summary
# -----------------------------
with tab6:
    st.header("Generate PDF Report")
    include_corr = st.checkbox("Include Correlation Heatmap", value=True)
    include_charts = st.checkbox("Include Charts/Visuals", value=False)
    dataset_choice = st.selectbox("Select dataset for PDF", ["Dataset A", "Dataset B"])
    df = st.session_state.cleaned_a if dataset_choice=="Dataset A" else st.session_state.cleaned_b

    if df is None:
        st.warning("Please upload and clean the selected dataset first.")
    else:
        if st.button("Generate PDF"):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, f"Automated Data Report - {dataset_choice}", ln=True, align='C')
            pdf.set_font("Arial", '', 12)
            pdf.ln(10)
            pdf.multi_cell(0, 6, f"Rows: {df.shape[0]}  Columns: {df.shape[1]}")

            # Dataset Overview
            pdf.set_font("Arial", 'B', 12)
            pdf.ln(5)
            pdf.cell(0, 10, "Dataset Overview", ln=True)
            pdf.set_font("Arial", '', 10)
            missing = df.isna().sum()
            for col in df.columns:
                pdf.multi_cell(0, 5, f"{col} ({df[col].dtype}): missing {missing[col]} ({missing[col]/len(df)*100:.1f}%)")

            # Descriptive stats
            pdf.set_font("Arial", 'B', 12)
            pdf.ln(5)
            pdf.cell(0, 10, "Descriptive Statistics", ln=True)
            pdf.set_font("Arial", '', 9)
            desc = df.describe().transpose()
            pdf.multi_cell(0, 5, desc.to_string())

            # Automatic insights
            pdf.ln(2)
            insights = []
            numeric_cols = df.select_dtypes(include='number').columns
            for col in numeric_cols:
                median = desc.loc[col, '50%']
                max_val = desc.loc[col, 'max']
                if max_val > median * 3:
                    insights.append(f"Column `{col}` has a max ({max_val}) much larger than median ({median}) — possible outlier.")
            if insights:
                pdf.set_font("Arial", 'I', 10)
                pdf.multi_cell(0, 5, "Insights:\n" + "\n".join(insights))

            # Optional: Correlation heatmap
            if include_corr and numeric_cols.any():
                pdf.add_page()
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Correlation Heatmap", ln=True)
                corr = df[numeric_cols].corr()
                plt.figure(figsize=(6,5))
                plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
                plt.colorbar()
                plt.xticks(range(len(corr)), corr.columns, rotation=90)
                plt.yticks(range(len(corr)), corr.columns)
                plt.tight_layout()
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    plt.savefig(tmpfile.name, format='png')
                    tmpfile_path = tmpfile.name
                plt.close()
                pdf.image(tmpfile_path, x=10, w=180)
                pdf.set_font("Arial", 'I', 10)
                strong_corrs = []
                for i in corr.columns:
                    for j in corr.columns:
                        if i != j and abs(corr.loc[i,j]) > 0.7:
                            strong_corrs.append(f"{i} ↔ {j} (r={corr.loc[i,j]:.2f})")
                if strong_corrs:
                    pdf.multi_cell(0, 5, "Strong correlations:\n" + "\n".join(strong_corrs))

            # Optional: Charts placeholder
            if include_charts:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Charts / Visuals", ln=True)
                st.warning("Chart visuals not yet implemented for PDF; placeholder added.")

            # Save PDF to buffer
            buffer = io.BytesIO()
            pdf.output(buffer)
            buffer.seek(0)
            st.download_button("Download PDF Report", data=buffer, file_name=f"{dataset_choice}_Data_Report.pdf", mime="application/pdf")
            st.success("PDF generated successfully!")
