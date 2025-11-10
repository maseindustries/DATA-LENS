# data_lens_pro_with_lastfig.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import tempfile
import os
from datetime import datetime
import plotly.express as px
from fpdf import FPDF

st.set_page_config(layout="wide", page_title="DataLens Pro")
st.title("DataLens Pro — Interactive Data QA & EDA")

# ----------------------------
# Session state defaults
# ----------------------------
keys = [
    "file_a_name", "file_b_name",
    "original_a", "original_b",
    "cleaned_a", "cleaned_b",
    "preview_a", "preview_b",
    "dupes_a", "dupes_b",
    "outliers_a", "outliers_b",
    "compare_report",
    "last_plotly_figure"
]
for k in keys:
    if k not in st.session_state:
        st.session_state[k] = None

# ----------------------------
# Helper functions
# ----------------------------
def read_file(uploaded):
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    elif name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded)
    else:
        st.error("Unsupported file type. Use CSV or Excel.")
        return None

def detect_internal_duplicates(df):
    if df is None:
        return pd.DataFrame()
    return df[df.duplicated(keep=False)].copy()

def detect_outliers_iqr(df):
    if df is None:
        return pd.DataFrame()
    numeric_cols = df.select_dtypes(include=['number']).columns
    out = pd.DataFrame()
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if pd.isna(IQR) or IQR == 0:
            continue
        mask = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
        if mask.any():
            out = pd.concat([out, df[mask]])
    return out.drop_duplicates()

def compute_cleaning_changes(before_df, after_df):
    changes = []
    if before_df is None or after_df is None:
        return changes
    if before_df.shape != after_df.shape:
        changes.append(f"Shape {before_df.shape} → {after_df.shape}")
    dropped = set(before_df.columns) - set(after_df.columns)
    if dropped:
        changes.append(f"Dropped columns: {sorted(list(dropped))}")
    num_cols = [c for c in before_df.select_dtypes(include=['number']).columns if c in after_df.columns]
    for col in num_cols:
        before_na = before_df[col].isna().sum()
        after_na = after_df[col].isna().sum()
        if after_na < before_na:
            changes.append(f"Filled {before_na - after_na} missing numeric values in '{col}'.")
    cat_cols = [c for c in before_df.select_dtypes(include=['object', 'category']).columns if c in after_df.columns]
    for col in cat_cols:
        before_na = before_df[col].isna().sum()
        after_na = after_df[col].isna().sum()
        if after_na < before_na:
            changes.append(f"Filled {before_na - after_na} missing categorical values in '{col}'.")
    for col in cat_cols:
        sample_before = before_df[col].dropna().astype(str).head(50).tolist()
        sample_after = after_df[col].dropna().astype(str).head(50).tolist()
        if sample_before != sample_after:
            changes.append(f"Trimmed whitespace in '{col}' (sample evidence).")
    return changes

def file_title(name):
    return name if name else "Untitled Dataset"

def safe_cols(df, dtype=None):
    if df is None:
        return []
    if dtype == 'numeric':
        return df.select_dtypes(include=['number']).columns.tolist()
    elif dtype == 'categorical':
        return df.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        return df.columns.tolist()

def fig_to_png_bytes(fig):
    try:
        img = fig.to_image(format="png", scale=2)
        return img
    except Exception:
        return None

def generate_insights_paragraph(df, top_n=3):
    if df is None or df.empty:
        return "No data available to generate insights."
    insights = []
    nrows, ncols = df.shape
    insights.append(f"The dataset has {nrows} rows and {ncols} columns.")
    missing = df.isna().sum()
    mv = missing[missing > 0].sort_values(ascending=False)
    if not mv.empty:
        top = mv.head(3)
        kv = "; ".join([f"{idx}: {int(val)} missing" for idx, val in top.items()])
        insights.append(f"Columns with most missing values — {kv}.")
    else:
        insights.append("No missing values detected.")
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        insights.append(f"There are {dup_count} internal duplicate rows (see duplicate export).")
    cat_cols = safe_cols(df, dtype='categorical')
    if cat_cols:
        top_cat = []
        for col in cat_cols[:3]:
            vc = df[col].value_counts().head(1)
            if not vc.empty:
                top_cat.append(f"{col} → {vc.index[0]} ({int(vc.iloc[0])})")
        if top_cat:
            insights.append("Top categories: " + "; ".join(top_cat))
    num_cols = safe_cols(df, dtype='numeric')
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().stack().abs().sort_values(ascending=False)
        corr = corr[corr < 1.0]
        if not corr.empty:
            top_corr = corr.head(top_n)
            pair = top_corr.index[0]
            insights.append(f"Highest numeric correlation: {pair[0]} vs {pair[1]} = {top_corr.iloc[0]:.2f}.")
    return " ".join(insights)

# ----------------------------
# Layout - Tabs
# ----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Upload", "Cleaning", "EDA", "Compare & Contrast", "Export", "PDF Report"
])

# ----------------------------
# Tab 1: Upload
# ----------------------------
with tab1:
    st.header("Upload Datasets (one or two files)")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_a = st.file_uploader("Upload primary dataset (required)", type=["csv", "xlsx"], key="u_a")
    with col2:
        uploaded_b = st.file_uploader("Upload secondary dataset (optional)", type=["csv", "xlsx"], key="u_b")

    if uploaded_a:
        df_a = read_file(uploaded_a)
        st.session_state.original_a = df_a.copy()
        st.session_state.cleaned_a = df_a.copy()
        st.session_state.file_a_name = uploaded_a.name
        st.success(f"Loaded: {uploaded_a.name} — shape {df_a.shape}")

    if uploaded_b:
        df_b = read_file(uploaded_b)
        st.session_state.original_b = df_b.copy()
        st.session_state.cleaned_b = df_b.copy()
        st.session_state.file_b_name = uploaded_b.name
        st.success(f"Loaded: {uploaded_b.name} — shape {df_b.shape}")

    st.markdown("---")
    def compact_summary(df, filename):
        if df is None:
            return
        st.markdown(f"**{file_title(filename)}** — rows: {df.shape[0]}, cols: {df.shape[1]}")
        dtypes = pd.DataFrame(df.dtypes, columns=["dtype"])
        missing = df.isna().sum().rename("missing")
        summary = dtypes.join(missing)
        st.dataframe(summary, use_container_width=True, height=300)

    compact_summary(st.session_state.cleaned_a, st.session_state.file_a_name)
    compact_summary(st.session_state.cleaned_b, st.session_state.file_b_name)

# ----------------------------
# Tab 2: Cleaning (Preview & Apply)
# ----------------------------
with tab2:
    st.header("Cleaning — Preview changes before applying")
    st.write("Choose cleaning options, preview results, then Apply to commit changes. You can Reset to original anytime.")

    opts = st.multiselect(
        "Cleaning operations",
        [
            "Drop duplicate rows",
            "Fill missing numeric values with median",
            "Fill missing categorical values with mode",
            "Trim whitespace from string columns",
            "Remove columns with all nulls"
        ],
        default=["Drop duplicate rows"]
    )
    cols = st.columns([1,1,1])
    with cols[0]:
        if st.button("Preview Changes"):
            for label, key in [("a", "cleaned_a"), ("b", "cleaned_b")]:
                df = st.session_state.get(key)
                if df is None:
                    st.session_state[f"preview_{label}"] = None
                    continue
                preview = df.copy()
                if "Drop duplicate rows" in opts:
                    preview = preview.drop_duplicates()
                if "Fill missing numeric values with median" in opts:
                    for c in preview.select_dtypes(include=['number']).columns:
                        if preview[c].isna().sum() > 0:
                            preview[c] = preview[c].fillna(preview[c].median())
                if "Fill missing categorical values with mode" in opts:
                    for c in preview.select_dtypes(include=['object', 'category']).columns:
                        if preview[c].isna().sum() > 0 and not preview[c].mode().empty:
                            preview[c] = preview[c].fillna(preview[c].mode()[0])
                if "Trim whitespace from string columns" in opts:
                    for c in preview.select_dtypes(include=['object']).columns:
                        preview[c] = preview[c].apply(lambda x: x.strip() if isinstance(x, str) else x)
                if "Remove columns with all nulls" in opts:
                    preview = preview.dropna(axis=1, how='all')
                st.session_state[f"preview_{label}"] = preview
            st.success("Preview generated. Inspect below and Apply or Reset as needed.")

    with cols[1]:
        if st.button("Apply Changes"):
            applied = False
            for label, key in [("a", "cleaned_a"), ("b", "cleaned_b")]:
                preview = st.session_state.get(f"preview_{label}")
                if preview is not None:
                    before = st.session_state.get(key)
                    st.session_state[key] = preview
                    st.session_state[f"dupes_{label}"] = detect_internal_duplicates(preview)
                    st.session_state[f"outliers_{label}"] = detect_outliers_iqr(preview)
                    applied = True
            if applied:
                st.success("Applied preview changes to session.")
            else:
                st.info("No preview available to apply. Use 'Preview Changes' first.")

    with cols[2]:
        if st.button("Reset to Original"):
            if st.session_state.original_a is not None:
                st.session_state.cleaned_a = st.session_state.original_a.copy()
                st.session_state.preview_a = None
            if st.session_state.original_b is not None:
                st.session_state.cleaned_b = st.session_state.original_b.copy()
                st.session_state.preview_b = None
            st.success("Reset cleaned datasets to original uploaded versions.")

    st.markdown("---")
    for label, filename_key in [("a", "file_a_name"), ("b", "file_b_name")]:
        filename = st.session_state.get(filename_key)
        cleaned = st.session_state.get(f"cleaned_{label}")
        preview = st.session_state.get(f"preview_{label}")
        if cleaned is None and preview is None:
            continue
        st.subheader(f"{file_title(filename)} — Current vs Preview")
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Current (cleaned)**")
            if cleaned is not None:
                st.dataframe(cleaned.head(5), use_container_width=True)
            else:
                st.info("No current dataset.")
        with col_right:
            st.markdown("**Preview (after selected ops)**")
            if preview is not None:
                st.dataframe(preview.head(5), use_container_width=True)
                changes = compute_cleaning_changes(cleaned, preview)
                if changes:
                    st.write("Preview change highlights:")
                    for c in changes:
                        st.markdown(f"- {c}")
                else:
                    st.info("No notable changes detected in preview.")
            else:
                st.info("No preview generated yet.")

# ----------------------------
# Tab 3: EDA (Reactive)
# ----------------------------
with tab3:
    st.header("Exploratory Data Analysis — reactive")
    ds_choice = st.selectbox("Select dataset to analyze", options=[
        file_title(st.session_state.file_a_name) if st.session_state.cleaned_a is not None else None,
        file_title(st.session_state.file_b_name) if st.session_state.cleaned_b is not None else None
    ], format_func=lambda x: x if x else "No dataset", key="eda_ds")

    df_map = {}
    if st.session_state.cleaned_a is not None:
        df_map[file_title(st.session_state.file_a_name)] = st.session_state.cleaned_a
    if st.session_state.cleaned_b is not None:
        df_map[file_title(st.session_state.file_b_name)] = st.session_state.cleaned_b

    df = df_map.get(ds_choice)
    if df is None:
        st.warning("Please upload and clean a dataset first.")
    else:
        st.subheader(generate_insights_paragraph(df, top_n=3))
        left, right = st.columns([2, 1])

        with left:
            st.markdown("### Charts")
            chart_opt = st.selectbox("Chart type", ["Histogram", "Boxplot", "Scatter", "Correlation Heatmap"])
            if chart_opt in ("Histogram", "Boxplot"):
                num_cols = safe_cols(df, dtype='numeric')
                if not num_cols:
                    st.warning("No numeric columns available for this chart.")
                else:
                    col = st.selectbox("Select numeric column", num_cols, key=f"eda_num_{ds_choice}_{chart_opt}")
                    if chart_opt == "Histogram":
                        fig = px.histogram(df, x=col, nbins=40, title=f"{col} distribution")
                        st.plotly_chart(fig, use_container_width=True)
                        st.session_state['last_plotly_figure'] = fig
                    else:
                        fig = px.box(df, y=col, title=f"{col} boxplot")
                        st.plotly_chart(fig, use_container_width=True)
                        st.session_state['last_plotly_figure'] = fig
            elif chart_opt == "Scatter":
                all_cols = safe_cols(df)
                num_cols = safe_cols(df, dtype='numeric')
                if not all_cols or not num_cols:
                    st.warning("Need at least one numeric and one other column for scatter.")
                else:
                    x = st.selectbox("X (any column)", all_cols, key=f"scatter_x_{ds_choice}")
                    y = st.selectbox("Y (numeric)", num_cols, key=f"scatter_y_{ds_choice}")
                    fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}", hover_data=df.columns.tolist())
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state['last_plotly_figure'] = fig
            else:
                num_cols = safe_cols(df, dtype='numeric')
                if len(num_cols) < 2:
                    st.warning("Need at least two numeric columns for correlation heatmap.")
                else:
                    corr = df[num_cols].corr()
                    fig = px.imshow(corr, text_auto=True, title="Correlation matrix")
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state['last_plotly_figure'] = fig

        with right:
            st.markdown("### Top N insights")
            n = st.number_input("Top N (categorical)", min_value=1, max_value=20, value=5, key="topn_cats")
            cat_cols = safe_cols(df, dtype='categorical')
            if cat_cols:
                for c in cat_cols[:5]:
                    st.markdown(f"**{c}** (top {n})")
                    st.dataframe(df[c].value_counts().head(n).rename_axis(c).reset_index(name="count"), use_container_width=True)
            else:
                st.info("No categorical columns to display top categories for.")
            st.markdown("### Quick stats (numeric)")
            if safe_cols(df, dtype='numeric'):
                st.dataframe(df[safe_cols(df, dtype='numeric')].describe().transpose(), use_container_width=True)
            else:
                st.info("No numeric columns.")

# ----------------------------
# Tab 4: Compare & Contrast
# ----------------------------
with tab4:
    st.header("Compare & Contrast")
    df_a = st.session_state.cleaned_a
    df_b = st.session_state.cleaned_b
    a_name = file_title(st.session_state.file_a_name)
    b_name = file_title(st.session_state.file_b_name)

    if df_a is None and df_b is None:
        st.warning("Upload at least one dataset.")
    else:
        compare_types = ['Schema compare', 'Summary compare']
        if df_a is not None and df_b is not None:
            compare_types = ['Row presence check', 'Cell-by-cell comparison'] + compare_types
        compare_type = st.selectbox("Comparison mode", compare_types)

        if compare_type == 'Row presence check' and df_a is not None and df_b is not None:
            possible_keys = list(set(df_a.columns).intersection(df_b.columns))
            key = st.selectbox("Select key column for row matching (suggested)", options=possible_keys)
            if st.button("Run Row Presence Check"):
                ids_a = set(df_a[key].dropna().unique())
                ids_b = set(df_b[key].dropna().unique())
                only_a = sorted(list(ids_a - ids_b))
                only_b = sorted(list(ids_b - ids_a))
                overlap = len(ids_a.intersection(ids_b))
                total = len(ids_a.union(ids_b))
                sim = (overlap / total * 100) if total else 0
                st.metric("Row Match Rate", f"{sim:.1f}%")
                max_len = max(len(only_a), len(only_b))
                only_a += [None] * (max_len - len(only_a))
                only_b += [None] * (max_len - len(only_b))
                report = pd.DataFrame({f"Only in {a_name} ({key})": only_a, f"Only in {b_name} ({key})": only_b})
                st.session_state.compare_report = report
                st.dataframe(report, use_container_width=True)
                st.write(f"{len(only_a)} unique in {a_name}, {len(only_b)} unique in {b_name}, {overlap} overlap.")

        elif compare_type == 'Cell-by-cell comparison' and df_a is not None and df_b is not None:
            if st.button("Run Cell-by-cell Comparison"):
                common = list(set(df_a.columns).intersection(df_b.columns))
                diffs = []
                total_diff = 0
                for c in common:
                    n = min(len(df_a), len(df_b))
                    acol = df_a[c].iloc[:n].reset_index(drop=True)
                    bcol = df_b[c].iloc[:n].reset_index(drop=True)
                    mismatch = (acol.fillna("__NA__") != bcol.fillna("__NA__"))
                    diff_count = int(mismatch.sum())
                    total_diff += diff_count
                    if pd.api.types.is_numeric_dtype(acol):
                        try:
                            mean_diff = round(float(acol.mean()) - float(bcol.mean()), 6)
                        except Exception:
                            mean_diff = None
                        diffs.append({'Column': c, 'Type': 'Numeric', 'Mean Diff': mean_diff, 'Mismatched': diff_count})
                    else:
                        diffs.append({'Column': c, 'Type': 'Categorical', 'Mean Diff': None, 'Mismatched': diff_count})
                report = pd.DataFrame(diffs).sort_values('Mismatched', ascending=False)
                st.session_state.compare_report = report
                st.bar_chart(report.set_index('Column')["Mismatched"])
                st.dataframe(report, use_container_width=True)
                st.write(f"Total differing cells (over first {min(len(df_a), len(df_b))} rows): {total_diff}")

        elif compare_type == 'Schema compare':
            cols_a = set(df_a.columns) if df_a is not None else set()
            cols_b = set(df_b.columns) if df_b is not None else set()
            only_a = sorted(list(cols_a - cols_b))
            only_b = sorted(list(cols_b - cols_a))
            max_len = max(len(only_a), len(only_b))
            only_a += [None] * (max_len - len(only_a))
            only_b += [None] * (max_len - len(only_b))
            report = pd.DataFrame({f"Only in {a_name}": only_a, f"Only in {b_name}": only_b})
            st.session_state.compare_report = report
            st.metric("Schema overlap", f"{len(cols_a & cols_b)} shared / {len(cols_a | cols_b)} total")
            st.dataframe(report, use_container_width=True)

        elif compare_type == 'Summary compare':
            summary_a = df_a.describe(include='all').transpose() if df_a is not None else pd.DataFrame()
            summary_b = df_b.describe(include='all').transpose() if df_b is not None else pd.DataFrame()
            combined = summary_a.join(summary_b, lsuffix=f'_{a_name}', rsuffix=f'_{b_name}', how='outer')
            st.session_state.compare_report = combined
            st.dataframe(combined, use_container_width=True)
            st.write("Side-by-side summary statistics.")

# ----------------------------
# Tab 5: Export
# ----------------------------
with tab5:
    st.header("Export cleaned datasets, duplicates, outliers, and charts")
    export_items = st.multiselect(
        "Select items to include in Excel",
        options=[
            f"Cleaned - {file_title(st.session_state.file_a_name)}" if st.session_state.cleaned_a is not None else None,
            f"Cleaned - {file_title(st.session_state.file_b_name)}" if st.session_state.cleaned_b is not None else None,
            "Duplicates (separate sheets)",
            "Outliers (separate sheets)"
        ],
        default=[]
    )
    export_items = [i for i in export_items if i is not None]
    filename = st.text_input("Export filename", value=f"DataLens_Export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

    if st.button("Export to Excel"):
        if not export_items:
            st.warning("Pick at least one item to export.")
        else:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                if f"Cleaned - {file_title(st.session_state.file_a_name)}" in export_items and st.session_state.cleaned_a is not None:
                    st.session_state.cleaned_a.to_excel(writer, sheet_name='Cleaned_A', index=False)
                if f"Cleaned - {file_title(st.session_state.file_b_name)}" in export_items and st.session_state.cleaned_b is not None:
                    st.session_state.cleaned_b.to_excel(writer, sheet_name='Cleaned_B', index=False)
                if "Duplicates (separate sheets)" in export_items:
                    if st.session_state.dupes_a is not None and not st.session_state.dupes_a.empty:
                        st.session_state.dupes_a.to_excel(writer, sheet_name='Duplicates_A', index=False)
                    if st.session_state.dupes_b is not None and not st.session_state.dupes_b.empty:
                        st.session_state.dupes_b.to_excel(writer, sheet_name='Duplicates_B', index=False)
                if "Outliers (separate sheets)" in export_items:
                    if st.session_state.outliers_a is not None and not st.session_state.outliers_a.empty:
                        st.session_state.outliers_a.to_excel(writer, sheet_name='Outliers_A', index=False)
                    if st.session_state.outliers_b is not None and not st.session_state.outliers_b.empty:
                        st.session_state.outliers_b.to_excel(writer, sheet_name='Outliers_B', index=False)
                writer.save()
                buf.seek(0)
            st.download_button("Download Excel", data=buf.getvalue(), file_name=filename,
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.success("Export prepared.")

    st.markdown("---")
    st.write("Optionally export the currently visible chart as a PNG (requires plotly + kaleido).")
    if st.button("Export last chart as PNG"):
        st.info("This app attempts to export the last generated chart. If no chart was displayed this run, nothing will be exported.")
        last_fig = st.session_state.get("last_plotly_figure")
        if last_fig is None:
            st.warning("No chart to export for this session. Generate a chart in EDA first.")
        else:
            png = fig_to_png_bytes(last_fig)
            if png is None:
                st.error("PNG export failed. Install or enable 'kaleido' to allow exporting plotly figures to PNG.")
            else:
                name = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                st.download_button("Download chart PNG", data=png, file_name=name, mime="image/png")

# ----------------------------
# Tab 6: PDF Report with charts & insights
# ----------------------------
with tab6:
    st.header("PDF Report — customizable with charts and auto-insights")
    ds_select = st.selectbox("Choose dataset for PDF", options=[
        file_title(st.session_state.file_a_name) if st.session_state.cleaned_a is not None else None,
        file_title(st.session_state.file_b_name) if st.session_state.cleaned_b is not None else None
    ], format_func=lambda x: x if x else "No dataset", key="pdf_ds")

    df_map = {}
    if st.session_state.cleaned_a is not None:
        df_map[file_title(st.session_state.file_a_name)] = st.session_state.cleaned_a
    if st.session_state.cleaned_b is not None:
        df_map[file_title(st.session_state.file_b_name)] = st.session_state.cleaned_b

    df_sel = df_map.get(ds_select)
    if df_sel is None:
        st.warning("Please upload and clean a dataset to generate PDF.")
    else:
        sections = st.multiselect(
            "Select sections to include in PDF",
            ["Data Overview", "Descriptive Statistics", "Top N Categories", "Outlier Summary", "Charts", "Correlation Matrix"],
            default=["Data Overview", "Top N Categories"]
        )
        topn = st.number_input("Top N categories", min_value=1, max_value=20, value=5)
        chart_choices = st.multiselect("Choose charts to include", ["Histogram", "Boxplot", "Scatter"], default=["Histogram"])

        if st.button("Generate PDF"):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=12)
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            title = ds_select if ds_select else "Dataset"
            pdf.cell(0, 10, f"{title} — DataLens Report", ln=True, align="C")
            pdf.ln(4)

            insights = generate_insights_paragraph(df_sel, top_n=3)
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 6, "Insights:")
            pdf.set_font("Arial", "", 9)
            pdf.multi_cell(0, 5, insights)
            pdf.ln(3)

            if "Data Overview" in sections:
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 7, "Data Overview", ln=True)
                pdf.set_font("Arial", "", 9)
                pdf.multi_cell(0, 5, f"Rows: {df_sel.shape[0]}, Columns: {df_sel.shape[1]}")
                mv = df_sel.isna().sum()
                mv_str = mv[mv > 0].to_string() if mv.sum() > 0 else "None"
                pdf.multi_cell(0, 5, f"Missing values (non-zero):\n{mv_str}")
                pdf.ln(3)

            if "Descriptive Statistics" in sections:
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 7, "Descriptive Statistics", ln=True)
                pdf.set_font("Courier", "", 8)
                stats = df_sel.describe(include='all').transpose().head(60)
                pdf.multi_cell(0, 5, stats.to_string())
                pdf.ln(3)

            if "Top N Categories" in sections:
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 7, f"Top {topn} Categories", ln=True)
                pdf.set_font("Arial", "", 9)
                cat_cols = safe_cols(df_sel, dtype='categorical')
                if not cat_cols:
                    pdf.cell(0, 5, "No categorical columns found.", ln=True)
                else:
                    for c in cat_cols:
                        top = df_sel[c].value_counts().head(topn)
                        pdf.multi_cell(0, 5, f"{c}:\n{top.to_string()}")
                        pdf.ln(1)

            if "Outlier Summary" in sections:
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 7, "Outlier Summary", ln=True)
                pdf.set_font("Arial", "", 9)
                out = detect_outliers_iqr(df_sel)
                if out.empty:
                    pdf.cell(0, 5, "No outliers detected (IQR method).", ln=True)
                else:
                    pdf.multi_cell(0, 5, f"Found {len(out)} outlier rows. Showing first 10 rows:\n{out.head(10).to_string(index=False)}")
                pdf.ln(2)

            temp_files = []
            if "Charts" in sections:
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 7, "Charts", ln=True)
                numeric_cols = safe_cols(df_sel, dtype='numeric')
                chart_cols = numeric_cols[:6]
                for chart in chart_choices:
                    pdf.set_font("Arial", "B", 11)
                    pdf.cell(0, 6, chart, ln=True)
                    for col in chart_cols:
                        try:
                            if chart == "Histogram":
                                fig = px.histogram(df_sel, x=col, nbins=40, title=f"{col} distribution")
                            elif chart == "Boxplot":
                                fig = px.box(df_sel, y=col, title=f"{col} boxplot")
                            elif chart == "Scatter":
                                other = [c for c in numeric_cols if c != col]
                                if not other:
                                    continue
                                fig = px.scatter(df_sel, x=col, y=other[0], title=f"{other[0]} vs {col}")
                            else:
                                continue
                            png = fig_to_png_bytes(fig)
                            if png is None:
                                pdf.set_font("Arial", "", 9)
                                pdf.cell(0, 6, "PNG generation not available (install kaleido). Skipping chart embedding.", ln=True)
                                continue
                            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                            tf.write(png)
                            tf.flush()
                            tf.close()
                            temp_files.append(tf.name)
                            pdf.image(tf.name, w=pdf.epw)
                            pdf.ln(3)
                        except Exception as e:
                            pdf.set_font("Arial", "", 8)
                            pdf.multi_cell(0, 5, f"Failed to render chart for {col}: {e}")
                            continue

            if "Correlation Matrix" in sections:
                numeric_cols = safe_cols(df_sel, dtype='numeric')
                if len(numeric_cols) < 2:
                    pdf.set_font("Arial", "", 9)
                    pdf.cell(0, 6, "Not enough numeric columns for correlation matrix.", ln=True)
                else:
                    corr = df_sel[numeric_cols].corr()
                    fig = px.imshow(corr, text_auto=True, title="Correlation matrix")
                    png = fig_to_png_bytes(fig)
                    if png is not None:
                        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                        tf.write(png)
                        tf.flush()
                        tf.close()
                        temp_files.append(tf.name)
                        pdf.image(tf.name, w=pdf.epw)
                        pdf.ln(3)
                    else:
                        pdf.cell(0, 6, "PNG export for correlation matrix unavailable (install kaleido).", ln=True)

            out_bytes = pdf.output(dest='S').encode('latin-1')
            for fp in temp_files:
                try:
                    os.remove(fp)
                except Exception:
                    pass

            out_name = f"DataLens_{ds_select.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button("Download PDF Report", data=out_bytes, file_name=out_name, mime="application/pdf")
            st.success("PDF generated.")
