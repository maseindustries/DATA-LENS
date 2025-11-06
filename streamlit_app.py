import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import plotly.express as px
import shap

# -----------------------------
# Session State Defaults
# -----------------------------
if 'cleaned_a' not in st.session_state:
    st.session_state.cleaned_a = None
if 'cleaned_b' not in st.session_state:
    st.session_state.cleaned_b = None
if 'compare_report' not in st.session_state:
    st.session_state.compare_report = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None

# -----------------------------
# Mode Selection
# -----------------------------
st.title("DataLens - Flexible Analysis & Comparison")
mode = st.selectbox("Select Mode", ["Single Dataset Analysis", "Compare Datasets"])

# -----------------------------
# Dataset Upload / Cleaning
# -----------------------------
with st.expander("Upload / Clean Datasets"):
    uploaded_a = st.file_uploader("Upload Dataset A", type=['csv', 'xlsx'], key="upload_a")
    if uploaded_a:
        st.session_state.cleaned_a = pd.read_csv(uploaded_a) if uploaded_a.name.endswith('.csv') else pd.read_excel(uploaded_a)

    uploaded_b = st.file_uploader("Upload Dataset B", type=['csv', 'xlsx'], key="upload_b")
    if uploaded_b:
        st.session_state.cleaned_b = pd.read_csv(uploaded_b) if uploaded_b.name.endswith('.csv') else pd.read_excel(uploaded_b)

# -----------------------------
# Single Dataset Mode
# -----------------------------
if mode == "Single Dataset Analysis":
    dataset_choice = st.selectbox("Select dataset to analyze", ["Dataset A", "Dataset B"])
    df = st.session_state.cleaned_a if dataset_choice == "Dataset A" else st.session_state.cleaned_b

    if df is not None:
        # ---- Tab 1: Quick Info ----
        st.subheader("Dataset Overview")
        st.dataframe(df.head())
        st.write("Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())

        # ---- Tab 2: Filtering ----
        st.subheader("Filter Dataset")
        filters = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val, max_val = float(df[col].min()), float(df[col].max())
                filters[col] = st.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 20:
                options = df[col].unique().tolist()
                filters[col] = st.multiselect(f"Filter {col}", options, default=options)
        # Apply filters
        df_filtered = df.copy()
        for col, val in filters.items():
            if isinstance(val, tuple):
                df_filtered = df_filtered[(df_filtered[col] >= val[0]) & (df_filtered[col] <= val[1])]
            else:
                df_filtered = df_filtered[df_filtered[col].isin(val)]
        st.dataframe(df_filtered)

        # ---- Tab 3: Visualization ----
        st.subheader("Visualization")
        chart_type = st.selectbox("Select chart type", ["Histogram", "Scatter", "Box"])
        if chart_type == "Histogram":
            col = st.selectbox("Select numeric column", df_filtered.select_dtypes(include=np.number).columns)
            fig = px.histogram(df_filtered, x=col, nbins=30)
            st.plotly_chart(fig)
        elif chart_type == "Scatter":
            numeric_cols = df_filtered.select_dtypes(include=np.number).columns
            x_col = st.selectbox("X axis", numeric_cols)
            y_col = st.selectbox("Y axis", numeric_cols)
            fig = px.scatter(df_filtered, x=x_col, y=y_col)
            st.plotly_chart(fig)
        elif chart_type == "Box":
            numeric_cols = df_filtered.select_dtypes(include=np.number).columns
            col = st.selectbox("Select column", numeric_cols)
            fig = px.box(df_filtered, y=col)
            st.plotly_chart(fig)

        # ---- Tab 5: Modeling ----
        st.subheader("Modeling & Prediction")
        target_column = st.selectbox("Select target column", df_filtered.columns)
        feature_columns = st.multiselect(
            "Select features", [c for c in df_filtered.columns if c != target_column], 
            default=[c for c in df_filtered.columns if c != target_column]
        )
        model_choice = st.selectbox("Model", ["Random Forest", "Logistic Regression"])
        if st.button("Train Model"):
            X = pd.get_dummies(df_filtered[feature_columns])
            y = df_filtered[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            if model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=500)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            st.session_state.model_metrics = pd.DataFrame({'Metric': ['Accuracy', 'RMSE'], 'Value': [acc, rmse]})
            st.session_state.model = model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.dataframe(st.session_state.model_metrics)

        # ---- Tab 6: Explainability ----
        st.subheader("Explainability")
        if st.button("Show Feature Importance"):
            if st.session_state.model:
                explainer = shap.TreeExplainer(st.session_state.model)
                shap_values = explainer.shap_values(st.session_state.X_test)
                st.write("SHAP Summary Plot")
                shap.summary_plot(shap_values, st.session_state.X_test, plot_type="bar")
            else:
                st.warning("Train a model first!")

# -----------------------------
# Compare Datasets Mode
# -----------------------------
elif mode == "Compare Datasets":
    if st.session_state.cleaned_a is None or st.session_state.cleaned_b is None:
        st.warning("Please upload both datasets first")
    else:
        st.subheader("Compare & Contrast")
        common_cols = list(set(st.session_state.cleaned_a.columns).intersection(st.session_state.cleaned_b.columns))
        compare_type = st.selectbox("Select Compare Type", ['Row presence check', 'Cell-by-cell comparison', 'Summary compare', 'Schema compare'])

        key_col = None
        if compare_type == 'Row presence check' and common_cols:
            key_col = st.selectbox("Key column for row matching (optional)", common_cols)

        if st.button("Run Compare"):
            report = None
            if compare_type == 'Row presence check' and key_col:
                ids_a = set(st.session_state.cleaned_a[key_col])
                ids_b = set(st.session_state.cleaned_b[key_col])
                report = pd.DataFrame({
                    f'Only in A ({key_col})': list(ids_a - ids_b),
                    f'Only in B ({key_col})': list(ids_b - ids_a)
                })
            elif compare_type == 'Schema compare':
                cols_a = set(st.session_state.cleaned_a.columns)
                cols_b = set(st.session_state.cleaned_b.columns)
                report = pd.DataFrame({
                    'Columns only in A': list(cols_a - cols_b),
                    'Columns only in B': list(cols_b - cols_a)
                })
            elif compare_type == 'Cell-by-cell comparison':
                diffs = []
                max_len = max(len(st.session_state.cleaned_a), len(st.session_state.cleaned_b))
                df_a = st.session_state.cleaned_a.reindex(range(max_len))
                df_b = st.session_state.cleaned_b.reindex(range(max_len))
                for col in common_cols:
                    col_a = df_a[col]
                    col_b = df_b[col]
                    if pd.api.types.is_numeric_dtype(col_a):
                        diffs.append({'Column': col, 'Mean Difference': col_a.mean()-col_b.mean(), 'Count Differences': (col_a != col_b).sum()})
                    else:
                        diffs.append({'Column': col, 'Mismatched Count': (col_a != col_b).sum()})
                report = pd.DataFrame(diffs)
            else:
                summary_a = st.session_state.cleaned_a.describe(include='all')
                summary_b = st.session_state.cleaned_b.describe(include='all')
                report = pd.concat([summary_a, summary_b], keys=['Dataset A', 'Dataset B'])
            
            st.session_state.compare_report = report
            st.dataframe(report)
            buffer = io.BytesIO()
            report.to_excel(buffer)
            st.download_button("Export Comparison Report", data=buffer, file_name="comparison_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
