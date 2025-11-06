# -----------------------------
# streamlit_app.py
# DataLens — Smart Data Analysis Tool
# -----------------------------

# Install dependencies if needed (run in terminal or Colab)
# pip install streamlit pandas numpy plotly ydata-profiling scikit-learn shap rapidfuzz openpyxl tqdm

import streamlit as st
import pandas as pd
import io
import plotly.express as px

# -----------------------------
# Initialize session state
# -----------------------------
for key in [
    'dataset_a', 'dataset_b', 'cleaned_a', 'cleaned_b',
    'eda_report_a', 'eda_report_b', 'compare_report',
    'model_metrics', 'shap_plots'
]:
    if key not in st.session_state:
        st.session_state[key] = None

# -----------------------------
# App Title
# -----------------------------
st.title("DataLens — Smart Data Analysis Tool")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Upload", "Cleaning", "EDA", "Compare", "Modeling", "Explainability", "Export"
])

# -----------------------------
# Tab 1: Upload
# -----------------------------
with tab1:
    st.header("Upload Datasets")
    dataset_a_file = st.file_uploader("Upload Dataset A", type=['csv','xlsx'])
    dataset_b_file = st.file_uploader("Upload Dataset B", type=['csv','xlsx'])

    if st.button("Load Datasets"):
        try:
            if dataset_a_file is not None:
                st.session_state.dataset_a = pd.read_csv(dataset_a_file) if dataset_a_file.name.endswith('.csv') else pd.read_excel(dataset_a_file)
            if dataset_b_file is not None:
                st.session_state.dataset_b = pd.read_csv(dataset_b_file) if dataset_b_file.name.endswith('.csv') else pd.read_excel(dataset_b_file)
            st.success("Datasets loaded successfully!")

            if st.session_state.dataset_a is not None:
                st.subheader("Dataset A Preview")
                st.dataframe(st.session_state.dataset_a.head())
            if st.session_state.dataset_b is not None:
                st.subheader("Dataset B Preview")
                st.dataframe(st.session_state.dataset_b.head())

        except Exception as e:
            st.error(f"Error loading datasets: {e}")

# -----------------------------
# Tab 2: Cleaning
# -----------------------------
with tab2:
    st.header("Cleaning & Preparation")
    if st.button("Run Cleaning"):
        st.write("Cleaning placeholder: implement missing value handling, deduplication, renaming, etc.")
        st.session_state.cleaned_a = st.session_state.dataset_a.copy() if st.session_state.dataset_a is not None else None
        st.session_state.cleaned_b = st.session_state.dataset_b.copy() if st.session_state.dataset_b is not None else None

        st.write("Preview of cleaned Dataset A:")
        st.dataframe(st.session_state.cleaned_a.head() if st.session_state.cleaned_a is not None else "No data")
        st.write("Preview of cleaned Dataset B:")
        st.dataframe(st.session_state.cleaned_b.head() if st.session_state.cleaned_b is not None else "No data")


# -----------------------------
# Tab 3: EDA
# -----------------------------
with tab3:
    st.header("Exploratory Data Analysis")
    if st.button("Run EDA"):
        st.write("EDA placeholder: summary stats, histograms, correlations, automated profiling.")

        # Function to display summary and charts for a dataset
        def display_eda(dataset, name, key_prefix):
            if dataset is not None:
                st.write(f"{name} Summary")
                st.dataframe(dataset.describe())

                numeric_cols = dataset.select_dtypes(include='number').columns
                if len(numeric_cols) > 0:
                    for i, col in enumerate(numeric_cols):
                        fig = px.histogram(dataset, x=col, title=f"Histogram of {col}")
                        st.plotly_chart(fig, key=f"{key_prefix}_{i}")  # unique key for each chart

        # Display EDA for both datasets
        display_eda(st.session_state.cleaned_a, "Dataset A", "eda_a")
        display_eda(st.session_state.cleaned_b, "Dataset B", "eda_b")

# -----------------------------
# Tab 4: Compare & Contrast
# -----------------------------
with tab4:
    st.header("Compare & Contrast")
    compare_type = st.selectbox("Select Compare Type", ['Row presence check', 'Cell-by-cell comparison', 'Summary compare', 'Schema compare'])
    if st.button("Run Compare"):
        if st.session_state.cleaned_a is None or st.session_state.cleaned_b is None:
            st.warning("Please run cleaning or upload both datasets first")
        else:
            st.write(f"Compare placeholder: running {compare_type}")
            # Placeholder: store a dummy compare report
            st.session_state.compare_report = pd.DataFrame({'Example':'Diff report placeholder'})
            st.dataframe(st.session_state.compare_report)

# -----------------------------
# Tab 5: Modeling
# -----------------------------
with tab5:
    st.header("Modeling & Prediction")
    target_column = st.text_input("Enter target column for prediction")
    if st.button("Train Model"):
        st.write("Modeling placeholder: train/test split, train model, show metrics")
        st.session_state.model_metrics = pd.DataFrame({'Metric':['Accuracy','RMSE'],'Value':[0.0,0.0]})
        st.dataframe(st.session_state.model_metrics)

# -----------------------------
# Tab 6: Explainability
# -----------------------------
with tab6:
    st.header("Explainability")
    if st.button("Show Explainability"):
        st.write("Explainability placeholder: show feature importance, SHAP values")
        st.session_state.shap_plots = "SHAP plots placeholder"
        st.write(st.session_state.shap_plots)

# -----------------------------
# Tab 7: Export
# -----------------------------
with tab7:
    st.header("Export Reports")
    export_options = st.multiselect(
        "Select items to export",
        options=['Cleaned Dataset A', 'Cleaned Dataset B', 'EDA Reports', 'Compare Report', 'Model Metrics', 'SHAP Plots']
    )
    file_name = st.text_input("Enter export file name", value="DataLens_Report.xlsx")

    if st.button("Export"):
        with io.BytesIO() as output:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                if 'Cleaned Dataset A' in export_options and st.session_state.cleaned_a is not None:
                    st.session_state.cleaned_a.to_excel(writer, sheet_name='Cleaned A', index=False)
                if 'Cleaned Dataset B' in export_options and st.session_state.cleaned_b is not None:
                    st.session_state.cleaned_b.to_excel(writer, sheet_name='Cleaned B', index=False)
                if 'Compare Report' in export_options and st.session_state.compare_report is not None:
                    st.session_state.compare_report.to_excel(writer, sheet_name='Compare', index=False)
                if 'Model Metrics' in export_options and st.session_state.model_metrics is not None:
                    st.session_state.model_metrics.to_excel(writer, sheet_name='Model Metrics', index=False)
                # Placeholder for EDA reports or SHAP plots: can be added here
            data = output.getvalue()
            st.download_button(
                label="Download Excel Report",
                data=data,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success(f"Exported {file_name} successfully!")
