# -----------------------------
# streamlit_app.py
# DataLens — Advanced Analytics Tool
# -----------------------------

import streamlit as st
import pandas as pd
import io
import plotly.express as px
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import shap

# -----------------------------
# Initialize session state
# -----------------------------
for key in [
    'dataset_a', 'dataset_b', 'cleaned_a', 'cleaned_b',
    'eda_report_a', 'eda_report_b', 'compare_report',
    'model_metrics', 'shap_plots', 'model', 'X_train', 'X_test', 'y_train', 'y_test'
]:
    st.session_state.setdefault(key, None)

# -----------------------------
# App Title
# -----------------------------
st.title("DataLens — Advanced Analytics Tool")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Upload", "Cleaning", "EDA", "Compare", "Modeling", "Explainability", "Export"
])
# -----------------------------
# Tab 1: Upload (Advanced Pro Edition)
# -----------------------------
with tab1:
    st.header("Upload & Inspect Datasets")

    # File uploaders
    dataset_a_file = st.file_uploader("Upload Dataset A", type=['csv', 'xlsx', 'parquet', 'json'])
    dataset_b_file = st.file_uploader("Upload Dataset B", type=['csv', 'xlsx', 'parquet', 'json'])

    # Display file details
    def show_file_details(file, label):
        if file:
            file_details = {
                "Filename": file.name,
                "Type": file.type or "N/A",
                "Size (KB)": round(file.size / 1024, 2)
            }
            st.markdown(f"**{label} Details:**")
            st.json(file_details)

    show_file_details(dataset_a_file, "Dataset A")
    show_file_details(dataset_b_file, "Dataset B")

    # Load datasets
    if st.button("Load Datasets", use_container_width=True):
        try:
            def load_file(file):
                if file.name.endswith('.csv'):
                    return pd.read_csv(file)
                elif file.name.endswith('.xlsx'):
                    return pd.read_excel(file)
                elif file.name.endswith('.parquet'):
                    return pd.read_parquet(file)
                elif file.name.endswith('.json'):
                    return pd.read_json(file)
                else:
                    st.warning(f"Unsupported file type: {file.name}")
                    return None

            if dataset_a_file:
                st.session_state.dataset_a = load_file(dataset_a_file)
            if dataset_b_file:
                st.session_state.dataset_b = load_file(dataset_b_file)

            st.success("✅ Datasets loaded successfully!")

        except Exception as e:
            st.error(f"Error loading datasets: {e}")

    # Utility function to preview datasets safely
    def preview_dataset(df, name):
        if df is not None:
            st.subheader(f"{name} Preview")

            # Smart sampling for large datasets
            max_rows = 10000
            if len(df) > max_rows:
                st.info(f"Dataset too large ({len(df)} rows). Showing first {max_rows} rows.")
                df = df.sample(max_rows, random_state=42)

            # Basic info
            st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.markdown(f"**Missing Values:** {df.isnull().sum().sum()}")
            st.dataframe(df.head())

            # Column selection
            with st.expander(f"🔍 Configure Columns for {name}"):
                columns = df.columns.tolist()
                st.session_state[f"target_{name.lower()}"] = st.selectbox(
                    "Select Target Column (if applicable)", [None] + columns, key=f"target_{name}"
                )
                st.session_state[f"id_{name.lower()}"] = st.selectbox(
                    "Select ID Column (optional)", [None] + columns, key=f"id_{name}"
                )

            # Quick data preview plots
            with st.expander(f"📊 Quick Visuals for {name}"):
                numeric_cols = df.select_dtypes(include='number').columns
                categorical_cols = df.select_dtypes(exclude='number').columns

                if len(numeric_cols) > 0:
                    selected_num = st.selectbox(f"Numeric Column for Histogram ({name})", numeric_cols)
                    st.plotly_chart(
                        px.histogram(df, x=selected_num, nbins=30, title=f"{name}: Distribution of {selected_num}"),
                        use_container_width=True
                    )

                if len(categorical_cols) > 0:
                    selected_cat = st.selectbox(f"Categorical Column for Bar Chart ({name})", categorical_cols)
                    st.plotly_chart(
                        px.bar(df[selected_cat].value_counts().reset_index(),
                               x='index', y=selected_cat,
                               title=f"{name}: Category Counts for {selected_cat}"),
                        use_container_width=True
                    )

    # Show previews if datasets are loaded
    preview_dataset(st.session_state.dataset_a, "Dataset A")
    preview_dataset(st.session_state.dataset_b, "Dataset B")

    # Management tools
    st.markdown("---")
    st.subheader("🧹 Dataset Management")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear Datasets"):
            for key in ["dataset_a", "dataset_b", "cleaned_a", "cleaned_b", "eda_report_a", "eda_report_b"]:
                st.session_state[key] = None
            st.warning("Datasets cleared from session.")

    with col2:
        if st.button("Reload Last Datasets"):
            if st.session_state.dataset_a is not None or st.session_state.dataset_b is not None:
                st.info("Datasets reloaded from session cache.")
            else:
                st.error("No datasets in cache to reload.")
# -----------------------------
# Tab 2: Cleaning
# -----------------------------
with tab2:
    st.header("Cleaning & Preparation")
    if st.button("Run Cleaning"):
        def clean_dataset(df):
            if df is None:
                return None
            df = df.copy()
            # Handle missing values
            for col in df.columns:
                if df[col].dtype == 'O':  # object columns
                    df[col].fillna('Unknown', inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
            # Convert dates
            if 'JoinDate' in df.columns:
                df['JoinDate'] = pd.to_datetime(df['JoinDate'], errors='coerce')
                df['JoinDate'].fillna(pd.Timestamp('2000-01-01'), inplace=True)
            # Remove duplicates
            df.drop_duplicates(inplace=True)
            return df

        st.session_state.cleaned_a = clean_dataset(st.session_state.dataset_a)
        st.session_state.cleaned_b = clean_dataset(st.session_state.dataset_b)

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
        st.write("EDA: Summary stats, histograms, and profiling")

        def display_eda(dataset, name, key_prefix):
            if dataset is not None:
                st.write(f"{name} Summary")
                st.dataframe(dataset.describe(include='all'))

                numeric_cols = dataset.select_dtypes(include='number').columns
                if len(numeric_cols) > 0:
                    for i, col in enumerate(numeric_cols[:5]):  # limit to 5 charts
                        fig = px.histogram(dataset, x=col, title=f"Histogram of {col}")
                        st.plotly_chart(fig, key=f"{key_prefix}_{i}")

                # Automated profiling
                profile = ProfileReport(dataset, title=f"{name} Profiling Report", explorative=True)
                st.session_state[f"eda_report_{key_prefix}"] = profile
                st.write(profile.to_html(), unsafe_allow_html=True)

        display_eda(st.session_state.cleaned_a, "Dataset A", "a")
        display_eda(st.session_state.cleaned_b, "Dataset B", "b")

# -----------------------------
# Tab 4: Compare & Contrast
# -----------------------------
with tab4:
    st.header("Compare & Contrast")
    compare_type = st.selectbox("Select Compare Type", [
        'Row presence check', 'Cell-by-cell comparison', 'Summary compare', 'Schema compare'
    ])

    if st.button("Run Compare"):
        if st.session_state.cleaned_a is None or st.session_state.cleaned_b is None:
            st.warning("Please run cleaning or upload both datasets first")
        else:
            if compare_type == 'Row presence check':
                ids_a = set(st.session_state.cleaned_a['EmployeeID']) if 'EmployeeID' in st.session_state.cleaned_a.columns else set()
                ids_b = set(st.session_state.cleaned_b['EmployeeID']) if 'EmployeeID' in st.session_state.cleaned_b.columns else set()
                only_in_a = ids_a - ids_b
                only_in_b = ids_b - ids_a
                report = pd.DataFrame({
                    'Only in A': list(only_in_a) + [None]*(max(len(only_in_a), len(only_in_b)) - len(only_in_a)),
                    'Only in B': list(only_in_b) + [None]*(max(len(only_in_a), len(only_in_b)) - len(only_in_b))
                })
                st.session_state.compare_report = report

            elif compare_type == 'Schema compare':
                cols_a = set(st.session_state.cleaned_a.columns)
                cols_b = set(st.session_state.cleaned_b.columns)
                cols_only_a = list(cols_a - cols_b)
                cols_only_b = list(cols_b - cols_a)
                max_len = max(len(cols_only_a), len(cols_only_b))
                cols_only_a += [None] * (max_len - len(cols_only_a))
                cols_only_b += [None] * (max_len - len(cols_only_b))
                report = pd.DataFrame({
                    'Columns only in A': cols_only_a,
                    'Columns only in B': cols_only_b
                })
                st.session_state.compare_report = report

            elif compare_type == 'Cell-by-cell comparison':
                common_cols = set(st.session_state.cleaned_a.columns).intersection(st.session_state.cleaned_b.columns)
                diffs = []
                for col in common_cols:
                    if pd.api.types.is_numeric_dtype(st.session_state.cleaned_a[col]):
                        diff_count = st.session_state.cleaned_a[col].mean() - st.session_state.cleaned_b[col].mean()
                        diffs.append({'Column': col, 'Mean Difference': diff_count})
                st.session_state.compare_report = pd.DataFrame(diffs)

            else:  # Summary compare
                summary_a = st.session_state.cleaned_a.describe()
                summary_b = st.session_state.cleaned_b.describe()
                st.session_state.compare_report = pd.concat([summary_a, summary_b], keys=['Dataset A', 'Dataset B'])

            # Display the compare report if it exists
            if st.session_state.compare_report is not None:
                st.dataframe(st.session_state.compare_report)

# -----------------------------
# Tab 5: Modeling
# -----------------------------
with tab5:
    st.header("Modeling & Prediction")
    target_column = st.text_input("Enter target column for prediction")
    if st.button("Train Model"):
        if target_column and target_column in st.session_state.cleaned_a.columns:
            df = st.session_state.cleaned_a.dropna()
            X = df.drop(columns=[target_column])
            y = df[target_column]
            X = pd.get_dummies(X)  # Encode categorical
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            st.session_state.model_metrics = pd.DataFrame({'Metric': ['Accuracy', 'RMSE'], 'Value': [acc, rmse]})
            st.session_state.model = model
            st.session_state.X_train, st.session_state.X_test = X_train, X_test
            st.session_state.y_train, st.session_state.y_test = y_train, y_test
            st.dataframe(st.session_state.model_metrics)
        else:
            st.warning("Please enter a valid target column.")

# -----------------------------
# Tab 6: Explainability
# -----------------------------
with tab6:
    st.header("Explainability")
    if st.button("Show Explainability"):
        if st.session_state.model is not None:
            explainer = shap.TreeExplainer(st.session_state.model)
            shap_values = explainer.shap_values(st.session_state.X_test)
            st.write("Feature Importance (SHAP):")
            shap.summary_plot(shap_values, st.session_state.X_test, plot_type="bar")
        else:
            st.warning("Train a model first.")

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
            data = output.getvalue()
            st.download_button(
                label="Download Excel Report",
                data=data,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success(f"Exported {file_name} successfully!")
