import streamlit as st
import pandas as pd
import io
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import shap

# Initialize session state
for key in ['raw_a','raw_b','cleaned_a','cleaned_b','compare_report','model','X_train','X_test','y_train','y_test','model_metrics']:
    if key not in st.session_state:
        st.session_state[key] = None

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Upload / Clean", "EDA", "Visualization", "Compare & Contrast",
    "Modeling", "Explainability", "Export"
])

# -----------------------------
# Tab 1: Upload & Clean
# -----------------------------
with tab1:
    st.header("Upload and Clean Datasets")
    uploaded_a = st.file_uploader("Upload Dataset A", type=['csv','xlsx'], key="upload_a")
    uploaded_b = st.file_uploader("Upload Dataset B (optional)", type=['csv','xlsx'], key="upload_b")

    def load_file(file):
        if file is None:
            return None
        try:
            if str(file).endswith('.csv'):
                return pd.read_csv(file)
            else:
                return pd.read_excel(file)
        except:
            st.error("Error reading file")
            return None

    st.session_state.raw_a = load_file(uploaded_a)
    st.session_state.raw_b = load_file(uploaded_b)

    if st.session_state.raw_a is not None:
        st.session_state.cleaned_a = st.session_state.raw_a.copy()
        st.write("Dataset A preview:")
        st.dataframe(st.session_state.cleaned_a.head())
    if st.session_state.raw_b is not None:
        st.session_state.cleaned_b = st.session_state.raw_b.copy()
        st.write("Dataset B preview:")
        st.dataframe(st.session_state.cleaned_b.head())

# -----------------------------
# Tab 2: EDA (Flexible & Multi-Dataset)
# -----------------------------
with tab2:
    st.header("Exploratory Data Analysis (EDA)")

    # -----------------------------
    # Dataset selection
    # -----------------------------
    datasets_choice = st.multiselect(
        "Select dataset(s) for EDA",
        ["Dataset A", "Dataset B"],
        default=["Dataset A"]
    )

    # Ensure at least one dataset is selected
    if not datasets_choice:
        st.warning("Please select at least one dataset to analyze.")
    else:
        # -----------------------------
        # Loop through selected datasets
        # -----------------------------
        for ds in datasets_choice:
            df = st.session_state.cleaned_a if ds == "Dataset A" else st.session_state.cleaned_b
            if df is None:
                st.info(f"{ds} is not available. Please upload or clean the dataset first.")
                continue

            st.subheader(f"EDA for {ds}")

            # -----------------------------
            # Summary statistics
            # -----------------------------
            st.write("**Summary Statistics:**")
            st.dataframe(df.describe(include='all'))

            # -----------------------------
            # Column type counts
            # -----------------------------
            st.write("**Column Types:**")
            col_types = pd.DataFrame({
                "Column": df.columns,
                "Type": [df[col].dtype for col in df.columns],
                "Missing Values": [df[col].isna().sum() for col in df.columns]
            })
            st.dataframe(col_types)

            # -----------------------------
            # Pivot table option
            # -----------------------------
            st.write("**Pivot Table:**")
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            pivot_index = st.selectbox(f"{ds} - Select pivot index", options=[None]+categorical_cols, key=f"{ds}_pivot_index")
            pivot_values = st.multiselect(f"{ds} - Select values", options=numeric_cols, key=f"{ds}_pivot_values")

            if pivot_index and pivot_values:
                pivot_table = pd.pivot_table(df, index=pivot_index, values=pivot_values, aggfunc='mean')
                st.dataframe(pivot_table)

            # -----------------------------
            # Optionally, chart selection
            # -----------------------------
            st.write("**Charts:**")
            chart_type = st.selectbox(f"{ds} - Select chart type", ["Bar", "Line", "Histogram"], key=f"{ds}_chart_type")
            chart_col = st.selectbox(f"{ds} - Select column for chart", df.columns, key=f"{ds}_chart_col")

            if chart_col:
                if chart_type == "Histogram":
                    st.bar_chart(df[chart_col])
                else:
                    st.line_chart(df[chart_col]) if chart_type == "Line" else st.bar_chart(df[chart_col])


# -----------------------------
# Tab 3: EDA (Flexible & Multi-Dataset)
# -----------------------------
with tab3:
    st.header("Exploratory Data Analysis (EDA)")

    # -----------------------------
    # Dataset selection
    # -----------------------------
    datasets_choice = st.multiselect(
        "Select dataset(s) for EDA",
        ["Dataset A", "Dataset B"],
        default=["Dataset A"]
    )

    if not datasets_choice:
        st.warning("Please select at least one dataset to analyze.")
    else:
        # -----------------------------
        # Loop through selected datasets
        # -----------------------------
        for ds in datasets_choice:
            df = st.session_state.cleaned_a if ds == "Dataset A" else st.session_state.cleaned_b
            if df is None:
                st.info(f"{ds} is not available. Please upload or clean the dataset first.")
                continue

            st.subheader(f"EDA for {ds}")

            # -----------------------------
            # Summary statistics
            # -----------------------------
            st.write("**Summary Statistics:**")
            st.dataframe(df.describe(include='all'))

            # -----------------------------
            # Column type counts
            # -----------------------------
            st.write("**Column Types & Missing Values:**")
            col_types = pd.DataFrame({
                "Column": df.columns,
                "Type": [df[col].dtype for col in df.columns],
                "Missing Values": [df[col].isna().sum() for col in df.columns]
            })
            st.dataframe(col_types)

            # -----------------------------
            # Pivot table
            # -----------------------------
            st.write("**Pivot Table:**")
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            pivot_index = st.selectbox(f"{ds} - Select pivot index", options=[None]+categorical_cols, key=f"{ds}_pivot_index")
            pivot_values = st.multiselect(f"{ds} - Select values", options=numeric_cols, key=f"{ds}_pivot_values")

            if pivot_index and pivot_values:
                pivot_table = pd.pivot_table(df, index=pivot_index, values=pivot_values, aggfunc='mean')
                st.dataframe(pivot_table)

            # -----------------------------
            # Chart visualization
            # -----------------------------
            st.write("**Charts:**")
            chart_type = st.selectbox(f"{ds} - Select chart type", ["Bar", "Line", "Histogram"], key=f"{ds}_chart_type")
            chart_col = st.selectbox(f"{ds} - Select column for chart", df.columns, key=f"{ds}_chart_col")

            if chart_col:
                if chart_type == "Histogram":
                    st.bar_chart(df[chart_col])
                else:
                    st.line_chart(df[chart_col]) if chart_type == "Line" else st.bar_chart(df[chart_col])
# -----------------------------
# Tab 4: Compare & Contrast
# -----------------------------
with tab4:
    st.header("Compare & Contrast")

    # Ensure we have datasets
    if st.session_state.cleaned_a is None and st.session_state.cleaned_b is None:
        st.warning("Please run cleaning or upload at least one dataset first")
    else:
        datasets_available = []
        if st.session_state.cleaned_a is not None:
            datasets_available.append("Dataset A")
        if st.session_state.cleaned_b is not None:
            datasets_available.append("Dataset B")

        # Dataset selection for comparison
        compare_datasets = st.multiselect(
            "Select datasets to compare",
            options=datasets_available,
            default=datasets_available
        )

        if not compare_datasets:
            st.warning("Select at least one dataset to compare.")
        else:
            # Dynamic common columns
            if len(compare_datasets) == 2:
                df1 = st.session_state.cleaned_a if compare_datasets[0] == "Dataset A" else st.session_state.cleaned_b
                df2 = st.session_state.cleaned_a if compare_datasets[1] == "Dataset A" else st.session_state.cleaned_b
                common_cols = list(set(df1.columns).intersection(df2.columns))
            else:
                df1 = st.session_state.cleaned_a if compare_datasets[0] == "Dataset A" else st.session_state.cleaned_b
                df2 = None
                common_cols = df1.columns.tolist()

            # Compare type
            compare_type = st.selectbox("Select Compare Type", [
                'Row presence check', 'Cell-by-cell comparison', 'Summary compare', 'Schema compare'
            ])

            # Key column for row presence
            key_col = None
            if compare_type == 'Row presence check' and common_cols:
                key_col = st.selectbox("Select key column for row matching (optional)", [None]+common_cols)

            if st.button("Run Compare"):
                report = None

                # -----------------------------
                # Row presence check
                # -----------------------------
                if compare_type == 'Row presence check' and df2 is not None:
                    if key_col:
                        ids_a = set(df1[key_col])
                        ids_b = set(df2[key_col])
                        only_in_a = ids_a - ids_b
                        only_in_b = ids_b - ids_a
                        report = pd.DataFrame({
                            f'Only in {compare_datasets[0]} ({key_col})': list(only_in_a) + [None]*(max(len(only_in_a), len(only_in_b)) - len(only_in_a)),
                            f'Only in {compare_datasets[1]} ({key_col})': list(only_in_b) + [None]*(max(len(only_in_a), len(only_in_b)) - len(only_in_b))
                        })
                    else:
                        st.info("No key column selected. Skipping row presence check.")

                # -----------------------------
                # Schema compare
                # -----------------------------
                elif compare_type == 'Schema compare' and df2 is not None:
                    cols_a = set(df1.columns)
                    cols_b = set(df2.columns)
                    cols_only_a = list(cols_a - cols_b)
                    cols_only_b = list(cols_b - cols_a)
                    max_len = max(len(cols_only_a), len(cols_only_b))
                    cols_only_a += [None] * (max_len - len(cols_only_a))
                    cols_only_b += [None] * (max_len - len(cols_only_b))
                    report = pd.DataFrame({
                        f'Columns only in {compare_datasets[0]}': cols_only_a,
                        f'Columns only in {compare_datasets[1]}': cols_only_b
                    })

                # -----------------------------
                # Cell-by-cell comparison
                # -----------------------------
                elif compare_type == 'Cell-by-cell comparison' and df2 is not None:
                    diffs = []
                    max_len = max(len(df1), len(df2))
                    df1_aligned = df1.reindex(range(max_len))
                    df2_aligned = df2.reindex(range(max_len))

                    for col in common_cols:
                        col1 = df1_aligned[col]
                        col2 = df2_aligned[col]

                        if pd.api.types.is_numeric_dtype(col1):
                            mean_diff = col1.mean(skipna=True) - col2.mean(skipna=True)
                            diff_count = (col1 != col2).sum()
                            diffs.append({'Column': col, 'Mean Difference': mean_diff, 'Count Differences': diff_count})
                        else:
                            mismatches = (col1 != col2).sum()
                            diffs.append({'Column': col, 'Mismatched Count': mismatches})
                    report = pd.DataFrame(diffs)

                # -----------------------------
                # Summary compare
                # -----------------------------
                elif compare_type == 'Summary compare':
                    summary1 = df1.describe(include='all')
                    if df2 is not None:
                        summary2 = df2.describe(include='all')
                        report = pd.concat([summary1, summary2], keys=compare_datasets)
                    else:
                        report = summary1

                # -----------------------------
                # Display & export
                # -----------------------------
                if report is not None:
                    st.session_state.compare_report = report
                    st.dataframe(report)

                    buffer = io.BytesIO()
                    report.to_excel(buffer, index=True)
                    st.download_button(
                        "Export Comparison Report",
                        data=buffer,
                        file_name="comparison_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
# -----------------------------
# Tab 5: Modeling & Prediction
# -----------------------------
with tab5:
    st.header("Modeling & Prediction")

    # -----------------------------
    # Dataset selection
    # -----------------------------
    datasets_available = []
    if st.session_state.cleaned_a is not None:
        datasets_available.append("Dataset A")
    if st.session_state.cleaned_b is not None:
        datasets_available.append("Dataset B")

    selected_dataset = st.selectbox(
        "Select dataset for modeling",
        options=datasets_available
    )

    if selected_dataset:
        df = st.session_state.cleaned_a if selected_dataset == "Dataset A" else st.session_state.cleaned_b

        if df.empty:
            st.warning(f"{selected_dataset} is empty. Please upload or clean the dataset first.")
        else:
            # -----------------------------
            # Target & feature selection
            # -----------------------------
            target_column = st.selectbox("Select target column", df.columns)
            feature_columns = st.multiselect(
                "Select feature columns (default: all except target)",
                options=[c for c in df.columns if c != target_column],
                default=[c for c in df.columns if c != target_column]
            )

            # -----------------------------
            # Model & hyperparameters
            # -----------------------------
            model_choice = st.selectbox("Select model", ["Random Forest", "Logistic Regression"])
            test_size = st.slider("Test set size (%)", 10, 50, 20)
            n_estimators = st.slider(
                "Random Forest n_estimators",
                50, 500, 100
            ) if model_choice == "Random Forest" else None

            if st.button("Train Model"):
                if target_column and feature_columns:
                    # Drop rows with missing target or features
                    df_model = df.dropna(subset=[target_column] + feature_columns)
                    X = df_model[feature_columns]
                    y = df_model[target_column]

                    # Encode categoricals
                    X = pd.get_dummies(X)

                    # Train/test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=42
                    )

                    # Model selection
                    if model_choice == "Random Forest":
                        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                    else:
                        from sklearn.linear_model import LogisticRegression
                        model = LogisticRegression(max_iter=500)

                    # Fit model
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # -----------------------------
                    # Metrics
                    # -----------------------------
                    from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
                    acc = accuracy_score(y_test, y_pred)
                    rmse = mean_squared_error(y_test, y_pred, squared=False)

                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'RMSE'],
                        'Value': [acc, rmse]
                    })
                    st.subheader("Model Metrics")
                    st.dataframe(metrics_df)

                    # -----------------------------
                    # Confusion matrix
                    # -----------------------------
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig = px.imshow(
                        cm,
                        text_auto=True,
                        labels=dict(x="Predicted", y="Actual"),
                        title="Confusion Matrix"
                    )
                    st.plotly_chart(fig)

                    # -----------------------------
                    # Feature importance (RF only)
                    # -----------------------------
                    if model_choice == "Random Forest":
                        importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
                        st.subheader("Feature Importances")
                        fig2 = px.bar(
                            importances.head(10),
                            x=importances.head(10).index,
                            y=importances.head(10).values,
                            title="Top 10 Features"
                        )
                        st.plotly_chart(fig2)

                    # Store in session state
                    st.session_state.model = model
                    st.session_state.X_train, st.session_state.X_test = X_train, X_test
                    st.session_state.y_train, st.session_state.y_test = y_train, y_test
                    st.session_state.model_metrics = metrics_df

                else:
                    st.warning("Please select a target and at least one feature column.")
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
            shap.summary_plot(shap_values, st.session_state.X_test, plot_type="bar", show=False)
            st.pyplot(bbox_inches='tight')
        else:
            st.warning("Train a model first")

# -----------------------------
# Tab 7: Export
# -----------------------------
with tab7:
    st.header("Export Reports")
    export_options = st.multiselect("Select items to export", options=['Cleaned Dataset A','Cleaned Dataset B','Compare Report','Model Metrics'])
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
            st.download_button("Download Excel Report", data=data, file_name=file_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.success(f"Exported {file_name} successfully!")
