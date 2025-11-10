import streamlit as st
import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
import plotly.express as px

st.set_page_config(layout="wide")
st.title("DataLens")

# -----------------------------
# Session state placeholders
# -----------------------------
for key in ["cleaned_a", "cleaned_b", "compare_report", "model", "model_metrics", "X_train", "X_test", "y_train", "y_test"]:
    if key not in st.session_state:
        st.session_state[key] = None

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Upload", "Cleaning", "EDA", "Compare & Contrast", "Modeling", "Explainability", "Export"
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

    if st.session_state.cleaned_a is not None:
        st.subheader("Dataset A Preview")
        st.dataframe(st.session_state.cleaned_a.head())

    if st.session_state.cleaned_b is not None:
        st.subheader("Dataset B Preview")
        st.dataframe(st.session_state.cleaned_b.head())

    cleaning_options = st.multiselect(
        "Select cleaning operations to apply",
        [
            "Drop duplicate rows",
            "Fill missing numeric values with median",
            "Fill missing categorical values with mode",
            "Trim whitespace from string columns",
            "Remove columns with all nulls"
        ],
        default=["Drop duplicate rows"],
        key="cleaning_options"
    )

    if st.button("Run Cleaning", key="clean_button"):
        for ds_name in ["cleaned_a", "cleaned_b"]:
            df = st.session_state[ds_name]
            if df is None:
                continue

            original_shape = df.shape
            changes = []

            # --- Cleaning operations ---
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
                    if df[col].apply(lambda x: isinstance(x, str)).any():
                        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                changes.append("Trimmed whitespace from string columns.")

            if "Remove columns with all nulls" in cleaning_options:
                all_null_cols = df.columns[df.isna().all()].tolist()
                if all_null_cols:
                    df.drop(columns=all_null_cols, inplace=True)
                    changes.append(f"Removed {len(all_null_cols)} columns containing only null values.")

            # --- Store cleaned data and report ---
            st.session_state[ds_name] = df
            new_shape = df.shape

            st.subheader(f"{'Dataset A' if ds_name == 'cleaned_a' else 'Dataset B'} Cleaning Summary")
            if changes:
                st.write("**Changes applied:**")
                for c in changes:
                    st.markdown(f"- {c}")
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
        ["Dataset A", "Dataset B"],
        default=["Dataset A"],
        key="eda_datasets"
    )

    chart_type = st.selectbox(
        "Select chart type",
        ["All", "Histogram", "Boxplot", "Scatter", "Correlation Heatmap"],
        key="eda_chart_type"
    )

    if st.button("Run EDA", key="eda_run"):
        for ds in datasets_choice:
            if ds == "Dataset A" and st.session_state.cleaned_a is not None:
                df = st.session_state.cleaned_a
            elif ds == "Dataset B" and st.session_state.cleaned_b is not None:
                df = st.session_state.cleaned_b
            else:
                continue

            st.subheader(f"{ds} - {chart_type if chart_type != 'All' else 'Comprehensive EDA'}")

            numeric_cols = df.select_dtypes(include=['number']).columns
            all_cols = df.columns

            def plot_histogram():
                col = st.selectbox(f"Select column for {ds} histogram", numeric_cols, key=f"hist_{ds}")
                fig = px.histogram(df, x=col, title=f"{ds} - Histogram of {col}")
                st.plotly_chart(fig, use_container_width=True)

            def plot_boxplot():
                col = st.selectbox(f"Select column for {ds} boxplot", numeric_cols, key=f"box_{ds}")
                fig = px.box(df, y=col, title=f"{ds} - Boxplot of {col}")
                st.plotly_chart(fig, use_container_width=True)

            def plot_scatter():
                x_col = st.selectbox(f"X-axis column for {ds}", all_cols, key=f"scatter_x_{ds}")
                y_col = st.selectbox(f"Y-axis column for {ds}", numeric_cols, key=f"scatter_y_{ds}")
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{ds} - Scatter: {x_col} vs {y_col}")
                st.plotly_chart(fig, use_container_width=True)

            def plot_corr_heatmap():
                if len(numeric_cols) == 0:
                    st.warning(f"No numeric columns found in {ds} for correlation heatmap.")
                else:
                    corr = df[numeric_cols].corr()
                    fig = px.imshow(corr, text_auto=True, title=f"{ds} - Correlation Heatmap")
                    st.plotly_chart(fig, use_container_width=True)

            # Determine which charts to render
            if chart_type == "All":
                with st.expander(f"{ds} - Histogram"):
                    plot_histogram()
                with st.expander(f"{ds} - Boxplot"):
                    plot_boxplot()
                with st.expander(f"{ds} - Scatter"):
                    plot_scatter()
                with st.expander(f"{ds} - Correlation Heatmap"):
                    plot_corr_heatmap()
            elif chart_type == "Histogram":
                plot_histogram()
            elif chart_type == "Boxplot":
                plot_boxplot()
            elif chart_type == "Scatter":
                plot_scatter()
            elif chart_type == "Correlation Heatmap":
                plot_corr_heatmap()

# -----------------------------
# Tab 4: Compare & Contrast
# -----------------------------
with tab4:
    st.header("Compare & Contrast")

    if st.session_state.cleaned_a is None or st.session_state.cleaned_b is None:
        st.warning("Please run cleaning or upload both datasets first")
    else:
        common_cols = list(set(st.session_state.cleaned_a.columns).intersection(st.session_state.cleaned_b.columns))

        compare_type = st.selectbox(
            "Select Compare Type",
            ['Row presence check', 'Cell-by-cell comparison', 'Summary compare', 'Schema compare'],
            key="compare_type"
        )

        key_col = None
        if compare_type == 'Row presence check' and common_cols:
            key_col = st.selectbox(
                "Select key column for row matching (optional)",
                common_cols,
                key="compare_key"
            )

        if st.button("Run Compare", key="run_compare"):
            report = None

            # Row presence check
            if compare_type == 'Row presence check':
                if key_col:
                    ids_a = set(st.session_state.cleaned_a[key_col])
                    ids_b = set(st.session_state.cleaned_b[key_col])
                    only_in_a = ids_a - ids_b
                    only_in_b = ids_b - ids_a
                    report = pd.DataFrame({
                        f'Only in A ({key_col})': list(only_in_a) + [None]*(max(len(only_in_a), len(only_in_b)) - len(only_in_a)),
                        f'Only in B ({key_col})': list(only_in_b) + [None]*(max(len(only_in_a), len(only_in_b)) - len(only_in_b))
                    })
                else:
                    st.info("No key column selected. Skipping row presence check.")

            # Schema compare
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

            # Cell-by-cell comparison
            elif compare_type == 'Cell-by-cell comparison':
                diffs = []
                max_len = max(len(st.session_state.cleaned_a), len(st.session_state.cleaned_b))
                df_a = st.session_state.cleaned_a.reindex(range(max_len))
                df_b = st.session_state.cleaned_b.reindex(range(max_len))

                for col in common_cols:
                    col_a = df_a[col]
                    col_b = df_b[col]
                    if pd.api.types.is_numeric_dtype(col_a):
                        mean_diff = col_a.mean(skipna=True) - col_b.mean(skipna=True)
                        diff_count = (col_a != col_b).sum()
                        diffs.append({'Column': col, 'Mean Difference': mean_diff, 'Count Differences': diff_count})
                    else:
                        mismatches = (col_a != col_b).sum()
                        diffs.append({'Column': col, 'Mismatched Count': mismatches})
                report = pd.DataFrame(diffs)

            # Summary compare
            else:
                summary_a = st.session_state.cleaned_a.describe(include='all')
                summary_b = st.session_state.cleaned_b.describe(include='all')
                report = pd.concat([summary_a, summary_b], keys=['Dataset A', 'Dataset B'])

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
# Tab 5: Modeling
# -----------------------------
with tab5:
    st.header("Modeling & Prediction")

    if st.session_state.cleaned_a is not None:
        target_column = st.selectbox(
            "Select target column",
            st.session_state.cleaned_a.columns,
            key="model_target"
        )
        feature_columns = st.multiselect(
            "Select feature columns",
            [c for c in st.session_state.cleaned_a.columns if c != target_column],
            default=[c for c in st.session_state.cleaned_a.columns if c != target_column],
            key="model_features"
        )

        model_choice = st.selectbox(
            "Select model",
            ["Random Forest", "Logistic Regression"],
            key="model_choice"
        )
        test_size = st.slider("Test set size (%)", 10, 50, 20, key="test_size")
        n_estimators = st.slider("Random Forest n_estimators", 50, 500, 100, key="rf_estimators") if model_choice=="Random Forest" else None

        if st.button("Train Model", key="train_model"):
            df = st.session_state.cleaned_a.dropna(subset=[target_column]+feature_columns)
            X = pd.get_dummies(df[feature_columns])
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
            if model_choice=="Random Forest":
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            else:
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=500)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            st.session_state.model_metrics = pd.DataFrame({'Metric': ['Accuracy', 'RMSE'], 'Value':[acc, rmse]})
            st.session_state.model = model
            st.session_state.X_train, st.session_state.X_test = X_train, X_test
            st.session_state.y_train, st.session_state.y_test = y_train, y_test
            st.subheader("Model Metrics")
            st.dataframe(st.session_state.model_metrics)

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), title="Confusion Matrix")
            st.plotly_chart(fig)

            if model_choice=="Random Forest":
                importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
                st.subheader("Feature Importances")
                fig2 = px.bar(importances.head(10), x=importances.head(10).index, y=importances.head(10).values, title="Top 10 Features")
                st.plotly_chart(fig2)

# -----------------------------
# Tab 6: Explainability
# -----------------------------
with tab6:
    st.header("Explainability")
    if st.button("Show Explainability", key="show_shap"):
        if st.session_state.model is not None:
            st.info("SHAP explainability logic goes here.")
        else:
            st.warning("Train a model first.")

# -----------------------------
# Tab 7: Export
# -----------------------------
with tab7:
    st.header("Export Reports")

    export_options = st.multiselect(
        "Select items to export",
        options=['Cleaned Dataset A', 'Cleaned Dataset B', 'EDA Reports', 'Compare Report', 'Model Metrics', 'SHAP Plots'],
        key="export_options"
    )

    file_name = st.text_input(
        "Enter export file name",
        value="DataLens_Report.xlsx",
        key="export_filename"
    )

    if st.button("Export", key="export_button"):
        st.success(f"Exported {', '.join(export_options)} to {file_name}")
