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
# Tab 1: Upload (Advanced Pro Edition - Fixed)
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

                # Numeric preview
                if len(numeric_cols) > 0:
                    selected_num = st.selectbox(f"Numeric Column for Histogram ({name})", numeric_cols)
                    st.plotly_chart(
                        px.histogram(df, x=selected_num, nbins=30, title=f"{name}: Distribution of {selected_num}"),
                        use_container_width=True
                    )

                # Categorical preview (fixed section)
                if len(categorical_cols) > 0:
                    selected_cat = st.selectbox(f"Categorical Column for Bar Chart ({name})", categorical_cols)
                    if selected_cat and df[selected_cat].notna().sum() > 0:
                        cat_counts = df[selected_cat].value_counts(dropna=False).reset_index()
                        cat_counts.columns = [selected_cat, "count"]
                        st.plotly_chart(
                            px.bar(
                                cat_counts,
                                x=selected_cat,
                                y="count",
                                title=f"{name}: Category Counts for {selected_cat}",
                            ),
                            use_container_width=True
                        )
                    else:
                        st.info(f"No valid categorical data to plot for {selected_cat}.")

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
# Tab 2: Cleaning & Preparation (Advanced Pro Edition)
# -----------------------------
with tab2:
    st.header("Cleaning & Preparation")

    st.markdown("""
    Configure how you'd like to clean your datasets.
    You can choose different strategies for missing values, dates, and outliers.
    """)

    # --- Cleaning options ---
    with st.expander("⚙️ Cleaning Options"):
        missing_strategy = st.selectbox(
            "Missing Value Strategy",
            ["Median (numeric) / 'Unknown' (categorical)", "Drop Rows with Missing Values", "Fill with Mean (numeric)"],
        )
        handle_outliers = st.checkbox("Clip Outliers to 1st–99th Percentile (numeric columns)", value=True)
        convert_dates = st.checkbox("Auto-convert Date Columns", value=True)
        drop_duplicates = st.checkbox("Remove Duplicates", value=True)

    if st.button("🧹 Run Cleaning", use_container_width=True):

        def clean_dataset(df, label):
            if df is None:
                st.warning(f"No data found for {label}. Please upload first.")
                return None

            df = df.copy()
            original_shape = df.shape
            original_missing = df.isnull().sum().sum()

            # --- Handle missing values ---
            for col in df.columns:
                if df[col].dtype == 'O':
                    if missing_strategy == "Drop Rows with Missing Values":
                        df = df[df[col].notna()]
                    else:
                        df[col].fillna('Unknown', inplace=True)
                else:
                    if missing_strategy == "Drop Rows with Missing Values":
                        df = df[df[col].notna()]
                    elif missing_strategy == "Fill with Mean (numeric)":
                        df[col].fillna(df[col].mean(), inplace=True)
                    else:
                        df[col].fillna(df[col].median(), inplace=True)

            # --- Auto-convert dates ---
            if convert_dates:
                date_like = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
                for c in date_like:
                    df[c] = pd.to_datetime(df[c], errors='coerce')

            # --- Outlier clipping ---
            if handle_outliers:
                num_cols = df.select_dtypes(include='number').columns
                for c in num_cols:
                    lower, upper = df[c].quantile(0.01), df[c].quantile(0.99)
                    df[c] = df[c].clip(lower, upper)

            # --- Remove duplicates ---
            if drop_duplicates:
                df.drop_duplicates(inplace=True)

            # --- Summary ---
            st.success(f"✅ Cleaned {label}")
            st.markdown(f"**Original shape:** {original_shape[0]} rows × {original_shape[1]} columns")
            st.markdown(f"**Cleaned shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.markdown(f"**Missing values fixed:** {original_missing - df.isnull().sum().sum()}")

            # --- Quick numeric preview ---
            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) > 0:
                with st.expander(f"📊 {label}: Quick Numeric Distribution"):
                    chosen_num = st.selectbox(f"{label} numeric column", numeric_cols, key=f"dist_{label}")
                    st.plotly_chart(px.histogram(df, x=chosen_num, nbins=30, title=f"{label}: {chosen_num} Distribution"))

            return df

        # Run cleaning for both datasets
        st.session_state.cleaned_a = clean_dataset(st.session_state.dataset_a, "Dataset A")
        st.session_state.cleaned_b = clean_dataset(st.session_state.dataset_b, "Dataset B")

        # --- Show previews ---
        st.markdown("### 🧾 Cleaned Dataset Previews")
        if st.session_state.cleaned_a is not None:
            st.dataframe(st.session_state.cleaned_a.head())
        if st.session_state.cleaned_b is not None:
            st.dataframe(st.session_state.cleaned_b.head())

# -----------------------------
# Tab 3: EDA Visualization (Dynamic)
# -----------------------------
with tab3:
    st.header("Exploratory Data Analysis (Visualization)")

    if st.session_state.cleaned_a is not None:
        df = st.session_state.cleaned_a

        chart_type = st.selectbox(
            "Select chart type",
            ["Histogram", "Boxplot", "Scatter Plot", "Correlation Heatmap", "Countplot (Categorical)"]
        )

        if chart_type == "Histogram":
            col = st.selectbox("Select column for Histogram", df.columns)
            bins = st.slider("Number of bins", 5, 100, 20)
            fig = px.histogram(df, x=col, nbins=bins, title=f"Histogram of {col}")
            st.plotly_chart(fig)

        elif chart_type == "Boxplot":
            col = st.selectbox("Select column for Boxplot", df.columns)
            fig = px.box(df, y=col, title=f"Boxplot of {col}")
            st.plotly_chart(fig)

        elif chart_type == "Scatter Plot":
            x_col = st.selectbox("X-axis", df.columns, key="scatter_x")
            y_col = st.selectbox("Y-axis", df.columns, key="scatter_y")
            color_col = st.selectbox("Color by (optional)", [None] + list(df.columns))
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
            st.plotly_chart(fig)

        elif chart_type == "Correlation Heatmap":
            numeric_df = df.select_dtypes(include='number')
            if numeric_df.shape[1] < 2:
                st.info("Need at least two numeric columns for correlation heatmap")
            else:
                corr = numeric_df.corr()
                fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Heatmap")
                st.plotly_chart(fig)

        elif chart_type == "Countplot (Categorical)":
            cat_cols = df.select_dtypes(include='object').columns
            if len(cat_cols) == 0:
                st.info("No categorical columns found")
            else:
                col = st.selectbox("Select categorical column", cat_cols)
                fig = px.histogram(df, x=col, title=f"Countplot of {col}")
                st.plotly_chart(fig)
                
# -----------------------------
# Tab 4: Compare & Contrast (Robust)
# -----------------------------
with tab4:
    st.header("Compare & Contrast")

    # Ensure datasets exist
    if st.session_state.cleaned_a is None or st.session_state.cleaned_b is None:
        st.warning("Please run cleaning or upload both datasets first")
    else:
        common_cols = list(set(st.session_state.cleaned_a.columns).intersection(
            st.session_state.cleaned_b.columns))

        compare_type = st.selectbox("Select Compare Type", [
            'Row presence check', 'Cell-by-cell comparison', 'Summary compare', 'Schema compare'
        ])

        key_col = None
        if compare_type == 'Row presence check' and common_cols:
            key_col = st.selectbox("Select key column for row matching (optional)", common_cols)

        if st.button("Run Compare"):
            report = None

            # -----------------------------
            # Row presence check
            # -----------------------------
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

            # -----------------------------
            # Schema compare
            # -----------------------------
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

            # -----------------------------
            # Cell-by-cell comparison
            # -----------------------------
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
                        diff_count = ((col_a != col_b) & (~col_a.isna()) & (~col_b.isna())).sum()
                        diffs.append({'Column': col, 'Mean Difference': mean_diff, 'Count Differences': diff_count})
                    else:
                        mismatches = ((col_a != col_b) & (~col_a.isna()) & (~col_b.isna())).sum()
                        diffs.append({'Column': col, 'Mismatched Count': mismatches})

                report = pd.DataFrame(diffs)

            # -----------------------------
            # Summary comparison
            # -----------------------------
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
# Tab 5: Modeling & Prediction (Advanced)
# -----------------------------
with tab5:
    st.header("Modeling & Prediction")

    if st.session_state.cleaned_a is not None:
        target_column = st.selectbox("Select target column for prediction", st.session_state.cleaned_a.columns)
        feature_columns = st.multiselect(
            "Select feature columns (default: all except target)",
            options=[c for c in st.session_state.cleaned_a.columns if c != target_column],
            default=[c for c in st.session_state.cleaned_a.columns if c != target_column]
        )

        model_choice = st.selectbox("Select model", ["Random Forest", "Logistic Regression"])
        test_size = st.slider("Test set size (%)", 10, 50, 20)
        n_estimators = st.slider("Random Forest n_estimators", 50, 500, 100) if model_choice == "Random Forest" else None

        if st.button("Train Model"):
            if target_column and feature_columns:
                df = st.session_state.cleaned_a.dropna(subset=[target_column]+feature_columns)
                X = df[feature_columns]
                y = df[target_column]

                # Encode categoricals
                X = pd.get_dummies(X)

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

                # Model selection
                if model_choice == "Random Forest":
                    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                else:
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(max_iter=500)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                st.session_state.model_metrics = pd.DataFrame({'Metric': ['Accuracy', 'RMSE'], 'Value': [acc, rmse]})
                st.session_state.model = model
                st.session_state.X_train, st.session_state.X_test = X_train, X_test
                st.session_state.y_train, st.session_state.y_test = y_train, y_test

                # Display metrics
                st.subheader("Model Metrics")
                st.dataframe(st.session_state.model_metrics)

                # Confusion matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), title="Confusion Matrix")
                st.plotly_chart(fig)

                # Feature importance (Random Forest only)
                if model_choice == "Random Forest":
                    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
                    st.subheader("Feature Importances")
                    fig2 = px.bar(importances.head(10), x=importances.head(10).index, y=importances.head(10).values, title="Top 10 Features")
                    st.plotly_chart(fig2)

            else:
                st.warning("Please select a valid target and features.")

# -----------------------------
# Tab 6: Explainability
# -----------------------------
with tab6:
    st.header("Explainability")
    if st.button("Show Explainability"):
        if st.session_state.model is not None:
            import shap
            explainer = shap.TreeExplainer(st.session_state.model)
            shap_values = explainer.shap_values(st.session_state.X_test)

            st.write("Feature Importance (SHAP):")
            shap.summary_plot(shap_values, st.session_state.X_test, plot_type="bar", show=False)
            st.pyplot(bbox_inches='tight')
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
                    st.session_state.compare_report.to_excel(writer, sheet_name='Compare', index=True)
                if 'Model Metrics' in export_options and st.session_state.model_metrics is not None:
                    st.session_state.model_metrics.to_excel(writer, sheet_name='Model Metrics', index=False)
                # SHAP plots could be saved as images and added to Excel if needed
            data = output.getvalue()
            st.download_button(
                label="Download Excel Report",
                data=data,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success(f"Exported {file_name} successfully!")



