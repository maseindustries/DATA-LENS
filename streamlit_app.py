import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
from fpdf import FPDF
from datetime import datetime
import os
from io import BytesIO
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import ttest_ind, chi2_contingency
import pickle # New import for session state saving/loading

st.set_page_config(layout="wide")
st.title("DataLens")

# -----------------------------
# Session state placeholders
# -----------------------------
for key in [
    "cleaned_a", "cleaned_b", "cleaned_a_saved", "cleaned_b_saved",
    "cleaned_a_name", "cleaned_b_name", "cleaned_a_operations", "cleaned_b_operations",
    "compare_report", "saved_charts",
    "original_a", "original_b", "original_a_name", "original_b_name"
]:
    if key not in st.session_state:
        st.session_state[key] = None

for key in ["is_cleaned_a", "is_cleaned_b"]:
    if key not in st.session_state:
        st.session_state[key] = False
# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Upload", "Cleaning", "EDA", "Compare & Contrast", "PDF Summary"
])
# -----------------------------
# Tab 1: Upload
# -----------------------------
with tab1:
    st.header("Upload Datasets")

    uploaded_file_a = st.file_uploader("Upload Dataset A", type=["csv", "xlsx"], key="upload_a")
    uploaded_file_b = st.file_uploader("Upload Dataset B", type=["csv", "xlsx"], key="upload_b")

    if uploaded_file_a or uploaded_file_b:
        if st.button("Confirm Upload"):
            if uploaded_file_a is not None:
                if uploaded_file_a.name.endswith(".csv"):
                    st.session_state.cleaned_a = pd.read_csv(uploaded_file_a)
                else:
                    st.session_state.cleaned_a = pd.read_excel(uploaded_file_a)
                st.session_state.cleaned_a_name = uploaded_file_a.name
                st.session_state.cleaned_a_saved = None
                st.session_state.cleaned_a_operations = []

            if uploaded_file_b is not None:
                if uploaded_file_b.name.endswith(".csv"):
                    st.session_state.cleaned_b = pd.read_csv(uploaded_file_b)
                else:
                    st.session_state.cleaned_b = pd.read_excel(uploaded_file_b)
                st.session_state.cleaned_b_name = uploaded_file_b.name
                st.session_state.cleaned_b_saved = None
                st.session_state.cleaned_b_operations = []

            st.success("Files uploaded successfully!")

    # Optional: Show a preview of uploaded files
    if st.session_state.cleaned_a is not None:
        st.write(f"Preview of {st.session_state.cleaned_a_name}")
        st.dataframe(st.session_state.cleaned_a.head())

    if st.session_state.cleaned_b is not None:
        st.write(f"Preview of {st.session_state.cleaned_b_name}")
        st.dataframe(st.session_state.cleaned_b.head())

    # -----------------------------
    # Save/Load Session State
    # -----------------------------
    st.subheader("Save/Load Session State")
    session_filename = st.text_input("Enter filename for session state (e.g., my_session.pkl)", "my_session.pkl")

    def save_session_state():
        state_to_save = {
            "cleaned_a": st.session_state.get("cleaned_a"),
            "cleaned_b": st.session_state.get("cleaned_b"),
            "cleaned_a_name": st.session_state.get("cleaned_a_name"),
            "cleaned_b_name": st.session_state.get("cleaned_b_name"),
            "cleaned_a_operations": st.session_state.get("cleaned_a_operations"),
            "cleaned_b_operations": st.session_state.get("cleaned_b_operations"),
            "compare_report": st.session_state.get("compare_report"),
            "saved_charts": st.session_state.get("saved_charts")
        }
        try:
            with open(session_filename, "wb") as f:
                pickle.dump(state_to_save, f)
            st.success(f"Session state saved to {session_filename}")
        except Exception as e:
            st.error(f"Error saving session state: {e}")

    def load_session_state():
        try:
            if os.path.exists(session_filename):
                with open(session_filename, "rb") as f:
                    loaded_state = pickle.load(f)
                for key, value in loaded_state.items():
                    st.session_state[key] = value
                st.success(f"Session state loaded from {session_filename}")
                st.rerun()
            else:
                st.error(f"File {session_filename} not found.")
        except Exception as e:
            st.error(f"Error loading session state: {e}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Session"):
            save_session_state()
    with col2:
        if st.button("Load Session"):
            load_session_state()


# -----------------------------
# Tab 2: Cleaning
# -----------------------------
with tab2:
    st.header("Data Cleaning")

    # -----------------------------
    # Safe cleaning function
    # -----------------------------
    def clean_dataset(df, label, cleaning_options, outlier_handling_method=None):
        # HARD safety check
        if df is None or not isinstance(df, pd.DataFrame):
            st.info(f"{label} is missing or invalid. Skipping cleaning.")
            return None, []

        # Debug
        st.write(f"DEBUG: Cleaning {label}, df type = {type(df)}")

        # Safe column detection
        numeric_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(include=['object']).columns

        applied_ops = []
        original_shape = df.shape

        # -----------------------------
        # Cleaning operations
        # -----------------------------
        if "Drop duplicate rows" in cleaning_options:
            before = len(df)
            df = df.drop_duplicates()
            if len(df) < before:
                applied_ops.append("Duplicate rows removed")

        if "Fill missing numeric values with median" in cleaning_options:
            for col in numeric_cols:
                na_count = df[col].isna().sum()
                if na_count > 0:
                    df[col].fillna(df[col].median(), inplace=True)
                    applied_ops.append(f"Filled {na_count} missing numeric values in {col}")

        if "Fill missing categorical values with mode" in cleaning_options:
            for col in cat_cols:
                na_count = df[col].isna().sum()
                if na_count > 0 and not df[col].mode().empty:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    applied_ops.append(f"Filled {na_count} missing categorical values in {col}")

        if "Trim whitespace from string columns" in cleaning_options:
            for col in cat_cols:
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            applied_ops.append("Trimmed whitespace from string columns")

        if "Remove columns with all nulls" in cleaning_options:
            all_null_cols = df.columns[df.isna().all()].tolist()
            if all_null_cols:
                df.drop(columns=all_null_cols, inplace=True)
                applied_ops.append(f"Removed {len(all_null_cols)} columns with all nulls")

        # -----------------------------
        # Outlier handling
        # -----------------------------
        if outlier_handling_method and len(numeric_cols) > 0:
            initial_rows = len(df)
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                if outlier_handling_method == "Remove outliers (IQR)":
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    if len(df) < initial_rows:
                        applied_ops.append(f"Removed outliers from {col} (IQR method)")
                elif outlier_handling_method == "Cap outliers (IQR)":
                    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
                    applied_ops.append(f"Capped outliers in {col} (IQR method)")

        return df, applied_ops

    # -----------------------------
    # Datasets
    # -----------------------------
    datasets = [
        ("cleaned_a", "Dataset A", st.session_state.get('cleaned_a_name', 'Dataset A')),
        ("cleaned_b", "Dataset B", st.session_state.get('cleaned_b_name', 'Dataset B'))
    ]

    # -----------------------------
    # Check at least one dataset exists
    # -----------------------------
    if all(st.session_state.get(ds_name) is None for ds_name, _, _ in datasets):
        st.warning("Please upload at least one dataset in Tab 1 before cleaning.")
    else:
        # Preview datasets
        for ds_name, label, _ in datasets:
            df = st.session_state.get(ds_name)
            if isinstance(df, pd.DataFrame):
                st.subheader(f"{label} Preview")
                st.dataframe(df.head())
            else:
                st.info(f"{label} is not uploaded or invalid. Skipping preview.")

        # -----------------------------
        # Cleaning options
        # -----------------------------
        st.subheader("General Cleaning Operations")
        cleaning_options = st.multiselect(
            "Select general cleaning operations to apply",
            [
                "Drop duplicate rows",
                "Fill missing numeric values with median",
                "Fill missing categorical values with mode",
                "Trim whitespace from string columns",
                "Remove columns with all nulls"
            ],
            default=["Drop duplicate rows"]
        )

        st.subheader("Outlier Handling (Numeric Columns)")
        outlier_handling_method = st.radio(
            "Select an outlier handling method (if any)",
            ["None", "Remove outliers (IQR)", "Cap outliers (IQR)"],
            index=0
        )

        # Custom dataset names
        custom_names = {
            "cleaned_a": st.text_input("Name Dataset A (optional)", value=st.session_state.get('cleaned_a_name', 'Dataset A')),
            "cleaned_b": st.text_input("Name Dataset B (optional)", value=st.session_state.get('cleaned_b_name', 'Dataset B'))
        }

        # -----------------------------
        # Run cleaning
        # -----------------------------
        if st.button("Run Cleaning"):
            for ds_name, label, _ in datasets:
                df = st.session_state.get(ds_name)
                cleaned_df, ops = clean_dataset(df, label, cleaning_options, outlier_handling_method)

                if cleaned_df is not None:
                    # Save to session state
                    st.session_state[ds_name] = cleaned_df
                    st.session_state[f"{ds_name}_name"] = custom_names[ds_name]
                    st.session_state[f"{ds_name}_operations"] = ops
                    st.session_state[f"{ds_name}_saved"] = True

                    # Display summary
                    new_shape = cleaned_df.shape
                    st.subheader(f"{label} Cleaning Summary")
                    if ops:
                        st.write("**Changes applied:**")
                        for op in ops:
                            st.markdown(f"- {op}")
                    else:
                        st.info("No changes were necessary based on selected options.")
                    st.write(f"**Original shape:** {df.shape}, **New shape:** {new_shape}")
                    st.dataframe(cleaned_df.head())
                else:
                    st.info(f"{label} skipped.")

        # -----------------------------
        # Optional debug
        # -----------------------------
        st.write("DEBUG: Session state types after cleaning")
        st.write(f"cleaned_a type = {type(st.session_state.get('cleaned_a'))}")
        st.write(f"cleaned_b type = {type(st.session_state.get('cleaned_b'))}")

        # -----------------------------
        # Export Cleaned Data Feature
        # -----------------------------
        st.subheader("Export Cleaned Data")
        export_formats = {'CSV': ('text/csv', 'csv'), 'Excel': ('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'xlsx')}

        for ds_key, ds_label, ds_name_key in datasets:
            df_to_export = st.session_state.get(ds_key)
            if isinstance(df_to_export, pd.DataFrame):
                actual_ds_name = st.session_state.get(ds_name_key, ds_label)
                st.markdown(f"#### Export {actual_ds_name}")
                col1, col2 = st.columns(2)
                with col1:
                    selected_format = st.radio(f"Select format for {actual_ds_name}", options=list(export_formats.keys()), key=f"export_format_{ds_key}")
                with col2:
                    if selected_format:
                        buffer = io.BytesIO()
                        mime_type, extension = export_formats[selected_format]
                        file_name = f"cleaned_{actual_ds_name.replace(' ', '_')}.{extension}"

                        if selected_format == 'CSV':
                            df_to_export.to_csv(buffer, index=False)
                        elif selected_format == 'Excel':
                            df_to_export.to_excel(buffer, index=False)

                        buffer.seek(0)
                        st.download_button(
                            label=f"Download {actual_ds_name} as {selected_format}",
                            data=buffer,
                            file_name=file_name,
                            mime=mime_type,
                            key=f"download_button_{ds_key}"
                        )
            else:
                st.info(f"No cleaned data available for {ds_label} to export.")

        # -------------------------------------
        # Advanced Data Transformations
        # -------------------------------------
        st.subheader("Advanced Data Transformations")

        # Selector for dataset to transform
        available_datasets_for_transform = []
        if st.session_state.cleaned_a is not None: available_datasets_for_transform.append(st.session_state.cleaned_a_name)
        if st.session_state.cleaned_b is not None: available_datasets_for_transform.append(st.session_state.cleaned_b_name)

        if not available_datasets_for_transform:
            st.info("No datasets available for advanced transformations. Please upload and clean data first.")
        else:
            selected_transform_dataset_name = st.selectbox(
                "Select dataset for advanced transformations",
                options=available_datasets_for_transform,
                key="select_transform_ds"
            )

            ds_key_transform = "cleaned_a" if selected_transform_dataset_name == st.session_state.cleaned_a_name else "cleaned_b"
            df_transform = st.session_state[ds_key_transform]
            ds_ops_key = f"{ds_key_transform}_operations"

            if df_transform is None: # Should not happen with the check above, but for safety
                st.error("Selected dataset not found for transformation.")
            else:
                # --- One-Hot Encoding ---
                with st.expander("One-Hot Encoding"):
                    cat_cols_transform = df_transform.select_dtypes(include=['object', 'category']).columns.tolist()
                    if cat_cols_transform:
                        cols_to_encode = st.multiselect(
                            f"Select categorical columns to one-hot encode in {selected_transform_dataset_name}",
                            options=cat_cols_transform,
                            key=f"ohe_cols_{ds_key_transform}"
                        )
                        if st.button(f"Apply One-Hot Encoding to {selected_transform_dataset_name}", key=f"apply_ohe_{ds_key_transform}"):
                            if cols_to_encode:
                                try:
                                    df_encoded = pd.get_dummies(df_transform, columns=cols_to_encode, drop_first=True)
                                    st.session_state[ds_key_transform] = df_encoded
                                    st.session_state[ds_ops_key].append(f"One-hot encoded columns: {', '.join(cols_to_encode)}")
                                    st.success(f"Successfully applied one-hot encoding to {selected_transform_dataset_name}.")
                                    st.dataframe(df_encoded.head())
                                except Exception as e:
                                    st.error(f"Error applying one-hot encoding: {e}")
                            else:
                                st.warning("Please select at least one column for one-hot encoding.")
                    else:
                        st.info(f"No categorical columns found in {selected_transform_dataset_name} for one-hot encoding.")

                # --- Feature Scaling ---
                with st.expander("Feature Scaling"):
                    numeric_cols_transform = df_transform.select_dtypes(include=np.number).columns.tolist()
                    if numeric_cols_transform:
                        cols_to_scale = st.multiselect(
                            f"Select numeric columns to scale in {selected_transform_dataset_name}",
                            options=numeric_cols_transform,
                            key=f"scale_cols_{ds_key_transform}"
                        )
                        scaling_method = st.radio(
                            f"Select scaling method for {selected_transform_dataset_name}",
                            options=["None", "Standardization (StandardScaler)", "Normalization (MinMaxScaler)"],
                            key=f"scaling_method_{ds_key_transform}"
                        )
                        if st.button(f"Apply Feature Scaling to {selected_transform_dataset_name}", key=f"apply_scaling_{ds_key_transform}"):
                            if scaling_method != "None" and cols_to_scale:
                                try:
                                    df_scaled = df_transform.copy()
                                    if scaling_method == "Standardization (StandardScaler)":
                                        scaler = StandardScaler()
                                        df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
                                        op_desc = "Standardized"
                                    elif scaling_method == "Normalization (MinMaxScaler)":
                                        scaler = MinMaxScaler()
                                        df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
                                        op_desc = "Normalized"

                                    st.session_state[ds_key_transform] = df_scaled
                                    st.session_state[ds_ops_key].append(f"{op_desc} columns: {', '.join(cols_to_scale)}")
                                    st.success(f"Successfully applied {scaling_method} to {selected_transform_dataset_name}.")
                                    st.dataframe(df_scaled.head())
                                except Exception as e:
                                    st.error(f"Error applying feature scaling: {e}")
                            elif scaling_method != "None" and not cols_to_scale:
                                st.warning("Please select at least one numeric column to scale.")
                            else:
                                st.info("No scaling method applied (selected 'None').")
                    else:
                        st.info(f"No numeric columns found in {selected_transform_dataset_name} for feature scaling.")

                # --- Custom Column Creation ---
                with st.expander("Create Custom Column"):
                    new_col_name = st.text_input(f"New Column Name for {selected_transform_dataset_name}", key=f"new_col_name_{ds_key_transform}")
                    formula = st.text_input(
                        f"Formula (e.g., 'col1 + col2', 'np.log(col3)') for {selected_transform_dataset_name}",
                        key=f"formula_{ds_key_transform}"
                    )
                    if st.button(f"Create Custom Column in {selected_transform_dataset_name}", key=f"apply_custom_col_{ds_key_transform}"):
                        if new_col_name and formula:
                            if new_col_name in df_transform.columns:
                                st.error(f"Column '{new_col_name}' already exists. Please choose a different name.")
                            else:
                                try:
                                    # Using df.eval for safer formula evaluation
                                    df_transformed_temp = df_transform.copy()
                                    # Use .loc to avoid SettingWithCopyWarning
                                    df_transformed_temp.loc[:, new_col_name] = df_transformed_temp.eval(formula)

                                    st.session_state[ds_key_transform] = df_transformed_temp
                                    st.session_state[ds_ops_key].append(f"Created custom column '{new_col_name}' using formula: {formula}")
                                    st.success(f"Successfully created custom column '{new_col_name}' in {selected_transform_dataset_name}.")
                                    st.dataframe(df_transformed_temp.head())
                                except Exception as e:
                                    st.error(f"Error creating custom column: {e}. Please check your formula and column names.")
                        else:
                            st.warning("Please provide both a new column name and a formula.")

    # -----------------------------
        # tab 3
    # -----------------------------
with tab3:
    st.title("Exploratory Data Analysis (EDA)")

    # Ensure saved_charts exists
    if "saved_charts" not in st.session_state:
        st.session_state["saved_charts"] = []

    # Datasets and friendly names
    datasets = [
        ("cleaned_a", st.session_state.get("cleaned_a_name", "Dataset A")),
        ("cleaned_b", st.session_state.get("cleaned_b_name", "Dataset B"))
    ]

    available = [(key, name) for key, name in datasets if isinstance(st.session_state.get(key), pd.DataFrame)]
    if not available:
        st.warning("Please upload & clean at least one dataset in Tabs 1–2 before running EDA.")
    else:
        # Column layout: left = overview, center = chart, right = queue
        left_col, center_col, right_col = st.columns([2, 3, 1])

        # ------------------ LEFT: Dataset selection & overview ------------------
        with left_col:
            st.subheader("Dataset selection & overview")
            display_names = [name for _, name in available]
            chosen_name = st.selectbox("Choose dataset", options=display_names, key="eda_choose_ds")
            ds_key = next(key for key, name in available if name == chosen_name)
            df = st.session_state.get(ds_key)

            if df is None:
                st.error("Dataset not available. Please return to Cleaning (Tab 2).")
            else:
                st.markdown(f"**{chosen_name}** — rows × cols: **{df.shape[0]} × {df.shape[1]}**")
                st.write("Columns:", list(df.columns))

                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

                # ------------------ Interactive Filtering ------------------
                st.markdown("### Interactive Filters")
                filtered_df = df.copy()
                current_filters = []

                # Numeric Filters
                if numeric_cols:
                    with st.expander("Filter Numeric Columns"):
                        for col in numeric_cols:
                            min_val, max_val = float(df[col].min()), float(df[col].max())
                            selected_min, selected_max = st.slider(
                                f"Select range for {col}",
                                min_value=min_val,
                                max_value=max_val,
                                value=(min_val, max_val),
                                key=f"filter_slider_{ds_key}_{col}"
                            )
                            if selected_min != min_val or selected_max != max_val:
                                filtered_df = filtered_df[(filtered_df[col] >= selected_min) & (filtered_df[col] <= selected_max)]
                                current_filters.append(f"{col} between {selected_min:.2f} and {selected_max:.2f}")

                # Categorical Filters
                if cat_cols:
                    with st.expander("Filter Categorical Columns"):
                        for col in cat_cols:
                            unique_vals = df[col].unique().tolist()
                            selected_vals = st.multiselect(
                                f"Select categories for {col}",
                                options=unique_vals,
                                default=unique_vals,
                                key=f"filter_multiselect_{ds_key}_{col}"
                            )
                            if set(selected_vals) != set(unique_vals):
                                filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
                                current_filters.append(f"{col} in ({', '.join(map(str, selected_vals))})")

                st.markdown("--- ")
                st.subheader("Filtered Data Preview")
                if not current_filters:
                    st.info("No filters applied. Showing full dataset.")
                else:
                    st.write("**Applied Filters:**")
                    for f in current_filters:
                        st.write(f"- {f}")
                st.markdown(f"**Filtered {chosen_name}** — rows × cols: **{filtered_df.shape[0]} × {filtered_df.shape[1]}**")
                if not filtered_df.empty:
                    st.dataframe(filtered_df.head())
                else:
                    st.warning("Filtered data is empty.")


                # Expanders for summaries (now using filtered_df)
                with st.expander("Missing values in Filtered Data"):
                    missing = filtered_df.isna().sum()
                    if missing.sum() > 0:
                        st.dataframe(missing[missing > 0].sort_values(ascending=False))
                    else:
                        st.info("No missing values in filtered data.")

                with st.expander("Numeric summary of Filtered Data"):
                    if numeric_cols:
                        st.dataframe(filtered_df[numeric_cols].describe().T)
                    else:
                        st.info("No numeric columns in filtered data.")

                with st.expander("Categorical summary of Filtered Data"):
                    if cat_cols:
                        sel_cat = st.selectbox("Pick a categorical column", options=cat_cols, key=f"{ds_key}_cat_filtered")
                        vc = filtered_df[sel_cat].value_counts(dropna=False)
                        st.dataframe(vc)
                    else:
                        st.info("No categorical columns in filtered data.")

                with st.expander("Row / Column Inspector of Filtered Data"):
                    col_to_view = st.selectbox("Pick a column to inspect", options=filtered_df.columns, key=f"{ds_key}_inspect_col_filtered")
                    st.dataframe(filtered_df[[col_to_view]].head(10))

                if st.button("Download column summary CSV of Filtered Data", key=f"{ds_key}_download_summary_filtered"):
                    summary = [{
                        "column": c,
                        "dtype": str(filtered_df[c].dtype),
                        "n_unique": int(filtered_df[c].nunique(dropna=True)),
                        "n_missing": int(filtered_df[c].isna().sum())
                    } for c in filtered_df.columns]
                    summary_df = pd.DataFrame(summary)
                    buffer = io.BytesIO()
                    summary_df.to_csv(buffer, index=False)
                    buffer.seek(0)
                    st.download_button("Download CSV", data=buffer, file_name=f"{ds_key}_summary_filtered.csv")

        # ------------------ CENTER: Charts (now using filtered_df) ------------------
        with center_col:
            st.subheader("Charts")
            chart_options = [
                "None",
                "Histogram (single numeric)",
                "Boxplot (single numeric)",
                "Scatter (numeric X & Y)",
                "Correlation heatmap (numeric columns)"
            ]
            chart_choice = st.selectbox("Choose chart", options=chart_options, key=f"{ds_key}_chart_choice")
            chart_params = {}

            # Chart-specific controls
            if chart_choice == "Histogram (single numeric)" and numeric_cols:
                x_col = st.selectbox("Numeric column", options=numeric_cols, key=f"{ds_key}_hist_x")
                bins = st.number_input("Bins", min_value=5, max_value=500, value=30, step=1, key=f"{ds_key}_hist_bins")
                color_col = None
                if cat_cols:
                    color_col = st.selectbox("Color by (categorical)", options=[None]+cat_cols, key=f"{ds_key}_hist_color")
                chart_params.update({"x_col": x_col, "bins": bins, "color_col": color_col})

            elif chart_choice == "Boxplot (single numeric)" and numeric_cols:
                y_col = st.selectbox("Numeric column", options=numeric_cols, key=f"{ds_key}_box_y")
                group_col = None
                if cat_cols:
                    group_col = st.selectbox("Group by (categorical)", options=[None]+cat_cols, key=f"{ds_key}_box_group")
                chart_params.update({"y_col": y_col, "group_col": group_col})

            elif chart_choice == "Scatter (numeric X & Y)" and len(numeric_cols)>=2:
                x_col = st.selectbox("X axis", options=numeric_cols, key=f"{ds_key}_scatter_x")
                y_col = st.selectbox("Y axis", options=[c for c in numeric_cols if c != x_col], key=f"{(ds_key)}_scatter_y")
                color_col = None
                if cat_cols:
                    color_col = st.selectbox("Color by (categorical)", options=[None]+cat_cols, key=f"{(ds_key)}_scatter_color")
                chart_params.update({"x_col": x_col, "y_col": y_col, "color_col": color_col})

            elif chart_choice == "Correlation heatmap (numeric columns)" and len(numeric_cols)>=2:
                chart_params.update({})

            # Show chart & save to PDF queue
            st.write("")
            col_show, col_save = st.columns([1,1])
            caption = st.text_input("Optional caption for PDF", key=f"{ds_key}_chart_caption")
            with col_show:
                if st.button("Show chart", key=f"{ds_key}_show_chart"):
                    fig = None
                    if not filtered_df.empty:
                        try:
                            if chart_choice == "Histogram (single numeric)":
                                if chart_params.get("color_col"):
                                    fig = px.histogram(filtered_df, x=chart_params["x_col"], color=chart_params["color_col"], nbins=chart_params["bins"])
                                else:
                                    fig = px.histogram(filtered_df, x=chart_params["x_col"], nbins=chart_params["bins"])
                            elif chart_choice == "Boxplot (single numeric)":
                                if chart_params.get("group_col"):
                                    fig = px.box(filtered_df, x=chart_params["group_col"], y=chart_params["y_col"])
                                else:
                                    fig = px.box(filtered_df, y=chart_params["y_col"])
                            elif chart_choice == "Scatter (numeric X & Y)":
                                if chart_params.get("color_col"):
                                    fig = px.scatter(filtered_df, x=chart_params["x_col"], y=chart_params["y_col"], color=filtered_df[chart_params["color_col"]].astype(str))
                                else:
                                    fig = px.scatter(filtered_df, x=chart_params["x_col"], y=chart_params["y_col"])
                            elif chart_choice == "Correlation heatmap (numeric columns)":
                                corr = filtered_df[numeric_cols].corr()
                                fig = px.imshow(corr, text_auto=True)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error rendering chart: {e}")
                    else:
                        st.warning("Cannot generate chart: Filtered data is empty.")

            with col_save:
                if st.button("Save chart to PDF", key=f"{ds_key}_save_chart"):
                    saved = {"ds_key": ds_key, "ds_name": chosen_name,
                             "chart_type": chart_choice, "params": chart_params,
                             "caption": caption, "time": datetime.utcnow().isoformat(),
                             "filters_applied": current_filters # Save current filters with chart
                            }
                    st.session_state["saved_charts"].append(saved)
                    st.success("Chart saved to PDF queue")

        # ------------------ RIGHT: PDF Queue ------------------
        with right_col:
            st.subheader("PDF Queue")
            queue = st.session_state.get("saved_charts", [])
            if not queue:
                st.info("No charts queued yet.")
            else:
                for i, c in enumerate(queue,1):
                    st.markdown(f"**{i}. {c['ds_name']}** — {c['chart_type']}")
                    if c.get("caption"):
                        st.caption(c["caption"])
                    remove_key = f"remove_{i}_{ds_key}"
                    if st.button("Remove", key=remove_key):
                        st.session_state["saved_charts"].pop(i-1)
                        st.experimental_rerun()

            if queue:
                if st.button("Clear PDF queue", key=f"{ds_key}_clear_queue"):
                    st.session_state["saved_charts"] = []
                    st.success("PDF queue cleared")
# ------------------ Tab 4 ------------------
with tab4:
    st.header("Compare & Contrast")

    # Safety: get cleaned datasets
    A = st.session_state.get("cleaned_a")
    B = st.session_state.get("cleaned_b")
    name_a = st.session_state.get("cleaned_a_name", "Dataset A")
    name_b = st.session_state.get("cleaned_b_name", "Dataset B")

    # Only proceed if both datasets exist
    if not isinstance(A, pd.DataFrame) or not isinstance(B, pd.DataFrame):
        st.info("Please upload and clean a second dataset to use this feature.")
        st.stop()  # Stop rendering further
    else:
        st.write(f"**Datasets available:** {name_a} (A), {name_b} (B)")

        # Matching keys
        common_cols = list(set(A.columns).intersection(set(B.columns)))
        st.write("Common columns detected:", common_cols or "(none)")

        auto_key = None
        for cand in ["id", "ID", "Id", "key", "Key", "email", "Email"]:
            if cand in common_cols:
                auto_key = cand
                break

        use_auto = False
        if auto_key:
            use_auto = st.checkbox(f"Auto-select '{auto_key}' as join key", value=True)

        selected_keys = st.multiselect("Select key column(s) to match rows", options=common_cols,
                                       default=[auto_key] if (auto_key and use_auto) else (common_cols[:1] if common_cols else []))
        if not selected_keys:
            st.warning("Select at least one key column to perform row-level comparisons.")
            st.stop()

        dupA = A.duplicated(subset=selected_keys, keep=False).sum()
        dupB = B.duplicated(subset=selected_keys, keep=False).sum()
        st.write(f"Key duplicates: {name_a}: {dupA}/{A.shape[0]}, {name_b}: {dupB}/{B.shape[0]}")

        # Merge and compare
        merged = A.merge(B, on=selected_keys, how="outer", indicator="_merge", suffixes=('_A', '_B'))
        only_a = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
        only_b = merged[merged["_merge"] == "right_only"].drop(columns=["_merge"])
        both = merged[merged["_merge"] == "both"].drop(columns=["_merge"])

        # Summary metrics
        st.markdown("### Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Only in {name_a}", f"{only_a.shape[0]:,}")
        c2.metric(f"Only in {name_b}", f"{only_b.shape[0]:,}")
        c3.metric("Matched (both)", f"{both.shape[0]:,}")

        # Expanders with previews and export
        for label, df_part, fname in [
            (f"Rows only in {name_a}", only_a, f"only_in_{name_a}.csv"),
            (f"Rows only in {name_b}", only_b, f"only_in_{name_b}.csv"),
            ("Rows in both", both, "matched_rows.csv")
        ]:
            with st.expander(f"Preview: {label} ({df_part.shape[0]})", expanded=False):
                if not df_part.empty:
                    st.dataframe(df_part.head(200))
                    buf = io.BytesIO()
                    df_part.to_csv(buf, index=False)
                    buf.seek(0)
                    st.download_button(f"Download {fname}", data=buf, file_name=fname)
                else:
                    st.info(f"No rows for {label}.")

        # Column comparison
        st.markdown("### Column presence comparison")
        cols_a = set(A.columns)
        cols_b = set(B.columns)
        only_cols_a = sorted(list(cols_a - cols_b))
        only_cols_b = sorted(list(cols_b - cols_a))
        common = sorted(list(cols_a & cols_b))
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"Columns only in {name_a} ({len(only_cols_a)})")
            st.write(only_cols_a or "None")
        with c2:
            st.subheader(f"Columns only in {name_b} ({len(only_cols_b)})")
            st.write(only_cols_b or "None")

        # Numeric differences
        st.markdown("### Numeric differences for common numeric columns")
        numeric_common = [c for c in common if pd.api.types.is_numeric_dtype(A[c]) and pd.api.types.is_numeric_dtype(B[c])]
        numeric_comparison_stats = []
        if numeric_common:
            stats = []
            for col in numeric_common:
                a_series = A.set_index(selected_keys)[col] if selected_keys else A[col]
                b_series = B.set_index(selected_keys)[col] if selected_keys else B[col]
                joined = a_series.to_frame("A").join(b_series.to_frame("B"), how="inner").dropna()
                if not joined.empty:
                    diff = joined["A"] - joined["B"]
                    stats.append({
                        "column": col,
                        "n_compared": int(joined.shape[0]),
                        "mean_diff": float(diff.mean()),
                        "median_diff": float(diff.median()),
                        "std_diff": float(diff.std())
                    })
            if stats:
                st.dataframe(pd.DataFrame(stats).set_index("column"))
                numeric_comparison_stats = stats
            else:
                st.info("No overlapping numeric values to compare.")
        else:
            st.info("No numeric columns in common.")

        # -------------------------------------
        # Statistical Comparison Tests
        # -------------------------------------
        st.markdown("### Statistical Comparison Tests")
        t_test_results = []
        chi2_test_results = []

        # T-tests for numeric columns
        numeric_common_for_ttest = [c for c in common if pd.api.types.is_numeric_dtype(A[c]) and pd.api.types.is_numeric_dtype(B[c])]
        if numeric_common_for_ttest:
            selected_numeric_cols_ttest = st.multiselect(
                "Select numeric columns for T-tests",
                options=numeric_common_for_ttest,
                key="ttest_cols"
            )
            if st.button("Run T-tests"):
                if selected_numeric_cols_ttest:
                    st.subheader("T-test Results (Comparing Means)")
                    for col in selected_numeric_cols_ttest:
                        # Drop NaNs for the t-test
                        data_a = A[col].dropna()
                        data_b = B[col].dropna()

                        if len(data_a) > 1 and len(data_b) > 1:
                            t_stat, p_val = ttest_ind(data_a, data_b, equal_var=False) # Welch's t-test
                            st.write(f"**Column: {col}**")
                            st.write(f"  - T-statistic: {t_stat:.3f}")
                            st.write(f"  - P-value: {p_val:.3e}")
                            if p_val < 0.05:
                                st.success(f"  - **Significant difference (p < 0.05)** between the means of {name_a} and {name_b} for {col}.")
                            else:
                                st.info(f"  - No significant difference (p >= 0.05) between the means of {name_a} and {name_b} for {col}.")
                            t_test_results.append({"column": col, "t_stat": float(t_stat), "p_val": float(p_val)})
                        else:
                            st.warning(f"Not enough non-null data to perform t-test for column {col}.")
                else:
                    st.warning("Please select at least one numeric column for T-tests.")
        else:
            st.info("No common numeric columns available for T-tests.")

        # Chi-squared tests for categorical columns
        cat_common_for_chi2 = [c for c in common if pd.api.types.is_object_dtype(A[c]) or pd.api.types.is_categorical_dtype(A[c])]
        if cat_common_for_chi2:
            selected_cat_cols_chi2 = st.multiselect(
                "Select categorical columns for Chi-squared tests",
                options=cat_common_for_chi2,
                key="chi2_cols"
            )
            if st.button("Run Chi-squared Tests"):
                if selected_cat_cols_chi2:
                    st.subheader("Chi-squared Test Results (Comparing Distributions)")
                    for col in selected_cat_cols_chi2:
                        # Create a contingency table
                        contingency_table = pd.crosstab(A[col], B[col])

                        if contingency_table.empty or contingency_table.sum().sum() == 0:
                            st.warning(f"Contingency table for column {col} is empty or has no data. Skipping Chi-squared test.")
                            continue

                        try:
                            chi2, p_val, dof, expected = chi2_contingency(contingency_table)
                            st.write(f"**Column: {col}**")
                            st.write(f"  - Chi-squared statistic: {chi2:.3f}")
                            st.write(f"  - P-value: {p_val:.3e}")
                            if p_val < 0.05:
                                st.success(f"  - **Significant difference (p < 0.05)** between the distributions of {name_a} and {name_b} for {col}.")
                            else:
                                st.info(f"  - No significant difference (p >= 0.05) between the distributions of {name_a} and {name_b} for {col}.")
                            chi2_test_results.append({"column": col, "chi2_stat": float(chi2), "p_val": float(p_val)})
                        except ValueError as ve:
                            st.error(f"Error performing Chi-squared test for column {col}: {ve}. This often happens if the contingency table contains zero values or insufficient data. Consider combining sparse categories.")

                else:
                    st.warning("Please select at least one categorical column for Chi-squared tests.")
        else:
            st.info("No common categorical columns available for Chi-squared tests.")

        # Save report for PDF/export
        st.session_state["compare_report"] = {
            "name_a": name_a,
            "name_b": name_b,
            "selected_keys": selected_keys,
            "counts": {"only_a": only_a.shape[0], "only_b": only_b.shape[0], "both": both.shape[0]},
            "only_cols_a": only_cols_a,
            "only_cols_b": only_cols_b,
            "numeric_comparison": numeric_comparison_stats,
            "t_test_results": t_test_results,
            "chi2_test_results": chi2_test_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        st.success("Compare completed and saved for export/PDF.")
# -----------------------------
# Tab 5: PDF Summary with Visualizations
# -----------------------------

def generate_insights_paragraph(cleaned_a, cleaned_b, name_a, name_b, compare_report, saved_charts):
    insights = []

    def analyze_df(df, df_name, ops_key):
        df_insights = []
        if isinstance(df, pd.DataFrame):
            # Cleaning operations
            operations = st.session_state.get(ops_key, [])
            if operations:
                df_insights.append(f"{df_name} underwent data cleaning operations including {', '.join(operations)}.")

            # Missing values
            missing_cols = df.isnull().sum()
            high_missing = missing_cols[missing_cols / len(df) > 0.2].index.tolist()
            if high_missing:
                df_insights.append(f"{df_name} has notable missing values in columns such as {', '.join(high_missing)}.")

            # Unique values
            n_unique = df.nunique()
            if not n_unique.empty:
                most_unique_col = n_unique.idxmax()
                df_insights.append(f"The column '{most_unique_col}' in {df_name} has the highest number of unique values ({n_unique.max():,}).")

            # Numeric variability
            numeric_cols = df.select_dtypes(include=np.number)
            if not numeric_cols.empty:
                std_devs = numeric_cols.std().sort_values(ascending=False)
                if not std_devs.empty:
                    most_variable_col = std_devs.index[0]
                    df_insights.append(f"'{most_variable_col}' is a highly variable numeric column in {df_name} with a standard deviation of {std_devs.iloc[0]:.2f}.")

        return df_insights

    # Insights for Dataset A
    a_ops_key = "cleaned_a_operations"
    if cleaned_a is not None:
        insights.extend(analyze_df(cleaned_a, name_a, a_ops_key))

    # Insights for Dataset B
    b_ops_key = "cleaned_b_operations"
    if cleaned_b is not None:
        insights.extend(analyze_df(cleaned_b, name_b, b_ops_key))

    # Comparison insights
    if compare_report and cleaned_a is not None and cleaned_b is not None:
        counts = compare_report.get('counts', {})
        insights.append(f"The comparison revealed {counts.get('both', 0):,} matched rows between {name_a} and {name_b}, with {counts.get('only_a', 0):,} rows unique to {name_a} and {counts.get('only_b', 0):,} unique to {name_b}.")

        if compare_report.get('only_cols_a'):
            insights.append(f"{name_a} has unique columns including {', '.join(compare_report['only_cols_a'])}.")
        if compare_report.get('only_cols_b'):
            insights.append(f"{name_b} has unique columns including {', '.join(compare_report['only_cols_b'])}.")

        numeric_comparison = compare_report.get('numeric_comparison')
        if isinstance(numeric_comparison, list) and numeric_comparison:
            # Define a threshold for 'significant' based on mean diff being greater than a fraction of std dev
            # This is a heuristic and can be refined.
            diff_cols = [stat['column'] for stat in numeric_comparison if stat['n_compared'] > 10 and abs(stat['mean_diff']) > (stat['std_diff'] / 4)]
            if diff_cols:
                insights.append(f"Significant numeric differences were observed in columns like {', '.join(diff_cols)} during the comparison.")

        t_test_results = compare_report.get('t_test_results')
        if t_test_results:
            significant_t_tests = [res['column'] for res in t_test_results if res['p_val'] < 0.05]
            if significant_t_tests:
                insights.append(f"T-tests indicated significant mean differences in columns such as {', '.join(significant_t_tests)}.")

        chi2_test_results = compare_report.get('chi2_test_results')
        if chi2_test_results:
            significant_chi2_tests = [res['column'] for res in chi2_test_results if res['p_val'] < 0.05]
            if significant_chi2_tests:
                insights.append(f"Chi-squared tests found significant distributional differences in categorical columns like {', '.join(significant_chi2_tests)}.")

    # Chart insights
    if saved_charts:
        insights.append(f"The report also includes {len(saved_charts)} visualizations, offering graphical insights into the data.")

    if not insights:
        return "No specific insights were generated based on the current data and analysis."

    # Combine all insights into a coherent paragraph
    return " ".join(insights)

with tab5:
    st.header("PDF Summary Report")

    # Get datasets from session state
    cleaned_a = st.session_state.get("cleaned_a")
    cleaned_b = st.session_state.get("cleaned_b")
    name_a = st.session_state.get("cleaned_a_name", "Dataset A")
    name_b = st.session_state.get("cleaned_b_name", "Dataset B")

    # Safety check: at least one dataset must exist
    if cleaned_a is None and cleaned_b is None:
        st.info("No datasets available to generate PDF. Please upload and clean datasets in Tabs 1–2.")
    else:
        st.info("This PDF will include dataset summaries and comparison (if both datasets exist), plus any saved charts.")

        # Optional notes for executive summary
        notes = st.text_area("Optional notes / observations for the report", value="")

        # Multiselect for PDF sections
        pdf_sections_options = [
            'Dataset A Summary',
            'Dataset B Summary',
            'Compare & Contrast Report',
            'Visualizations'
        ]
        selected_sections = st.multiselect(
            "Select sections to include in PDF report",
            options=pdf_sections_options,
            default=pdf_sections_options # Default to all selected
        )

        if st.button("Generate PDF Summary"):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)

            # Title page / overview
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "DataLens PDF Summary Report", ln=True, align="C")
            pdf.set_font("Arial", "", 12)
            pdf.ln(5)
            pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.cell(0, 10, f"Dataset A: {name_a}", ln=True)
            pdf.cell(0, 10, f"Dataset B: {name_b if cleaned_b is not None else 'Not uploaded'}", ln=True)
            pdf.ln(10)
            if notes:
                pdf.multi_cell(0, 8, f"Notes: {notes}")

            # Auto-generated insights (always included after initial info)
            insights_paragraph = generate_insights_paragraph(cleaned_a, cleaned_b, name_a, name_b, st.session_state.get("compare_report"), st.session_state.get("saved_charts"))
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Key Insights:", ln=True)
            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(0, 6, insights_paragraph)
            pdf.ln(5)

            def add_dataset_summary(df, dataset_name, operations_list=None):
                pdf.add_page()
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, f"{dataset_name} Summary", ln=True)
                pdf.set_font("Arial", "", 12)
                pdf.cell(0, 10, f"Shape: {df.shape[0]} rows x {df.shape[1]} columns", ln=True)

                if operations_list:
                    pdf.ln(2)
                    pdf.set_font("Arial", "B", 10)
                    pdf.cell(0, 5, "Cleaning Operations Applied:", ln=True)
                    pdf.set_font("Arial", "", 10)
                    for op in operations_list:
                        pdf.multi_cell(0, 4, f"- {op}")
                pdf.ln(5)

                # Numeric summary
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    pdf.set_font("Arial", "B", 10)
                    pdf.cell(0, 7, "Numeric Columns Summary:", ln=True)
                    pdf.set_font("Arial", "", 9)
                    for col in numeric_cols:
                        col_series = df[col]
                        pdf.cell(0, 6, f"{col} | count: {col_series.count()}, mean: {col_series.mean():.2f}, "
                                        f"median: {col_series.median():.2f}, min: {col_series.min():.2f}, "
                                        f"max: {col_series.max():.2f}, missing: {col_series.isna().sum()}", ln=True)
                else:
                    pdf.set_font("Arial", "", 10)
                    pdf.cell(0, 7, "No numeric columns.", ln=True)
                pdf.ln(3)

                # Categorical summary
                cat_cols = df.select_dtypes(include=['object', 'category']).columns
                if len(cat_cols) > 0:
                    pdf.set_font("Arial", "B", 10)
                    pdf.cell(0, 7, "Categorical Columns Summary:", ln=True)
                    pdf.set_font("Arial", "", 9)
                    for col in cat_cols:
                        col_series = df[col]
                        top_values = col_series.value_counts(dropna=False).head(3).to_dict()
                        pdf.multi_cell(0, 6, f"{col} | unique: {col_series.nunique()}, top values: {top_values}, missing: {col_series.isna().sum()}")
                else:
                    pdf.set_font("Arial", "", 10)
                    pdf.cell(0, 7, "No categorical columns.", ln=True)

            # Add summaries for available datasets based on selection
            if 'Dataset A Summary' in selected_sections and cleaned_a is not None:
                add_dataset_summary(cleaned_a, name_a, st.session_state.get("cleaned_a_operations"))
            if 'Dataset B Summary' in selected_sections and cleaned_b is not None:
                add_dataset_summary(cleaned_b, name_b, st.session_state.get("cleaned_b_operations"))

            # Compare & Contrast if both datasets exist and report is available AND selected
            compare_report = st.session_state.get("compare_report")
            if 'Compare & Contrast Report' in selected_sections and cleaned_a is not None and cleaned_b is not None and compare_report:
                pdf.add_page()
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "Compare & Contrast Report", ln=True)
                pdf.set_font("Arial", "", 12)
                pdf.ln(5)

                pdf.cell(0, 7, f"Matched rows: {compare_report['counts']['both']:,}", ln=True)
                pdf.cell(0, 7, f"Rows unique to {name_a}: {compare_report['counts']['only_a']:,}", ln=True)
                pdf.cell(0, 7, f"Rows unique to {name_b}: {compare_report['counts']['only_b']:,}", ln=True)
                pdf.ln(3)

                pdf.set_font("Arial", "B", 10)
                pdf.cell(0, 7, f"Columns unique to {name_a}:", ln=True)
                pdf.set_font("Arial", "", 10)
                pdf.multi_cell(0, 6, ", ".join(compare_report['only_cols_a']) if compare_report['only_cols_a'] else "None")
                pdf.ln(3)

                pdf.set_font("Arial", "B", 10)
                pdf.cell(0, 7, f"Columns unique to {name_b}:", ln=True)
                pdf.set_font("Arial", "", 10)
                pdf.multi_cell(0, 6, ", ".join(compare_report['only_cols_b']) if compare_report['only_cols_b'] else "None")
                pdf.ln(3)

                if isinstance(compare_report['numeric_comparison'], list) and compare_report['numeric_comparison']:
                    pdf.set_font("Arial", "B", 10)
                    pdf.cell(0, 7, "Numeric Column Differences (Mean Diff):")
                    pdf.set_font("Arial", "", 9)
                    for stat in compare_report['numeric_comparison']:
                        pdf.multi_cell(0, 6, f"- {stat['column']} | Compared: {stat['n_compared']}, Mean Diff: {stat['mean_diff']:.2f}")
                else:
                    pdf.set_font("Arial", "", 10)
                    pdf.cell(0, 7, "No specific numeric differences to report.", ln=True)
                pdf.ln(3)

                if compare_report.get('t_test_results'):
                    pdf.set_font("Arial", "B", 10)
                    pdf.cell(0, 7, "T-test Results:")
                    pdf.set_font("Arial", "", 9)
                    for res in compare_report['t_test_results']:
                        pdf.multi_cell(0, 6, f"- {res['column']} | T-stat: {res['t_stat']:.3f}, P-value: {res['p_val']:.3e} ({'Significant' if res['p_val'] < 0.05 else 'Not significant'})")
                else:
                    pdf.set_font("Arial", "", 10)
                    pdf.cell(0, 7, "No T-test results available.", ln=True)
                pdf.ln(3)

                if compare_report.get('chi2_test_results'):
                    pdf.set_font("Arial", "B", 10)
                    pdf.cell(0, 7, "Chi-squared Test Results:")
                    pdf.set_font("Arial", "", 9)
                    for res in compare_report['chi2_test_results']:
                        pdf.multi_cell(0, 6, f"- {res['column']} | Chi2-stat: {res['chi2_stat']:.3f}, P-value: {res['p_val']:.3e} ({'Significant' if res['p_val'] < 0.05 else 'Not significant'})")
                else:
                    pdf.set_font("Arial", "", 10)
                    pdf.cell(0, 7, "No Chi-squared test results available.", ln=True)
                pdf.ln(3)



            # Add saved charts if selected
            if 'Visualizations' in selected_sections and st.session_state.get("saved_charts"):
                pdf.add_page()
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "Visualizations", ln=True)
                pdf.ln(5)

                for i, chart_info in enumerate(st.session_state["saved_charts"]):
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, f"Chart {i+1}: {chart_info['ds_name']} - {chart_info['chart_type']}", ln=True)
                    pdf.set_font("Arial", "", 10)
                    if chart_info.get("caption"):
                        pdf.multi_cell(0, 5, f"Caption: {chart_info['caption']}")
                    if chart_info.get("filters_applied"): # Display filters applied to the chart
                         pdf.set_font("Arial", "I", 9)
                         pdf.multi_cell(0, 4, f"Filters: {'; '.join(chart_info['filters_applied'])}")
                         pdf.set_font("Arial", "", 10)

                    # Re-generate chart to save as image
                    df_chart = st.session_state.get(chart_info['ds_key'])
                    if df_chart is not None:
                        fig = None
                        try:
                            # Re-apply filters to the original df for chart generation if any were saved with the chart
                            current_df_for_chart = df_chart.copy()
                            if chart_info.get("filters_applied"):
                                for f_str in chart_info["filters_applied"]:
                                    # This parsing is a bit rudimentary; a more robust solution might store filter params directly
                                    if "between" in f_str:
                                        col = f_str.split(" between ")[0]
                                        min_val = float(f_str.split(" between ")[1].split(" and ")[0])
                                        max_val = float(f_str.split(" and ")[1])
                                        current_df_for_chart = current_df_for_chart[(current_df_for_chart[col] >= min_val) & (current_df_for_chart[col] <= max_val)]
                                    elif "in (" in f_str:
                                        col = f_str.split(" in (")[0]
                                        vals_str = f_str.split(" in (")[1][:-1] # remove closing parenthesis
                                        vals = [v.strip() for v in vals_str.split(',')]
                                        current_df_for_chart = current_df_for_chart[current_df_for_chart[col].astype(str).isin(vals)]

                            if current_df_for_chart.empty:
                                pdf.set_font("Arial", "I", 10)
                                pdf.multi_cell(0, 5, "Filtered data for this chart is empty.")
                                continue

                            numeric_cols_chart = current_df_for_chart.select_dtypes(include=np.number).columns.tolist()
                            cat_cols_chart = current_df_for_chart.select_dtypes(include=['object', 'category']).columns.tolist()

                            if chart_info['chart_type'] == "Histogram (single numeric)":
                                if chart_info['params'].get("color_col"):
                                    fig = px.histogram(current_df_for_chart, x=chart_info['params']["x_col"], color=chart_info['params']["color_col"], nbins=chart_info['params']["bins"])
                                else:
                                    fig = px.histogram(current_df_for_chart, x=chart_info['params']["x_col"], nbins=chart_info['params']["bins"])
                            elif chart_info['chart_type'] == "Boxplot (single numeric)":
                                if chart_info['params'].get("group_col"):
                                    fig = px.box(current_df_for_chart, x=chart_info['params']["group_col"], y=chart_info['params']["y_col"])
                                else:
                                    fig = px.box(current_df_for_chart, y=chart_info['params']["y_col"])
                            elif chart_info['chart_type'] == "Scatter (numeric X & Y)":
                                if chart_info['params'].get("color_col"):
                                    fig = px.scatter(current_df_for_chart, x=chart_info['params']["x_col"], y=chart_info['params']["y_col"], color=current_df_for_chart[chart_info['params']["color_col"]].astype(str))
                                else:
                                    fig = px.scatter(current_df_for_chart, x=chart_info['params']["x_col"], y=chart_info['params']["y_col"])
                            elif chart_info['chart_type'] == "Correlation heatmap (numeric columns)":
                                corr = current_df_for_chart[numeric_cols_chart].corr()
                                fig = px.imshow(corr, text_auto=True)

                            if fig:
                                img_bytes = BytesIO()
                                fig.write_image(img_bytes, format="png")
                                img_bytes.seek(0)
                                pdf.image(img_bytes, x=10, w=180) # Adjust x and w as needed
                                pdf.ln(5)
                        except Exception as e:
                            pdf.set_font("Arial", "I", 10)
                            pdf.multi_cell(0, 5, f"Error displaying chart: {e}")
                    else:
                        pdf.set_font("Arial", "I", 10)
                        pdf.multi_cell(0, 5, "Original dataset for this chart is no longer available.")
                    pdf.ln(10)


            # Output PDF to buffer and provide download
            pdf_buffer = io.BytesIO()
            pdf.output(pdf_buffer)
            pdf_buffer.seek(0)
            st.download_button("Download PDF Summary", data=pdf_buffer, file_name="data_summary.pdf")