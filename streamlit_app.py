import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
from fpdf import FPDF
from datetime import datetime
import os

st.set_page_config(layout="wide")
st.title("DataLens")

# -----------------------------
# Session state placeholders
# -----------------------------
for key in [
    "cleaned_a", "cleaned_b", "cleaned_a_saved", "cleaned_b_saved",
    "cleaned_a_name", "cleaned_b_name", "cleaned_a_operations", "cleaned_b_operations",
    "compare_report"
]:
    if key not in st.session_state:
        st.session_state[key] = None
# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Upload", "Cleaning", "EDA", "Compare & Contrast", "Export", "PDF Summary"
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
# Tab 2: Cleaning
# -----------------------------
with tab2:
    st.header("Data Cleaning")

    # -----------------------------
    # Safe cleaning function
    # -----------------------------
    def clean_dataset(df, label, cleaning_options):
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
                cleaned_df, ops = clean_dataset(df, label, cleaning_options)

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
