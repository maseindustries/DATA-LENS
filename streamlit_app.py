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
# Tab 2: EDA
# -----------------------------
with tab2:
    st.header("Exploratory Data Analysis")
    dataset_choice = st.selectbox("Select dataset for EDA", ["Dataset A","Dataset B"])
    df = st.session_state.cleaned_a if dataset_choice=="Dataset A" else st.session_state.cleaned_b

    if df is not None:
        st.subheader("Basic Stats")
        st.dataframe(df.describe(include='all'))

        if st.checkbox("Show Pivot Table"):
            index_col = st.selectbox("Select index", df.columns)
            value_col = st.selectbox("Select values", df.columns)
            pivot = pd.pivot_table(df, index=index_col, values=value_col, aggfunc='mean')
            st.dataframe(pivot)

# -----------------------------
# Tab 3: Visualization
# -----------------------------
with tab3:
    st.header("Visualization")
    dataset_choice = st.selectbox("Select dataset", ["Dataset A","Dataset B"], key="viz_dataset")
    df = st.session_state.cleaned_a if dataset_choice=="Dataset A" else st.session_state.cleaned_b

    if df is not None:
        chart_type = st.selectbox("Select Chart Type", ["Histogram","Box","Scatter"])
        col_x = st.selectbox("X-axis", df.columns)
        col_y = st.selectbox("Y-axis (for scatter only)", df.columns)
        if st.button("Generate Chart"):
            fig = None
            if chart_type=="Histogram":
                fig = px.histogram(df, x=col_x)
            elif chart_type=="Box":
                fig = px.box(df, y=col_x)
            elif chart_type=="Scatter":
                fig = px.scatter(df, x=col_x, y=col_y)
            if fig:
                st.plotly_chart(fig)

# -----------------------------
# Tab 4: Compare & Contrast
# -----------------------------
with tab4:
    st.header("Compare & Contrast")
    if st.session_state.cleaned_a is None:
        st.warning("Upload Dataset A first")
    else:
        df_a = st.session_state.cleaned_a
        df_b = st.session_state.cleaned_b
        common_cols = list(set(df_a.columns).intersection(df_b.columns)) if df_b is not None else []

        compare_type = st.selectbox("Select Compare Type", ['Row presence check','Cell-by-cell comparison','Summary compare','Schema compare'])
        key_col = None
        if compare_type=='Row presence check' and common_cols:
            key_col = st.selectbox("Key column (optional)", common_cols)

        if st.button("Run Compare"):
            report = None
            if compare_type=='Row presence check' and df_b is not None:
                if key_col:
                    ids_a = set(df_a[key_col])
                    ids_b = set(df_b[key_col])
                    only_in_a = ids_a - ids_b
                    only_in_b = ids_b - ids_a
                    report = pd.DataFrame({
                        f'Only in A ({key_col})': list(only_in_a) + [None]*(max(len(only_in_a), len(only_in_b))-len(only_in_a)),
                        f'Only in B ({key_col})': list(only_in_b) + [None]*(max(len(only_in_a), len(only_in_b))-len(only_in_b))
                    })
                else:
                    st.info("No key column selected. Skipping row presence check.")
            elif compare_type=='Schema compare' and df_b is not None:
                cols_only_a = list(set(df_a.columns)-set(df_b.columns))
                cols_only_b = list(set(df_b.columns)-set(df_a.columns))
                max_len = max(len(cols_only_a), len(cols_only_b))
                cols_only_a += [None]*(max_len-len(cols_only_a))
                cols_only_b += [None]*(max_len-len(cols_only_b))
                report = pd.DataFrame({'Columns only in A': cols_only_a,'Columns only in B': cols_only_b})
            elif compare_type=='Cell-by-cell comparison' and df_b is not None:
                diffs=[]
                max_len = max(len(df_a), len(df_b))
                df_a_r = df_a.reindex(range(max_len))
                df_b_r = df_b.reindex(range(max_len))
                for col in common_cols:
                    col_a = df_a_r[col]
                    col_b = df_b_r[col]
                    if pd.api.types.is_numeric_dtype(col_a):
                        diffs.append({'Column':col,'Mean Difference':col_a.mean()-col_b.mean(),'Count Differences':(col_a!=col_b).sum()})
                    else:
                        diffs.append({'Column':col,'Mismatched Count':(col_a!=col_b).sum()})
                report = pd.DataFrame(diffs)
            else:  # Summary compare
                report = pd.concat([df_a.describe(include='all'), df_b.describe(include='all')], keys=['Dataset A','Dataset B']) if df_b is not None else df_a.describe(include='all')
            st.session_state.compare_report = report
            st.dataframe(report)
            buffer = io.BytesIO()
            report.to_excel(buffer, index=True)
            st.download_button("Export Comparison Report", data=buffer, file_name="comparison_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -----------------------------
# Tab 5: Modeling
# -----------------------------
with tab5:
    st.header("Modeling & Prediction")
    if st.session_state.cleaned_a is not None:
        target_col = st.selectbox("Select target column", st.session_state.cleaned_a.columns)
        features = st.multiselect("Select features", [c for c in st.session_state.cleaned_a.columns if c!=target_col], default=[c for c in st.session_state.cleaned_a.columns if c!=target_col])
        model_choice = st.selectbox("Select model", ["Random Forest","Logistic Regression"])
        test_size = st.slider("Test set size (%)",10,50,20)
        n_estimators = st.slider("Random Forest n_estimators",50,500,100) if model_choice=="Random Forest" else None

        if st.button("Train Model"):
            df = st.session_state.cleaned_a.dropna(subset=[target_col]+features)
            X = pd.get_dummies(df[features])
            y = df[target_col]
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size/100,random_state=42)
            if model_choice=="Random Forest":
                model = RandomForestClassifier(n_estimators=n_estimators,random_state=42)
            else:
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=500)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test,y_pred)
            rmse = mean_squared_error(y_test,y_pred,squared=False)
            st.session_state.model_metrics = pd.DataFrame({'Metric':['Accuracy','RMSE'],'Value':[acc,rmse]})
            st.session_state.model = model
            st.session_state.X_train, st.session_state.X_test = X_train,X_test
            st.session_state.y_train, st.session_state.y_test = y_train,y_test
            st.subheader("Model Metrics")
            st.dataframe(st.session_state.model_metrics)

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
