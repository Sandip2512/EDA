import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="EDA App", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", "Extra Tools"])

# Upload File
@st.cache_data
def load_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except:
        return pd.read_excel(uploaded_file)

# Home Page
if page == "Home":
    st.title("📊 Exploratory Data Analysis App")
    st.markdown("""
    This app allows you to upload a dataset and perform a full EDA process.
    
    **Why EDA?**
    - Understand data shape, types, and distributions  
    - Detect outliers, duplicates, and missing values  
    - Explore relationships between variables  
    - Prepare data for modeling and predictions

    Upload a file below to get started:
    """)
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        df = load_data(uploaded_file)
        st.subheader("🔍 Data Preview")
        st.dataframe(df)

        st.subheader("🧾 Basic Info")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.write("Column Data Types:")
        st.write(df.dtypes)

        st.subheader("⚠️ Missing Values")
        st.write(df.isnull().sum())

        st.subheader("🧪 Descriptive Statistics")
        st.write(df.describe())

        # Save uploaded data to session state
        st.session_state.df = df

# Univariate
elif page == "Univariate Analysis":
    if 'df' not in st.session_state:
        st.warning("Please upload data from the Home page.")
    else:
        df = st.session_state.df
        st.title("📈 Univariate Analysis")
        col = st.selectbox("Select a column", df.columns)

        if pd.api.types.is_numeric_dtype(df[col]):
            st.write(df[col].describe())
            fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
            st.plotly_chart(fig)
        else:
            st.write(df[col].value_counts())
            fig = px.bar(df[col].value_counts().reset_index(), x='index', y=col)
            st.plotly_chart(fig)

# Bivariate
elif page == "Bivariate Analysis":
    if 'df' not in st.session_state:
        st.warning("Please upload data from the Home page.")
    else:
        df = st.session_state.df
        st.title("📊 Bivariate Analysis")

        col1 = st.selectbox("X-axis", df.columns)
        col2 = st.selectbox("Y-axis", df.columns)

        if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
            fig = px.scatter(df, x=col1, y=col2, trendline="ols")
            st.plotly_chart(fig)
        else:
            fig = px.box(df, x=col1, y=col2)
            st.plotly_chart(fig)

# Multivariate
elif page == "Multivariate Analysis":
    if 'df' not in st.session_state:
        st.warning("Please upload data from the Home page.")
    else:
        df = st.session_state.df
        st.title("🧬 Multivariate Analysis")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(num_cols) >= 2:
            st.subheader("Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            st.subheader("Pair Plot")
            if st.checkbox("Show Pair Plot (slow for large datasets)"):
                fig2 = sns.pairplot(df[num_cols])
                st.pyplot(fig2)

# Extra Tools
elif page == "Extra Tools":
    if 'df' not in st.session_state:
        st.warning("Please upload data from the Home page.")
    else:
        df = st.session_state.df
        st.title("🛠 Extra Data Tools")

        st.subheader("Column Selector")
        selected_cols = st.multiselect("Select columns to view", df.columns.tolist(), default=df.columns.tolist())
        st.dataframe(df[selected_cols])

        st.subheader("Missing Value Handling")
        if st.checkbox("Drop missing values"):
            df.dropna(inplace=True)
            st.success("Missing values dropped.")

        st.subheader("Duplicate Handling")
        if st.checkbox("Drop duplicate rows"):
            df.drop_duplicates(inplace=True)
            st.success("Duplicates removed.")

        st.subheader("Convert column to datetime")
        col = st.selectbox("Select column", df.columns)
        if st.button("Convert to datetime"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
            st.success(f"{col} converted.")

        st.subheader("Download Cleaned Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", csv, "cleaned_data.csv", "text/csv")

        st.session_state.df = df
