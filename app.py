import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="EDA App", layout="wide")

# Set sidebar navigation
page = st.sidebar.selectbox(
    "Select a Page",
    ["🏠 Home / Upload", "📊 Univariate", "🔗 Bivariate", "🌐 Multivariate"]
)

# Session state to store dataframe
if "df" not in st.session_state:
    st.session_state.df = None

# ---------------------------- PAGE 1: Upload + Motivation ----------------------------
if page == "🏠 Home / Upload":
    st.title("📊 Exploratory Data Analysis (EDA) Tool")

    st.markdown("""
    Welcome! This app helps you **upload a dataset** and explore it visually through:

    - 📍 Univariate (single-variable)
    - 🔗 Bivariate (two-variable)
    - 🌐 Multivariate (multiple-variable) analysis

    ---
    ### 💡 Why EDA?
    - Understand patterns and distributions  
    - Detect missing values and outliers  
    - Spot correlations and trends  
    - Guide data preprocessing and feature selection

    Upload your CSV file below to begin:
    """)

    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("File uploaded successfully!")
        st.subheader("👀 Preview of Uploaded Data")
        st.dataframe(df.head())
    else:
        st.info("Please upload a CSV file to get started.")

# ---------------------------- PAGE 2: Univariate ----------------------------
elif page == "📊 Univariate":
    st.title("📍 Univariate Analysis")

    if st.session_state.df is not None:
        df = st.session_state.df
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns

        col = st.selectbox("Select a numerical column", num_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Please upload data from the Home page first.")

# ---------------------------- PAGE 3: Bivariate ----------------------------
elif page == "🔗 Bivariate":
    st.title("🔗 Bivariate Analysis")

    if st.session_state.df is not None:
        df = st.session_state.df
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns

        col1 = st.selectbox("X-axis", num_cols, key="biv1")
        col2 = st.selectbox("Y-axis", num_cols, key="biv2")
        fig = px.scatter(df, x=col1, y=col2, title=f"{col1} vs {col2}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please upload data from the Home page first.")

# ---------------------------- PAGE 4: Multivariate ----------------------------
elif page == "🌐 Multivariate":
    st.title("🌐 Multivariate Analysis")

    if st.session_state.df is not None:
        df = st.session_state.df
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns

        fig = px.scatter_matrix(df, dimensions=num_cols)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please upload data from the Home page first.")
