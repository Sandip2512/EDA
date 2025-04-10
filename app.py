import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_lottie import st_lottie
import requests
import numpy as np

# Monkey patch for np.bool for compatibility with Plotly
if not hasattr(np, 'bool'):
    np.bool = bool

# Set page config
st.set_page_config(page_title="EDA Web App", layout="wide", page_icon="📊")

# Load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_eda = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_yr6zz3wv.json")

# Cached data loading
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Sidebar navigation
st.sidebar.title("EDA Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Cleaning", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Reset button
if st.sidebar.button("Reset File"):
    if "df" in st.session_state:
        del st.session_state.df
    uploaded_file = None

# Load file if uploaded
if uploaded_file is not None:
    st.session_state.df = load_data(uploaded_file)

df = st.session_state.get("df", None)

# Home page
if page == "Home":
    st.title("Exploratory Data Analysis App")
    st.write("Upload your dataset and explore data interactively with visual insights.")
    st_lottie(lottie_eda, height=300, key="eda")
    st.markdown("### Why EDA?")
    st.write(
        "- Understand your data distribution\n"
        "- Spot missing values or outliers\n"
        "- Find hidden relationships\n"
        "- Gain business insights before modeling"
    )
    if df is not None:
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

# Data Cleaning Page
elif page == "Data Cleaning":
    st.header("Data Cleaning")
    if df is not None:
        st.subheader("Missing Values")
        missing_counts = df.isnull().sum()
        st.dataframe(missing_counts[missing_counts > 0])

        st.subheader("Handle Missing Values")
        action = st.selectbox("Choose an action", ["None", "Drop missing values", "Fill missing values"])

        if action == "Drop missing values":
            df.dropna(inplace=True)
            st.success("Missing values dropped.")
        elif action == "Fill missing values":
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype == 'object':
                        mode_val = df[col].mode()[0]
                        df[col].fillna(mode_val, inplace=True)
                    else:
                        method = st.radio(f"How to fill missing values for {col}?", ["Mean", "Median"], key=col)
                        if method == "Mean":
                            df[col].fillna(df[col].mean(), inplace=True)
                        else:
                            df[col].fillna(df[col].median(), inplace=True)
            st.success("Missing values filled.")
    else:
        st.warning("Please upload a dataset.")

# Univariate
elif page == "Univariate Analysis":
    st.header("Univariate Analysis")
    if df is not None:
        column = st.selectbox("Select a column", df.columns)
        if df[column].dtype == "object":
            counts = df[column].value_counts().reset_index()
            counts.columns = [column, "Count"]
            fig = px.bar(counts, x=column, y="Count", color=column, title=f"Distribution of {column}")
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Most common value: {counts[column].iloc[0]} with {counts['Count'].iloc[0]} occurrences.")
        else:
            fig = px.histogram(df, x=column, nbins=30, title=f"Distribution of {column}", color_discrete_sequence=["#3E64FF"])
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Mean: {df[column].mean():.2f}, Std Dev: {df[column].std():.2f}")
    else:
        st.warning("Please upload a dataset.")

# Bivariate
elif page == "Bivariate Analysis":
    st.header("Bivariate Analysis")
    if df is not None:
        x_col = st.selectbox("X-axis", df.columns)
        y_col = st.selectbox("Y-axis", df.columns, index=1)
        if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
            fig = px.scatter(df, x=x_col, y=y_col, trendline="ols", title=f"{y_col} vs {x_col}")
            st.plotly_chart(fig, use_container_width=True)
            corr = df[[x_col, y_col]].corr().iloc[0,1]
            st.success(f"Correlation between {x_col} and {y_col} is {corr:.2f}")
        else:
            fig = px.box(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please upload a dataset.")

# Multivariate
elif page == "Multivariate Analysis":
    st.header("Multivariate Analysis")
    if df is not None:
        num_cols = df.select_dtypes(include='number').columns.tolist()
        if len(num_cols) >= 3:
            x = st.selectbox("X-axis", num_cols)
            y = st.selectbox("Y-axis", num_cols, index=1)
            color = st.selectbox("Color", num_cols, index=2)
            fig = px.scatter(
                df, x=x, y=y, color=color,
                title=f"{y} vs {x} colored by {color}",
                color_continuous_scale="Viridis",
                opacity=0.7,
                size_max=10
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 3 numerical columns for multivariate analysis.")
    else:
        st.warning("Please upload a dataset.")
