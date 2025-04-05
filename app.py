import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="EDA Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for UI
st.markdown("""
    <style>
    .main {
        background-color: #f4f4f4;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .css-1d391kg, .css-1avcm0n {
        background-color: #1f77b4;
        color: white;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])

# Upload
@st.cache_data

def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Home Page
if page == "Home":
    st.title("Exploratory Data Analysis App")
    st.markdown("Gain deep insights from your data in just a few clicks. Upload your dataset and begin exploring!")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.subheader("Preview of Dataset")
        st.dataframe(df.head())

        st.subheader("Dataset Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing %", f"{df.isnull().mean().mean() * 100:.2f}%")

        with st.expander("Column Data Types"):
            st.write(df.dtypes)

        with st.expander("Descriptive Statistics"):
            st.write(df.describe())

        st.markdown("---")
        st.markdown("### Why EDA?")
        st.markdown("""
        - Understand distribution & outliers
        - Detect correlations & patterns
        - Validate assumptions for modeling
        - Identify missing or incorrect values
        """)

# Univariate
elif page == "Univariate Analysis":
    st.title("Univariate Analysis")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="uni")
    if uploaded_file:
        df = load_data(uploaded_file)
        column = st.selectbox("Select a column", df.columns)

        if pd.api.types.is_numeric_dtype(df[column]):
            st.plotly_chart(px.histogram(df, x=column, nbins=30, title=f"Distribution of {column}"))
            st.markdown(f"**Insight**: {column} has mean {df[column].mean():.2f} and std {df[column].std():.2f}. Check for skew or outliers above 95th percentile.")
        else:
            vc = df[column].value_counts().reset_index()
            fig = px.bar(vc, x='index', y=column, title=f"Count of categories in {column}")
            fig.update_layout(xaxis_title=column, yaxis_title="Count")
            st.plotly_chart(fig)
            st.markdown(f"**Insight**: Most frequent value in {column} is '{vc.iloc[0]['index']}' with {vc.iloc[0][column]} entries.")

# Bivariate
elif page == "Bivariate Analysis":
    st.title("Bivariate Analysis")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="bi")
    if uploaded_file:
        df = load_data(uploaded_file)
        x = st.selectbox("X-axis", df.columns)
        y = st.selectbox("Y-axis", df.columns)

        if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
            fig = px.scatter(df, x=x, y=y, trendline="ols", title=f"Scatter Plot between {x} and {y}")
            st.plotly_chart(fig)
            corr = df[[x, y]].corr().iloc[0, 1]
            st.markdown(f"**Insight**: Correlation between {x} and {y} is {corr:.2f}.")
        else:
            fig = px.box(df, x=x, y=y, title=f"Box Plot: {y} by {x}")
            st.plotly_chart(fig)
            st.markdown("**Insight**: Check how {y} varies across different categories of {x}.")

# Multivariate
elif page == "Multivariate Analysis":
    st.title("Multivariate Analysis")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="multi")
    if uploaded_file:
        df = load_data(uploaded_file)
        st.markdown("#### Correlation Heatmap")
        num_cols = df.select_dtypes(include='number')
        if not num_cols.empty:
            corr = num_cols.corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, cmap='coolwarm', annot=True, ax=ax)
            st.pyplot(fig)
            st.markdown("**Insight**: Identify multicollinearity. Look for high correlations (>0.8) between numeric variables.")
        else:
            st.warning("No numeric columns found.")
            
