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
page = st.sidebar.radio("Go to", ["Home", "Chart Selection", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])

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

# Chart Selection Page
elif page == "Chart Selection":
    st.header("Custom Chart Selection")
    if df is not None:
        chart_type = st.selectbox("Select Chart Type", ["Histogram", "Bar Chart", "Box Plot", "Scatter Plot", "Line Chart", "Pie Chart", "Correlation Matrix", "Time Series Trend"])
        x_col = st.selectbox("Select X-axis", df.columns)
        y_col = st.selectbox("Select Y-axis (if applicable)", ["None"] + df.columns.tolist())

        fig = None
        if chart_type == "Histogram":
            fig = px.histogram(df, x=x_col, nbins=30, title=f"Histogram of {x_col}")
        elif chart_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=None if y_col == "None" else y_col, title=f"Bar Chart of {x_col} vs {y_col}")
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_col, y=None if y_col == "None" else y_col, title=f"Box Plot of {x_col} by {y_col}")
        elif chart_type == "Scatter Plot":
            if y_col != "None":
                fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot of {y_col} vs {x_col}",
                                 color=y_col, color_continuous_scale="Viridis", opacity=0.7)
            else:
                st.warning("Scatter plot requires both X and Y axes.")
        elif chart_type == "Line Chart":
            if y_col != "None":
                fig = px.line(df, x=x_col, y=y_col, title=f"Line Chart of {y_col} vs {x_col}")
            else:
                st.warning("Line chart requires both X and Y axes.")
        elif chart_type == "Pie Chart":
            fig = px.pie(df, names=x_col, title=f"Pie Chart of {x_col}")
        elif chart_type == "Correlation Matrix":
            num_df = df.select_dtypes(include='number')
            st.write("### Correlation Matrix")
            st.dataframe(num_df.corr().round(2))
            fig = px.imshow(num_df.corr(), text_auto=True, title="Correlation Heatmap")
        elif chart_type == "Time Series Trend":
            val_col = st.selectbox("Select Value Column", df.select_dtypes(include='number').columns.tolist())
            df_sorted = df.sort_values(x_col)
            fig = px.line(df_sorted, x=x_col, y=val_col, title=f"Trend of {val_col} over {x_col}")

        if fig:
            st.plotly_chart(fig, use_container_width=True)
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
