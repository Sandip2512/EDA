import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="EDA Web App", layout="wide")

# Sidebar navigation
pages = ["Home", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"]
selected_page = st.sidebar.selectbox("Select a page", pages)

# Upload data
if "df" not in st.session_state:
    st.session_state.df = None

if selected_page == "Home":
    st.title("Exploratory Data Analysis (EDA) Web App")
    st.markdown("""
    Upload your CSV file to explore and visualize your data easily.
    
    ### What is EDA?
    - Helps understand data distributions, patterns, and anomalies.
    - Identifies relationships between variables.
    - Aids in feature selection and business decision-making.
    """)
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("Data uploaded successfully!")
        st.dataframe(df.head())

elif st.session_state.df is not None:
    df = st.session_state.df

    if selected_page == "Univariate Analysis":
        st.title("Univariate Analysis")
        column = st.selectbox("Select a column", df.columns)

        if column:
            st.write(f"### Value Counts for `{column}`")
            
            if df[column].notna().sum() > 0:
                df[column] = df[column].astype(str).fillna("Missing")
                value_counts = df[column].value_counts().reset_index()
                value_counts.columns = [column, 'Count']

                fig = px.bar(
                    value_counts,
                    x=column,
                    y='Count',
                    color='Count',
                    color_continuous_scale='Rainbow',
                    title=f'Distribution of {column}'
                )
                st.plotly_chart(fig, use_container_width=True)

                top_val = value_counts.iloc[0]
                st.info(f"**Insight:** The most common value in `{column}` is **{top_val[0]}** with **{top_val[1]}** occurrences.")
            else:
                st.warning("Selected column is empty or not suitable for visualization.")

    elif selected_page == "Bivariate Analysis":
        st.title("Bivariate Analysis")
        col1 = st.selectbox("Select X-axis column", df.columns)
        col2 = st.selectbox("Select Y-axis column", df.columns)

        if col1 and col2:
            try:
                fig = px.scatter(df, x=col1, y=col2, color=col1,
                                 title=f"Scatter plot of {col1} vs {col2}",
                                 color_continuous_scale="Agsunset")
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"**Insight:** This scatter plot helps identify relationships between `{col1}` and `{col2}` such as trends, clusters, or outliers.")
            except:
                st.warning("These columns might not be numeric. Please try other columns.")

    elif selected_page == "Multivariate Analysis":
        st.title("Multivariate Analysis")
        num_cols = df.select_dtypes(include='number').columns.tolist()
        if len(num_cols) >= 2:
            fig = px.imshow(df[num_cols].corr(),
                            color_continuous_scale="RdBu_r",
                            title="Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
            st.info("**Insight:** The correlation matrix shows how numeric variables relate to one another. High positive or negative values indicate strong relationships.")
        else:
            st.warning("Not enough numeric columns for multivariate analysis.")

else:
    st.warning("Please upload a dataset from the Home page to begin.")

