import streamlit as st
import pandas as pd
import plotly.express as px

# --- App Config ---
st.set_page_config(page_title="EDA Visualizer", layout="wide", initial_sidebar_state="expanded")
st.title("Exploratory Data Analysis (EDA) Dashboard")

# --- Sidebar Navigation ---
menu = st.sidebar.radio("Go to", ["Home", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])

# --- File Upload ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if 'df' not in st.session_state:
    st.session_state.df = None

# --- HOME PAGE ---
if menu == "Home":
    st.header("Upload your dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.session_state.df = df
        st.success("Data loaded successfully!")

    if st.session_state.df is not None:
        df = st.session_state.df

        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Quick Stats")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing %", round((df.isnull().sum().sum() / df.size) * 100, 2))

        st.subheader("Why EDA?")
        st.markdown("""
        - EDA helps uncover hidden patterns, trends, and outliers.
        - Enables data-driven decisions by visualizing relationships.
        - Helps identify errors, missing data, and outliers.
        - Guides feature selection and business strategies before modeling.
        """)

# --- UNIVARIATE ---
elif menu == "Univariate Analysis":
    if st.session_state.df is not None:
        df = st.session_state.df
        st.header("Univariate Analysis")

        column = st.selectbox("Choose a column", df.columns)

        if pd.api.types.is_numeric_dtype(df[column]):
            fig = px.histogram(df, x=column, nbins=30, title=f"Distribution of {column}", color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Mean: {df[column].mean():.2f}, Std: {df[column].std():.2f}, Skewness: {df[column].skew():.2f}")

        else:
            fig = px.bar(df[column].value_counts().reset_index().rename(columns={'index': column, column: 'Count'}),
                         x=column, y="Count", color=column, title=f"Frequency of {column}")
            st.plotly_chart(fig, use_container_width=True)
            top_cat = df[column].value_counts().idxmax()
            st.info(f"Most frequent category in {column} is '{top_cat}'")

# --- BIVARIATE ---
elif menu == "Bivariate Analysis":
    if st.session_state.df is not None:
        df = st.session_state.df
        st.header("Bivariate Analysis")

        x_col = st.selectbox("Select X variable", df.columns, key="biv_x")
        y_col = st.selectbox("Select Y variable", df.columns, key="biv_y")

        if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
            fig = px.scatter(df, x=x_col, y=y_col, color_continuous_scale='Viridis', trendline="ols",
                             title=f"{y_col} vs {x_col}")
            st.plotly_chart(fig, use_container_width=True)
            corr = df[x_col].corr(df[y_col])
            st.info(f"Correlation between {x_col} and {y_col} is {corr:.2f}")

        elif pd.api.types.is_categorical_dtype(df[x_col]) or df[x_col].dtype == object:
            fig = px.box(df, x=x_col, y=y_col, color=x_col, title=f"{y_col} distribution across {x_col}")
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"This shows how {y_col} varies across different categories of {x_col}.")

# --- MULTIVARIATE ---
elif menu == "Multivariate Analysis":
    if st.session_state.df is not None:
        df = st.session_state.df
        st.header("Multivariate Analysis")

        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) >= 3:
            cols = st.multiselect("Select up to 3 numeric columns", numeric_cols, default=numeric_cols[:3])

            if len(cols) == 3:
                fig = px.scatter_3d(df, x=cols[0], y=cols[1], z=cols[2], color=cols[0],
                                    title=f"3D Scatter: {cols[0]} vs {cols[1]} vs {cols[2]}")
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"This 3D chart helps visualize interaction among {cols[0]}, {cols[1]}, and {cols[2]}.")

        st.subheader("Correlation Heatmap")
        fig = px.imshow(df[numeric_cols].corr(), color_continuous_scale='RdBu', title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Strong correlations (above 0.8 or below -0.8) indicate strong relationships, useful for business decisions.")
        
