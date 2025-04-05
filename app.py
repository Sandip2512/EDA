import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(page_title="Interactive EDA App", layout="wide")

# Sidebar Navigation
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Univariate", "Bivariate", "Multivariate", "Tools"])

# File Loader
@st.cache_data
def load_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except:
        return pd.read_excel(uploaded_file)

# HOME PAGE
if page == "Home":
    st.title("📊 Interactive EDA Web App")
    st.markdown("""
    Welcome to the **EDA App** powered by Streamlit and Plotly!

    **What can you explore?**
    - Dataset preview and structure  
    - Univariate, Bivariate, Multivariate insights  
    - Missing values, duplicates, and types  
    - Interactive and colorful charts

    **Get Started: Upload your dataset below**
    """)
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state.df = df

        st.subheader("🔍 Data Preview")
        st.dataframe(df, use_container_width=True)

        st.subheader("📐 Dataset Info")
        st.markdown(f"**Rows**: {df.shape[0]} | **Columns**: {df.shape[1]}")
        st.write("**Column Types**:")
        st.write(df.dtypes)

        st.subheader("⚠️ Missing Values")
        missing = df.isnull().sum()
        st.dataframe(missing[missing > 0], use_container_width=True)

        st.subheader("📊 Summary Statistics")
        st.write(df.describe())

# UNIVARIATE ANALYSIS
elif page == "Univariate":
    st.title("📈 Univariate Analysis")
    if 'df' not in st.session_state:
        st.warning("Please upload data from Home page.")
    else:
        df = st.session_state.df
        column = st.selectbox("Select a column", df.columns)

        if pd.api.types.is_numeric_dtype(df[column]):
            fig = px.histogram(df, x=column, nbins=40, color_discrete_sequence=["#636EFA"])
            fig.update_layout(title=f"Distribution of {column}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            freq = df[column].value_counts().reset_index()
            freq.columns = [column, "Count"]
            fig = px.bar(freq, x=column, y="Count", color=column, color_discrete_sequence=px.colors.qualitative.Safe)
            fig.update_layout(title=f"Category Count of {column}")
            st.plotly_chart(fig, use_container_width=True)

# BIVARIATE ANALYSIS
elif page == "Bivariate":
    st.title("📊 Bivariate Analysis")
    if 'df' not in st.session_state:
        st.warning("Please upload data from Home page.")
    else:
        df = st.session_state.df
        x = st.selectbox("X-axis", df.columns)
        y = st.selectbox("Y-axis", df.columns)

        if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
            fig = px.scatter(df, x=x, y=y, color=y, color_continuous_scale="Viridis")
            fig.update_layout(title=f"{x} vs {y} Scatter Plot")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.box(df, x=x, y=y, color=x, color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(title=f"{x} vs {y} Box Plot")
            st.plotly_chart(fig, use_container_width=True)

# MULTIVARIATE ANALYSIS
elif page == "Multivariate":
    st.title("🧬 Multivariate Analysis")
    if 'df' not in st.session_state:
        st.warning("Please upload data from Home page.")
    else:
        df = st.session_state.df
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(num_cols) >= 2:
            st.subheader("Heatmap of Correlation Matrix")
            corr = df[num_cols].corr().round(2)
            fig = ff.create_annotated_heatmap(
                z=corr.values,
                x=num_cols,
                y=num_cols,
                annotation_text=corr.values.astype(str),
                colorscale="YlGnBu",
                showscale=True
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Colored Parallel Coordinates Plot")
            color_col = st.selectbox("Select column for coloring", num_cols)
            fig = px.parallel_coordinates(df[num_cols], color=df[color_col], color_continuous_scale="Turbo")
            st.plotly_chart(fig, use_container_width=True)

# EXTRA TOOLS
elif page == "Tools":
    st.title("🛠 Data Tools")
    if 'df' not in st.session_state:
        st.warning("Please upload data from Home page.")
    else:
        df = st.session_state.df

        st.subheader("Filter Columns")
        selected_cols = st.multiselect("Select columns to show", df.columns, default=df.columns)
        st.dataframe(df[selected_cols], use_container_width=True)

        st.subheader("Handle Missing Data")
        if st.checkbox("Drop rows with missing values"):
            df.dropna(inplace=True)
            st.success("Missing values dropped.")

        st.subheader("Drop Duplicate Rows")
        if st.checkbox("Drop duplicates"):
            df.drop_duplicates(inplace=True)
            st.success("Duplicates removed.")

        st.subheader("Convert Column to Datetime")
        dt_col = st.selectbox("Select column to convert", df.columns)
        if st.button("Convert to datetime"):
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
            st.success(f"{dt_col} converted to datetime.")

        st.subheader("⬇️ Download Cleaned Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "cleaned_data.csv", "text/csv")

        st.session_state.df = df
