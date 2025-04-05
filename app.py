import streamlit as st

st.set_page_config(layout="wide")
st.title("📊 EDA Dashboard for Any Dataset")

st.markdown("📁 Use the sidebar to navigate through pages: **Overview**, **Univariate**, **Bivariate**, **Multivariate**, and **All Charts**.")

st.info("⬅️ Upload your dataset on any page to begin exploring!")

### ✅ `pages/1_Overview.py`

import streamlit as st
import pandas as pd

st.title("📄 Dataset Overview")

uploaded_file = st.file_uploader("Upload CSV File", type="csv", key="overview")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Data")
    st.dataframe(df.head())

    st.subheader("📐 Shape")
    st.write(df.shape)

    st.subheader("🧾 Missing Values")
    st.write(df.isnull().sum())

    st.subheader("📊 Data Types")
    st.write(df.dtypes)

    st.session_state["df"] = df

### ✅ `pages/2_Univariate_Analysis.py`

import streamlit as st
import plotly.express as px

st.title("📊 Univariate Analysis")

df = st.session_state.get("df", None)
if df is not None:
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    if numeric_cols:
        col = st.selectbox("Select numeric column", numeric_cols)
        fig = px.histogram(df, x=col, title=f"Histogram of {col}")
        st.plotly_chart(fig)

    if cat_cols:
        col = st.selectbox("Select categorical column", cat_cols)
        st.bar_chart(df[col].value_counts())
else:
    st.warning("Upload a dataset from the Overview page.")

### ✅ `pages/3_Bivariate_Analysis.py`

import streamlit as st
import plotly.express as px

st.title("📈 Bivariate Analysis")

df = st.session_state.get("df", None)
if df is not None:
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) >= 2:
        x = st.selectbox("X-axis", numeric_cols)
        y = st.selectbox("Y-axis", numeric_cols, index=1)
        fig = px.scatter(df, x=x, y=y, title=f"{x} vs {y}")
        st.plotly_chart(fig)

        fig2 = px.box(df, x=x, y=y, title=f"Boxplot: {y} by {x}")
        st.plotly_chart(fig2)
    else:
        st.warning("Need at least two numeric columns.")
else:
    st.warning("Upload a dataset from the Overview page.")

### ✅ `pages/4_Multivariate_Analysis.py`

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📉 Multivariate Analysis")

df = st.session_state.get("df", None)
if df is not None:
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Need at least two numeric columns.")
else:
    st.warning("Upload a dataset from the Overview page.")


### ✅ `pages/5_All_Charts.py`

import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 All Charts at a Glance")

df = st.session_state.get("df", None)
if df is not None:
    st.subheader("📌 Histogram")
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if num_cols:
        fig = px.histogram(df, x=num_cols[0])
        st.plotly_chart(fig)

    st.subheader("📌 Correlation Heatmap")
    if len(num_cols) >= 2:
        fig2, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig2)
else:
    st.warning("Upload a dataset from the Overview page.")
