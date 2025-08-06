import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Data Exploration & EDA", layout="wide")
st.title("📊 Data Exploration & Summary (EDA)")

st.markdown("""
This page enables you to perform an in-depth **Exploratory Data Analysis (EDA)** on any CSV file.

- Upload your dataset.
- Get automatic statistics, missing value insights, visual summaries, and correlation heatmap.
- Use this for rapid business/data science insight or as a prelude to model building.
""")

# --- Upload Section ---
uploaded_file = st.file_uploader(
    "Upload your CSV file for EDA", 
    type=["csv"],
    help="Upload any CSV file (rows=observations, columns=variables). Hover for help on all controls below."
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns.")

    st.subheader("Preview of Data")
    st.dataframe(
        df.head(30), 
        use_container_width=True
    )

    # --- Summary Statistics ---
    st.subheader("Descriptive Statistics")
    st.write(
        "Shows key statistics (mean, median, std, skewness, kurtosis, min, max, quartiles) for each numeric column. "
        
    )
    st.caption("**Tip:** Use these stats to quickly understand your data’s center, spread, and possible outliers.")

    desc = df.describe().T
    desc['median'] = df.median(numeric_only=True)
    desc['skew'] = df.skew(numeric_only=True)
    desc['kurtosis'] = df.kurtosis(numeric_only=True)
    st.dataframe(
        desc.round(3), 
        use_container_width=True
    )

    # --- Missing Values ---
    st.subheader("Missing Values Summary")
    st.write(
        "Summarizes the total and percent of missing values for each column."
    )
    st.caption(
        "Columns with many missing values may need imputation, removal, or special handling before analysis."
    )
    mv = df.isnull().sum()
    mv_percent = (mv / len(df)) * 100
    missing = pd.DataFrame({
        'Missing Values': mv,
        'Percent (%)': mv_percent
    })
    st.dataframe(
        missing[missing['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False),
        use_container_width=True
    )

    # --- Data Type Counts ---
    st.subheader("Data Types")
    st.write(
        "Shows count of columns by Python data type (e.g., float, int, object)."
    )
    st.caption(
        "Numeric types (float/int) are used for statistical and ML analysis. Object/category types are usually text or categorical data."
    )
    st.write(df.dtypes.value_counts())
    st.write(df.dtypes)

    # --- Numeric Feature Selection ---
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.subheader("Visualise Numeric Features")

        col1, col2 = st.columns(2)
        with col1:
            selected_num_col = st.selectbox(
                "Select numeric column for boxplot and histogram",
                options=numeric_cols,
                help="Pick a numeric column to visualize its distribution. "
                     "Boxplot shows outliers and spread."
            )

            st.write("Boxplot")
            st.caption(
                "A boxplot displays the distribution, outliers, and quartiles of the selected variable. "
                "Use it to spot unusual values or asymmetry in your data."
            )
            fig, ax = plt.subplots()
            sns.boxplot(y=df[selected_num_col].dropna(), ax=ax)
            st.pyplot(fig)

        with col2:
            st.write("Histogram")
            st.caption(
                "A histogram shows the frequency of values for the selected variable. "
                "Helps you see the shape, center, and spread of your data."
            )
            fig2, ax2 = plt.subplots()
            sns.histplot(df[selected_num_col].dropna(), kde=True, ax=ax2, bins=30)
            st.pyplot(fig2)

        # --- Correlation Heatmap ---
        st.subheader("Correlation Heatmap")
        st.caption(
            "Shows how strongly each pair of numeric columns is related (ranges -1 to 1). "
            "Bright colors = strong relationships. Use to check for redundancy or feature engineering opportunities."
        )
        corr = df[numeric_cols].corr()
        fig3, ax3 = plt.subplots(figsize=(min(12, 1.2*len(numeric_cols)), min(8, 0.8*len(numeric_cols))))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax3)
        st.pyplot(fig3)

        if st.button(
            "Download Correlation Matrix as CSV", 
            help="Download the correlation matrix for this dataset (for reports or further analysis)."
        ):
            csv = corr.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="Download correlation_matrix.csv",
                data=csv,
                file_name="correlation_matrix.csv",
                mime='text/csv',
                help="Click to download the displayed correlation matrix as a CSV file."
            )

    # --- Categorical Feature Summary ---
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        st.subheader("Categorical Columns: Top Values and Counts")
        st.caption(
            "For each text or category column, shows the most frequent values and their counts. "
            "Use to spot dominant categories or potential encoding needs."
        )
        for col in cat_cols:
            st.markdown(f"**{col}** (top 10)")
            st.write(df[col].value_counts().head(10))

else:
    st.info("Upload a CSV file to begin full data exploration.")

# --- Footer/Info ---
st.markdown("---")
st.caption("© 2025 Data Exploration Module | Built for MSc Applied AI and Data Science at Solent University.")
