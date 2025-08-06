import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Visualisation Gallery", layout="wide")
st.title("🖼️ Visualisation Gallery")

st.markdown("""
Upload your CSV dataset and explore a gallery of **robust, high-quality interactive data visualisations**.
All visuals have intelligent dropdowns that adapt to your dataset structure, avoiding common issues with IDs, years, and empty columns.
""")

uploaded_file = st.file_uploader(
    "Upload your CSV file", 
    type=["csv"],
    help="Upload a CSV to enable gallery features. Rows = observations, columns = variables. Hover on each control for more info."
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # --- Auto-detect and treat ID columns as categorical ---
    for col in df.columns:
        if 'id' in col.lower() or 'study' in col.lower():
            df[col] = df[col].astype(str)

    # Remove columns with all-null or single unique value
    valid_cols = [col for col in df.columns if df[col].nunique(dropna=True) > 1 and not df[col].isnull().all()]

    st.success(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns.")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Select a Visualisation")
    vis_type = st.selectbox(
        "Choose a visualisation type",
        [
            "Heatmap (Correlation Matrix)",
            "Treemap (Category Distribution)",
            "Radar Chart (Multivariate Profile)",
            "Pie Chart",
            "Box Plot",
        ],
        help="Select which type of visualisation to create. Hover over each type for a quick description."
    )

    # --- Utility: Column Choices ---
    def get_categorical_cols(df):
        cats = []
        for col in df.columns:
            if (df[col].dtype == 'object' or df[col].dtype.name == 'category' 
                or 'id' in col.lower() or 'study' in col.lower()):
                if 2 <= df[col].nunique() < min(25, len(df)//2):
                    cats.append(col)
        return cats

    def get_numeric_cols(df):
        return [col for col in df.select_dtypes(include=np.number).columns
                if df[col].nunique() > 1 and not df[col].isnull().all()]

    # --- Visualisation: Heatmap ---
    if vis_type == "Heatmap (Correlation Matrix)":
        st.info("Heatmaps show the correlation (relationship) between numeric variables. Use for pattern discovery, feature selection, or multicollinearity checks.")
        num_cols = get_numeric_cols(df)
        if len(num_cols) < 2:
            st.warning("At least two valid numeric columns are needed for a heatmap.")
        else:
            st.caption("Shows correlations between all numeric columns (from -1 to 1). Brighter colors mean stronger relationships.")
            corr = df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(min(12, 1.2*len(num_cols)), min(8, 0.8*len(num_cols))))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

    # --- Visualisation: Treemap ---
    elif vis_type == "Treemap (Category Distribution)":
        st.info("Treemaps visualise hierarchical or part-to-whole relationships. Ideal for proportions of categorical variables.")
        cat_cols = get_categorical_cols(df)
        if not cat_cols:
            st.warning("No suitable categorical columns detected for treemap (must have <25 unique values, not pure ID columns).")
        else:
            col = st.selectbox(
                "Select categorical column for treemap", 
                options=cat_cols,
                help="Select a categorical/text column to visualise as a treemap (shows relative sizes)."
            )
            st.caption("Treemap helps you see the proportion of each category or group within your data.")
            treemap_data = df[col].value_counts().reset_index()
            treemap_data.columns = [col, 'Count']
            fig = px.treemap(treemap_data, path=[col], values='Count', color='Count')
            st.plotly_chart(fig, use_container_width=True)

    # --- Visualisation: Radar Chart ---
    elif vis_type == "Radar Chart (Multivariate Profile)":
        st.info("Radar charts (spider charts) compare multiple quantitative variables for one or several observations.")
        num_cols = get_numeric_cols(df)
        if len(num_cols) < 3:
            st.warning("At least three valid numeric columns required for radar chart.")
        else:
            st.write("Choose a row (observation) to profile on radar chart:")
            idx = st.slider(
                "Row index", 
                min_value=0, max_value=len(df)-1, value=0,
                help="Pick which row of your dataset to visualise. Each axis shows a different variable for that row."
            )
            st.caption("Radar chart visualises all selected numeric variables for one data row as axes on a circle.")
            data_row = df.iloc[idx][num_cols]
            categories = list(num_cols)
            values = data_row.values.tolist()
            values += values[:1]  # close the circle

            fig, ax = plt.subplots(subplot_kw=dict(polar=True))
            angles = np.linspace(0, 2*np.pi, len(categories)+1, endpoint=True)
            ax.plot(angles, values, linewidth=2, linestyle='solid')
            ax.fill(angles, values, alpha=0.25)
            ax.set_thetagrids(angles[:-1]*180/np.pi, categories)
            plt.title(f"Radar Chart for Row {idx}")
            st.pyplot(fig)

    # --- Visualisation: Pie Chart ---
    elif vis_type == "Pie Chart":
        st.info("Pie charts show part-to-whole proportions. Use for categorical variables with few unique values.")
        cat_cols = [col for col in get_categorical_cols(df) if df[col].nunique() <= 12]
        if not cat_cols:
            st.warning("No suitable categorical columns for pie chart (≤12 unique values).")
        else:
            col = st.selectbox(
                "Select categorical column", 
                options=cat_cols,
                help="Select a categorical/text column to visualise as a pie chart (shows proportions)."
            )
            st.caption("Pie chart helps you visualise the percentage or fraction of each category in your data.")
            pie_data = df[col].value_counts().reset_index()
            pie_data.columns = [col, 'Count']
            fig = px.pie(pie_data, names=col, values='Count', title=f"Pie Chart for {col}")
            st.plotly_chart(fig, use_container_width=True)

    # --- Visualisation: Box Plot ---
    elif vis_type == "Box Plot":
        st.info("Box plots visualise value distributions, outliers, and quartiles. Use for numeric columns.")
        num_cols = get_numeric_cols(df)
        cat_cols = get_categorical_cols(df)
        if not num_cols:
            st.warning("No valid numeric columns detected for box plot.")
        else:
            col = st.selectbox(
                "Select numeric column", 
                num_cols,
                help="Choose a numeric variable to plot as a boxplot (shows spread and outliers)."
            )
            group_col = st.selectbox(
                "Group by (optional categorical column)", 
                ["None"] + cat_cols,
                help="(Optional) Select a categorical column to create grouped boxplots for each category."
            )
            st.caption("Boxplot shows data distribution (center, spread, outliers) for your variable, grouped if desired.")
            if group_col == "None":
                fig = px.box(df, y=col, points="all", title=f"Boxplot for {col}")
            else:
                fig = px.box(df, x=group_col, y=col, points="all", title=f"Boxplot of {col} by {group_col}")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.caption("Try different visualisations, columns, and rows to gain insights from your data. All visuals are for exploration and presentation.")

else:
    st.info("Upload a CSV file to start exploring visualisation options.")

st.markdown("---")
st.caption("© 2025 Visualisation Gallery | MSc Applied AI and Data Science, Solent University.")
