import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Experimental Models Lab", layout="wide")
st.title("🔬 Experimental Models & Concepts Lab")

st.markdown("""
Explore **additional advanced models** taught in the MSc but not yet included in the main LungGuard pages:

- 🤖 SVM, Naïve Bayes, K-NN  
- 🔁 Reinforcement Learning (conceptual placeholder)  
- 🧩 DBSCAN, Agglomerative Clustering  
- 🔄 AutoML (Hyperparameter Grid Search)  
- 🔢 Multi-Output Regression  
""")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns.")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Select a Task")
    task = st.selectbox("Choose your experiment:", [
        "Support Vector Machine (SVM)",
        "Naïve Bayes Classification",
        "K-Nearest Neighbours (KNN)",
        "DBSCAN Clustering",
        "Agglomerative Clustering",
        "AutoML with Grid Search (SVM)",
        "Reinforcement Learning (Concept)"
    ])

    if task in ["Support Vector Machine (SVM)", "Naïve Bayes Classification", "K-Nearest Neighbours (KNN)", "AutoML with Grid Search (SVM)"]:
        cat_targets = [c for c in df.columns if df[c].nunique() <= 10 and df[c].dtype != 'float']
        target = st.selectbox("Target variable", cat_targets)
        features = st.multiselect("Feature columns", [c for c in df.columns if c != target])

        if target and features:
            X = df[features].copy()
            y = df[target].astype(str)
            for col in X.select_dtypes('object').columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            y_enc = LabelEncoder().fit_transform(y)
            X = X.fillna(X.mean())

            X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, random_state=42)

            if task == "Support Vector Machine (SVM)":
                model = SVC(kernel='rbf', probability=True)
            elif task == "Naïve Bayes Classification":
                model = GaussianNB()
            elif task == "K-Nearest Neighbours (KNN)":
                k = st.slider("Select number of neighbours (k)", 1, 20, 5)
                model = KNeighborsClassifier(n_neighbors=k)
            elif task == "AutoML with Grid Search (SVM)":
                st.info("Tuning hyperparameters of SVM (Grid Search on C & gamma)")
                param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
                grid = GridSearchCV(SVC(probability=True), param_grid, cv=3)
                model = grid

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Accuracy: {acc:.3f}")
            st.text(classification_report(y_test, y_pred))



    elif task == "DBSCAN Clustering":
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        features = st.multiselect("Numeric columns for clustering", num_cols)
        if features:
            eps = st.slider("Epsilon (neighbourhood radius)", 0.1, 5.0, 0.5)
            min_samples = st.slider("Minimum samples per cluster", 1, 10, 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            X_scaled = StandardScaler().fit_transform(df[features].fillna(0))
            labels = model.fit_predict(X_scaled)
            st.success(f"Detected clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
            df['DBSCAN_Cluster'] = labels
            st.dataframe(df[[*features, 'DBSCAN_Cluster']].head(20))

    elif task == "Agglomerative Clustering":
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        features = st.multiselect("Numeric columns for clustering", num_cols)
        if features:
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            model = AgglomerativeClustering(n_clusters=n_clusters)
            X_scaled = StandardScaler().fit_transform(df[features].fillna(0))
            labels = model.fit_predict(X_scaled)
            df['AggCluster'] = labels
            st.success(f"Assigned {n_clusters} clusters.")
            st.dataframe(df[[*features, 'AggCluster']].head(20))

    elif task == "Reinforcement Learning (Concept)":
        st.warning("This is a placeholder: RL is conceptually covered but not deployed here.")
        st.markdown("""
        **Why RL is not implemented:**
        - Requires simulation environments (e.g., OpenAI Gym)
        - Not practical with tabular patient data
        - Better suited for treatment planning or dynamic scheduling

        This feature is for future work.
        """)

else:
    st.info("Upload a dataset to get started.")
