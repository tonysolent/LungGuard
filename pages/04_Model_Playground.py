import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Model Playground", layout="wide")
st.title("🧪 Model Playground (Regression, Classification, Clustering)")

st.markdown("""
Upload your CSV, choose your task, experiment with models, and visualise results.
- **Regression:** Linear Regression, Decision Tree Regressor
- **Classification:** Logistic Regression, Decision Tree Classifier
- **Clustering:** K-Means
""")

uploaded_file = st.file_uploader(
    "Upload your CSV file for modeling", 
    type=["csv"],
    help="Upload a CSV file to build and test models. Each row should be a data sample; columns are variables/features."
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns.")
    st.dataframe(df.head(10), use_container_width=True)

    # --- Column selection helpers ---
    all_cols = list(df.columns)
    num_cols = [col for col in all_cols if np.issubdtype(df[col].dtype, np.number) and df[col].nunique() > 1]
    cat_cols = [col for col in all_cols if df[col].dtype == 'object' and df[col].nunique() > 1 and df[col].nunique() < 30]

    st.subheader("Select Task")
    task = st.selectbox(
        "Model Task", 
        ["Regression", "Classification", "Clustering"],
        help="Choose your machine learning task: Regression (predict numbers), Classification (predict categories), or Clustering (find groups)."
    )

    # --- Regression Task ---
    if task == "Regression":
        st.info("**Regression:** Predict a continuous target variable (e.g., price, score).")
        if not num_cols:
            st.warning("No suitable numeric columns for regression.")
        else:
            target = st.selectbox(
                "Select target variable (numeric)", 
                num_cols,
                help="Choose the numeric column you want to predict (target variable)."
            )
            features = st.multiselect(
                "Select feature columns", 
                [c for c in all_cols if c != target], 
                default=[c for c in num_cols if c != target][:3],
                help="Select one or more columns as features (inputs) to use for prediction."
            )
            model_type = st.selectbox(
                "Model", 
                ["Linear Regression", "Decision Tree Regressor"],
                help="Choose which regression algorithm to use. Linear Regression for simple trends; Decision Tree for non-linear patterns."
            )
            test_size = st.slider(
                "Test set size (%)", 
                10, 50, 20,
                help="Percentage of data to hold out for testing (to measure performance)."
            ) / 100.0

            # Prepare data
            X = df[features].copy()
            y = df[target]
            # Encode categoricals
            for col in X.select_dtypes('object').columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Fit model
            if model_type == "Linear Regression":
                model = LinearRegression()
            else:
                max_depth = st.slider(
                    "Decision Tree max depth", 1, 10, 3,
                    help="Limits how complex the decision tree can be. Higher = more complex."
                )
                model = DecisionTreeRegressor(max_depth=max_depth)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.markdown("#### Performance Metrics")
            st.caption("Higher R² means better fit. Lower MSE means less error in predictions.")
            st.write(f"**R² score:** {r2_score(y_test, y_pred):.3f}")
            st.write(f"**Mean Squared Error:** {mean_squared_error(y_test, y_pred):.3f}")

            st.markdown("#### Predicted vs. Actual Plot")
            st.caption("See how well your model predicts. Perfect predictions would fall on a 45° line.")
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=y_pred, ax=ax)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs. Predicted")
            st.pyplot(fig)

    # --- Classification Task ---
    elif task == "Classification":
        st.info("**Classification:** Predict a categorical target (e.g., label, status, class).")
        cat_target_cols = [col for col in all_cols if df[col].dtype == 'object' or (df[col].nunique() < 20 and np.issubdtype(df[col].dtype, np.integer))]
        if not cat_target_cols:
            st.warning("No suitable categorical columns for classification.")
        else:
            target = st.selectbox(
                "Select target variable (categorical)", 
                cat_target_cols,
                help="Choose the categorical column to predict (target variable)."
            )
            features = st.multiselect(
                "Select feature columns", 
                [c for c in all_cols if c != target], 
                default=[c for c in num_cols if c != target][:3],
                help="Select columns to use as features (inputs) for classification."
            )
            model_type = st.selectbox(
                "Model", 
                ["Logistic Regression", "Decision Tree Classifier"],
                help="Choose classification algorithm. Logistic Regression for linear boundaries; Decision Tree for complex patterns."
            )
            test_size = st.slider(
                "Test set size (%)", 
                10, 50, 20,
                help="Percentage of data to hold out for testing model performance."
            ) / 100.0

            # Prepare data
            X = df[features].copy()
            y = df[target].astype(str)
            # Encode categoricals
            for col in X.select_dtypes('object').columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            X = X.fillna(X.mean())
            y_enc = LabelEncoder().fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=42)

            # Fit model
            if model_type == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            else:
                max_depth = st.slider(
                    "Decision Tree max depth", 1, 10, 3,
                    help="Controls how deep the tree can go. Shallow trees are simpler, deep trees fit more detail."
                )
                model = DecisionTreeClassifier(max_depth=max_depth)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.markdown("#### Performance Metrics")
            st.caption("Precision, recall, and F1-score measure model accuracy per class. Review these for balanced predictions.")
            st.write(classification_report(y_test, y_pred, output_dict=False))

            st.markdown("#### Confusion Matrix")
            st.caption("Rows = actual class; columns = predicted class. Diagonal = correct predictions.")
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)

            # ROC Curve (for binary)
            if len(np.unique(y_test)) == 2:
                st.markdown("#### ROC Curve")
                st.caption("Shows the trade-off between sensitivity and specificity for binary classification. Closer to top-left is better.")
                if model_type == "Logistic Regression":
                    y_score = model.predict_proba(X_test)[:, 1]
                else:
                    if hasattr(model, "predict_proba"):
                        y_score = model.predict_proba(X_test)[:, 1]
                    else:
                        y_score = model.predict(X_test)
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color='blue', label=f"ROC curve (AUC = {roc_auc:.2f})")
                ax.plot([0,1], [0,1], 'k--')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend()
                st.pyplot(fig)

    # --- Clustering Task ---
    elif task == "Clustering":
        st.info("**Clustering:** Unsupervised grouping of samples by similarity (K-Means).")
        if len(num_cols) < 2:
            st.warning("Need at least two valid numeric columns for clustering.")
        else:
            features = st.multiselect(
                "Select feature columns (numeric)", 
                num_cols, 
                default=num_cols[:3],
                help="Pick numeric columns as clustering features (inputs). Use more features for richer groupings."
            )
            n_clusters = st.slider(
                "Number of clusters (K)", 
                2, min(10, len(df)), 3,
                help="Choose how many clusters (groups) to find. Try different K values to see how groupings change."
            )
            max_iter = st.slider(
                "Max iterations", 50, 500, 300, step=50,
                help="Sets the maximum training steps for K-Means. Higher = more time for convergence."
            )
            scaler = StandardScaler()
            X = df[features].copy()
            X_scaled = scaler.fit_transform(X.fillna(X.mean()))

            kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            df_clusters = df.copy()
            df_clusters['Cluster'] = cluster_labels

            st.markdown("#### Cluster Assignment (first 20 rows)")
            st.caption("Shows cluster labels for each sample. Use to check grouping consistency.")
            st.dataframe(df_clusters[['Cluster'] + features].head(20))

            st.markdown("#### Cluster Centroids")
            st.caption("These are the center points of each cluster in feature space.")
            st.write(pd.DataFrame(kmeans.cluster_centers_, columns=features))

            # 2D plot of first two features
            if len(features) >= 2:
                st.markdown("#### Cluster Scatter Plot")
                st.caption("Shows clusters based on the first two features selected. Color = cluster label.")
                fig, ax = plt.subplots()
                sns.scatterplot(
                    x=X[features[0]], y=X[features[1]], hue=cluster_labels, palette="Set2", ax=ax)
                ax.set_xlabel(features[0])
                ax.set_ylabel(features[1])
                ax.set_title(f"K-Means Clusters (first 2 features)")
                st.pyplot(fig)

else:
    st.info("Upload a CSV to access the Model Playground.")

st.markdown("---")
st.caption("© 2025 Model Playground | MSc Applied AI and Data Science, Solent University.")
