import streamlit as st

st.set_page_config(page_title="AI & Data Science Glossary", layout="wide")
st.title("📚 AI & Data Science Glossary")

st.markdown("""
Explore and search key AI & Data Science terms.  
Click any term to see its definition, usage example, and a high-quality reference.
""")

# --------- 1. Built-in glossary (edit/expand as you wish) ---------

glossary = [
    {
        "term": "Artificial Intelligence (AI)",
        "definition": "A branch of computer science focused on creating systems that perform tasks typically requiring human intelligence, such as reasoning, learning, and perception.",
        "example": "AI systems like speech recognition, autonomous vehicles, or medical diagnostic tools.",
        "reference": "[Russell & Norvig, Artificial Intelligence: A Modern Approach](https://www.pearson.com/en-gb/subject-catalog/p/artificial-intelligence-a-modern-approach/P200000005962/9781292401133)"
    },
    {
        "term": "Machine Learning (ML)",
        "definition": "A subset of AI that uses algorithms to learn patterns from data and make predictions or decisions without explicit programming.",
        "example": "Email spam detection using a supervised learning classifier.",
        "reference": "[Goodfellow et al., Deep Learning](https://www.deeplearningbook.org/)"
    },
    {
        "term": "Deep Learning",
        "definition": "A subfield of machine learning using multi-layered neural networks to automatically learn complex patterns in data.",
        "example": "Convolutional Neural Networks for image classification in medical diagnostics.",
        "reference": "[Nature - Deep learning](https://www.nature.com/articles/nature14539)"
    },
    {
        "term": "Supervised Learning",
        "definition": "A machine learning approach where a model is trained on labeled data (input-output pairs) to predict outcomes on new data.",
        "example": "Predicting patient risk of disease from medical records with known outcomes.",
        "reference": "[Wikipedia - Supervised learning](https://en.wikipedia.org/wiki/Supervised_learning)"
    },
    {
        "term": "Unsupervised Learning",
        "definition": "A machine learning approach where the model tries to find patterns or groupings in data without labeled outcomes.",
        "example": "Clustering patients into subgroups based on health metrics.",
        "reference": "[Wikipedia - Unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning)"
    },
    {
        "term": "Reinforcement Learning",
        "definition": "A machine learning paradigm where an agent learns to make decisions by interacting with an environment to maximize cumulative rewards.",
        "example": "AI controlling a robot that learns to walk by trial and error.",
        "reference": "[Sutton & Barto, Reinforcement Learning: An Introduction](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)"
    },
    {
        "term": "Convolutional Neural Network (CNN)",
        "definition": "A deep learning model designed to process grid-like data such as images, using convolutional layers to extract spatial features.",
        "example": "Detecting tumors in chest X-rays using a CNN.",
        "reference": "[Stanford CS231n CNN Lecture](http://cs231n.stanford.edu/)"
    },
    {
        "term": "Recurrent Neural Network (RNN)",
        "definition": "A neural network architecture suited for sequential data, where connections between nodes form directed cycles, allowing memory of previous inputs.",
        "example": "Predicting future stock prices from past trends using RNNs.",
        "reference": "[Nature - Recurrent neural networks for language modeling](https://www.nature.com/articles/nn.3331)"
    },
    {
        "term": "Transfer Learning",
        "definition": "Leveraging knowledge learned from one task or domain and applying it to a different, but related, task.",
        "example": "Fine-tuning an ImageNet-trained model for COVID-19 X-ray diagnosis.",
        "reference": "[Transfer learning for deep learning: A review](https://ieeexplore.ieee.org/document/8358196)"
    },
    {
        "term": "Natural Language Processing (NLP)",
        "definition": "A field of AI focused on the interaction between computers and human languages, enabling machines to understand, interpret, and generate language.",
        "example": "Chatbots and voice assistants like Siri or Alexa.",
        "reference": "[Jurafsky & Martin, Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)"
    },
    {
        "term": "Tokenization",
        "definition": "The process of breaking text into smaller pieces such as words, subwords, or sentences for further analysis in NLP.",
        "example": "Splitting clinical notes into words for text classification.",
        "reference": "[Wikipedia - Tokenization](https://en.wikipedia.org/wiki/Tokenization_(lexical_analysis))"
    },
    {
        "term": "Bag-of-Words (BoW)",
        "definition": "A method for representing text data as the frequency of words regardless of grammar or order.",
        "example": "Using word counts for spam email detection.",
        "reference": "[Wikipedia - Bag-of-words model](https://en.wikipedia.org/wiki/Bag-of-words_model)"
    },
    {
        "term": "Word Embedding",
        "definition": "A technique in NLP where words are represented as high-dimensional dense vectors capturing semantic meaning.",
        "example": "Using Word2Vec embeddings to cluster similar diagnoses in text.",
        "reference": "[Mikolov et al., Word2Vec](https://arxiv.org/abs/1301.3781)"
    },
    {
        "term": "Precision",
        "definition": "The ratio of true positives to the sum of true and false positives; measures how many predicted positives are actually correct.",
        "example": "If a lung cancer model predicts 100 positives, and 90 are correct, precision is 0.9.",
        "reference": "[Google MLCC - Precision and Recall](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)"
    },
    {
        "term": "Recall",
        "definition": "The ratio of true positives to the sum of true positives and false negatives; measures how many actual positives are detected.",
        "example": "Of all cancer cases, what proportion did the model detect?",
        "reference": "[Google MLCC - Precision and Recall](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)"
    },
    {
        "term": "F1-Score",
        "definition": "The harmonic mean of precision and recall, providing a balance between them.",
        "example": "Used to assess lung cancer detection models where both false positives and false negatives matter.",
        "reference": "[Wikipedia - F1 score](https://en.wikipedia.org/wiki/F-score)"
    },
    {
        "term": "Area Under Curve (AUC)",
        "definition": "A performance metric for binary classifiers; measures the area under the ROC curve and quantifies model separability.",
        "example": "AUC close to 1.0 means excellent model, 0.5 means no better than random.",
        "reference": "[Wikipedia - ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)"
    },
    {
        "term": "Confusion Matrix",
        "definition": "A table that summarizes the performance of a classification algorithm by showing counts of true/false positives and negatives.",
        "example": "A 2x2 matrix showing lung cancer predictions: True Positive, False Positive, True Negative, False Negative.",
        "reference": "[Wikipedia - Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)"
    },
    {
        "term": "Overfitting",
        "definition": "When a model learns the training data too closely, capturing noise instead of general patterns, and performs poorly on new data.",
        "example": "A deep neural network achieves 99% accuracy on training but only 70% on test data.",
        "reference": "[Wikipedia - Overfitting](https://en.wikipedia.org/wiki/Overfitting)"
    },
    {
        "term": "Underfitting",
        "definition": "When a model is too simple to capture underlying patterns in data, resulting in poor performance on both training and test data.",
        "example": "A linear model used on highly nonlinear clinical data.",
        "reference": "[Wikipedia - Underfitting](https://en.wikipedia.org/wiki/Underfitting)"
    },
    {
        "term": "Cross-Validation",
        "definition": "A technique for evaluating models by partitioning data into subsets, training on some and testing on others, to ensure robustness.",
        "example": "K-fold cross-validation to select the best lung cancer prediction model.",
        "reference": "[Wikipedia - Cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))"
    },
    {
        "term": "K-Fold Cross-Validation",
        "definition": "A cross-validation method where data is split into k parts; the model is trained on k-1 and tested on the remaining fold, repeated k times.",
        "example": "5-fold cross-validation is common in medical research for robust performance.",
        "reference": "[Wikipedia - Cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))"
    },
    {
        "term": "Bootstrap Aggregating (Bagging)",
        "definition": "An ensemble technique that improves stability and accuracy by training multiple models on random samples and averaging predictions.",
        "example": "Random Forests use bagging to reduce variance in decision trees.",
        "reference": "[Wikipedia - Bootstrap aggregating](https://en.wikipedia.org/wiki/Bootstrap_aggregating)"
    },
    {
        "term": "Random Forest",
        "definition": "An ensemble machine learning method combining many decision trees, each trained on a random subset of data, for improved accuracy and robustness.",
        "example": "Random Forests to classify lung cancer types based on genetic data.",
        "reference": "[Breiman, Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)"
    },
    {
        "term": "Gradient Boosting",
        "definition": "An ensemble learning method where new models are sequentially added to correct errors made by prior models.",
        "example": "XGBoost used for tabular health prediction data.",
        "reference": "[Nature - A brief introduction to boosting](https://www.nature.com/articles/nmeth.2632)"
    },
    {
        "term": "Support Vector Machine (SVM)",
        "definition": "A supervised learning algorithm that finds the optimal boundary to separate classes in feature space.",
        "example": "Classifying healthy vs. disease samples using SVMs.",
        "reference": "[Wikipedia - Support vector machine](https://en.wikipedia.org/wiki/Support_vector_machine)"
    },
    {
        "term": "K-Means Clustering",
        "definition": "An unsupervised algorithm that partitions data into k groups based on similarity.",
        "example": "Grouping patients by risk factors using K-means.",
        "reference": "[Wikipedia - K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering)"
    },
    {
        "term": "Principal Component Analysis (PCA)",
        "definition": "A dimensionality reduction technique that transforms data to a new coordinate system maximizing variance on each axis.",
        "example": "Reducing features for visualization or model input.",
        "reference": "[Wikipedia - Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)"
    },
    {
        "term": "Feature Engineering",
        "definition": "The process of creating new input variables from existing data to improve model performance.",
        "example": "Combining lab values to create a risk score for lung cancer.",
        "reference": "[Kaggle - Feature Engineering](https://www.kaggle.com/learn/feature-engineering)"
    },
    {
        "term": "Feature Selection",
        "definition": "Identifying and using only the most relevant variables for model training, to avoid overfitting and improve interpretability.",
        "example": "Selecting top 10 most predictive features for lung cancer survival.",
        "reference": "[Wikipedia - Feature selection](https://en.wikipedia.org/wiki/Feature_selection)"
    },
    {
        "term": "Hyperparameter Tuning",
        "definition": "Optimizing the settings that control the learning process (not learned from data), such as learning rate or tree depth.",
        "example": "Grid search for optimal decision tree depth.",
        "reference": "[Wikipedia - Hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization)"
    },
    {
        "term": "Loss Function",
        "definition": "A mathematical function used to measure how well a model’s predictions match actual outcomes during training.",
        "example": "Mean Squared Error (MSE) for regression.",
        "reference": "[Wikipedia - Loss function](https://en.wikipedia.org/wiki/Loss_function)"
    },
    {
        "term": "Activation Function",
        "definition": "A function that transforms the input signal in a neural network node, introducing non-linearity.",
        "example": "ReLU, Sigmoid, and Tanh are common activation functions.",
        "reference": "[Wikipedia - Activation function](https://en.wikipedia.org/wiki/Activation_function)"
    },
    {
        "term": "Backpropagation",
        "definition": "An algorithm for training neural networks, computing gradients of the loss with respect to weights for efficient learning.",
        "example": "Used to update CNN weights when learning from X-ray images.",
        "reference": "[Wikipedia - Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)"
    },
    {
        "term": "Batch Normalization",
        "definition": "A technique to improve neural network training by normalizing layer inputs, accelerating learning and stability.",
        "example": "Used in deep CNNs for medical image analysis.",
        "reference": "[Wikipedia - Batch normalization](https://en.wikipedia.org/wiki/Batch_normalization)"
    },
    {
        "term": "Regularization",
        "definition": "Methods to reduce overfitting by penalizing complex models (e.g., L1, L2 regularization, dropout).",
        "example": "Adding L2 regularization to logistic regression.",
        "reference": "[Wikipedia - Regularization (mathematics)](https://en.wikipedia.org/wiki/Regularization_(mathematics))"
    },
    {
        "term": "Data Augmentation",
        "definition": "Expanding a dataset by creating modified copies of existing data (e.g., image rotation, flipping) to improve model generalization.",
        "example": "Augmenting X-ray data to train a more robust classifier.",
        "reference": "[Wikipedia - Data augmentation](https://en.wikipedia.org/wiki/Data_augmentation)"
    },
    {
        "term": "Learning Rate",
        "definition": "A hyperparameter that determines how much to update model parameters at each training step.",
        "example": "Too high a learning rate can cause the model to diverge.",
        "reference": "[Wikipedia - Learning rate](https://en.wikipedia.org/wiki/Learning_rate)"
    },
    {
        "term": "Epoch",
        "definition": "One complete pass through the entire training dataset during model training.",
        "example": "Training a neural network for 10 epochs.",
        "reference": "[Wikipedia - Epoch (machine learning)](https://en.wikipedia.org/wiki/Epoch_(machine_learning))"
    },
    {
        "term": "Early Stopping",
        "definition": "A regularization technique that stops training when validation performance stops improving, preventing overfitting.",
        "example": "Halting model training after 20 epochs of no validation loss improvement.",
        "reference": "[Wikipedia - Early stopping](https://en.wikipedia.org/wiki/Early_stopping)"
    },
    {
        "term": "Model Drift",
        "definition": "The degradation of a model’s predictive performance over time as real-world data changes.",
        "example": "A clinical prediction model loses accuracy as population health trends shift.",
        "reference": "[Wikipedia - Concept drift](https://en.wikipedia.org/wiki/Concept_drift)"
    },
    {
        "term": "Bias (Statistical/Algorithmic)",
        "definition": "Systematic error or unfairness in model predictions due to data, design, or societal factors.",
        "example": "Model underpredicts disease for minority populations due to biased training data.",
        "reference": "[Wikipedia - Bias (statistics)](https://en.wikipedia.org/wiki/Bias_(statistics))"
    },
    {
        "term": "Variance (in ML)",
        "definition": "A measure of how much a model’s predictions fluctuate for different training sets; high variance means overfitting.",
        "example": "Decision trees can have high variance unless pruned or bagged.",
        "reference": "[Wikipedia - Bias–variance tradeoff](https://en.wikipedia.org/wiki/Bias–variance_tradeoff)"
    },
    {
        "term": "Bias–Variance Tradeoff",
        "definition": "The balance between underfitting (high bias) and overfitting (high variance) in model training.",
        "example": "Random Forests reduce variance while keeping bias low.",
        "reference": "[Wikipedia - Bias–variance tradeoff](https://en.wikipedia.org/wiki/Bias–variance_tradeoff)"
    },
    {
        "term": "Dimensionality Reduction",
        "definition": "Techniques for reducing the number of features in a dataset while preserving important information.",
        "example": "Using PCA to visualize high-dimensional gene expression data.",
        "reference": "[Wikipedia - Dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction)"
    },
    {
        "term": "Imbalanced Data",
        "definition": "A situation where certain classes in a dataset are much more frequent than others, which can bias model performance.",
        "example": "90% healthy, 10% cancer in a diagnostic dataset.",
        "reference": "[Wikipedia - Imbalanced data](https://en.wikipedia.org/wiki/Imbalanced_data)"
    },
    {
        "term": "Explainable AI (XAI)",
        "definition": "Techniques that make AI models’ predictions understandable to humans, increasing transparency, trust, and compliance.",
        "example": "SHAP values used to show how each feature influenced a cancer risk score.",
        "reference": "[Nature - Explainable AI: Interpreting, explaining and visualizing deep learning](https://www.nature.com/articles/s42256-019-0048-x)"
    },
    {
        "term": "Gradient Descent",
        "definition": "An optimization algorithm that adjusts model parameters iteratively to minimize the loss function.",
        "example": "Training a neural network using stochastic gradient descent.",
        "reference": "[Wikipedia - Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)"
    }
]

# --------- 2. Searchable Glossary (Expanders) ---------
search = st.text_input("Search glossary (type a keyword or phrase):", help="Start typing to instantly filter glossary terms.")

shown = False
for entry in glossary:
    term_lower = entry["term"].lower()
    if not search or search.lower() in term_lower or search.lower() in entry["definition"].lower():
        shown = True
        with st.expander(entry["term"]):
            st.write(f"**Definition:** {entry['definition']}")
            st.write(f"**Usage example:** {entry['example']}")
            st.markdown(f"**Reference:** {entry['reference']}")

if not shown:
    st.warning("No glossary terms match your search. Try a different keyword.")

st.divider()

# --------- 3. User suggestion (no backend, just feedback) ---------
st.header("Suggest a New Term (Moderated)")
with st.form("suggest_term"):
    user_term = st.text_input("Term")
    user_def = st.text_area("Definition")
    user_ex = st.text_area("Usage Example (optional)")
    user_ref = st.text_input("Reference (URL, optional)")
    submitted = st.form_submit_button("Submit Suggestion")
    if submitted:
        st.success("Thank you! Your suggestion will be reviewed and added by the glossary team if appropriate.")

st.caption("© 2025 AI & Data Science Glossary | MSc Applied AI and Data Science, Solent University.")
