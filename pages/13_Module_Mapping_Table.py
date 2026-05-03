import streamlit as st

st.set_page_config(page_title="LungGuard Feature Alignment", layout="wide")
st.title("📚 LungGuard Feature Alignment – Individual System Build")

st.markdown("""
This table summarises the key concepts, tools, and outcomes demonstrated through the LungGuard artefact, focusing on the **individual features built in the system** rather than module mapping.
""", unsafe_allow_html=True)

st.markdown("""
<style>
th, td {
    border: 1px solid #444;
    padding: 8px;
}
table {
    border-collapse: collapse;
    width: 100%;
}
th {
    background-color: #222;
    color: #eee;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<table>
  <thead>
    <tr>
      <th>🧠 Capability Area</th>
      <th>📋 Techniques & Concepts</th>
      <th>✅ Real Features Built in LungGuard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Artificial Intelligence Foundations</b></td>
      <td>
        - CNNs, DNNs<br>
        - NLP/Text embeddings<br>
        - SVM, KNN, Naïve Bayes<br>
        - Explainable AI (SHAP, Grad-CAM)<br>
        - Reinforcement Learning (conceptual)<br>
        - Model Evaluation and Benchmarking
      </td>
      <td>
        CNN for Chest X-rays<br>
        DNN for tabular data<br>
        Clinical Text Embeddings<br>
        SHAP for tabular/text<br>
        Grad-CAM for images<br>
        SVM, KNN, Naïve Bayes (Experimental Models)<br>
        RL Concept Page (Simulation Tab)<br>
        Model Benchmarking Lab with Metrics
      </td>
    </tr>
    <tr>
      <td><b>Applied AI & Decision Systems</b></td>
      <td>
        - Forecasting Models<br>
        - AI Risk & Bias Audits<br>
        - Ethical Principles<br>
        - Societal Simulation<br>
        - AI for Decision-Making
      </td>
      <td>
        ARIMA Time Series Forecasting<br>
        BCS Code of Conduct Page<br>
        Risk/Bias Scanner (Imbalance, Privacy)<br>
        Societal Impact + Workflow Mapping<br>
        Real-world Simulation: Patient Stories
      </td>
    </tr>
    <tr>
      <td><b>Data Analytics & Visualisation</b></td>
      <td>
        - EDA<br>
        - Dashboarding<br>
        - Correlation + Outlier Analysis<br>
        - Visualisation Techniques<br>
        - Analytical Communication
      </td>
      <td>
        EDA Page: Missing Values, Type Detection<br>
        Correlation Heatmaps + Boxplots<br>
        Visualisation Gallery: Pie, Radar, Treemap<br>
        Benchmarking Radar, ROC, Calibration Plots<br>
        PDF Report Export
      </td>
    </tr>
    <tr>
      <td><b>Software Engineering & System Design</b></td>
      <td>
        - Modular Programming<br>
        - App Deployment<br>
        - Clustering Techniques<br>
        - Report Automation<br>
        - Use of Libraries (matplotlib, pandas, etc.)
      </td>
      <td>
        Streamlit App with Modular Pages<br>
        Python models/, utils/, pages/ structure<br>
        DBSCAN, Agglomerative, KMeans Clustering<br>
        PDF Report Generation with ReportLab<br>
        System Architecture + Audit + Deployment Mapping<br>
        Reflection Log with SQLite<br>
        Editable CRUD UI using real-time Streamlit + SQL integration<br>
      </td>
    </tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("© 2025 LungGuard | Individual AI System Build and Feature Alignment by Tony Livins")
