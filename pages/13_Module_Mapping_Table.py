import streamlit as st

st.set_page_config(page_title="Module Feature Alignment", layout="wide")
st.title("📚 Module Alignment Summary – LungGuard System")

st.markdown("""
This table summarises how the LungGuard artefact maps to the key concepts, tools, and outcomes from each MSc module in **Applied AI & Data Science** at Solent University.
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
      <th>🎓 Module</th>
      <th>📋 Features Listed in Curriculum</th>
      <th>✅ Real Features Built in LungGuard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>COM727</b><br>Introduction to AI</td>
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
      <td><b>COM724</b><br>Applied AI in Business</td>
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
      <td><b>COM725</b><br>Data Analytics & Visualisation</td>
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
      <td><b>COM731</b><br>Programming for Problem Solving</td>
      <td>
        - Modular Programming<br>
        - App Deployment<br>
        - Clustering Techniques<br>
        - Report Automation<br>
        - Use of Libraries (matplotlib, pandas, etc.)
      </td>
      <td>
        Streamlit App with Modular Pages<br>
        Python `utils/`, `models/`, `pages/` folders<br>
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
st.caption("© 2025 LungGuard | MSc Applied AI and Data Science | Module Mapping Summary by Tony Livins")
