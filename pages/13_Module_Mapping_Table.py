import streamlit as st

st.set_page_config(page_title="LungGuard Capabilities", layout="wide")
st.title("🧠 LungGuard – System Capabilities & Built Features")

st.markdown("""
This table presents the **core capabilities, techniques, and real features built in LungGuard**,  
highlighting the full technical scope of the system as an end-to-end AI solution.
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
      <th>🧩 Capability Area</th>
      <th>📋 Techniques & Methods Used</th>
      <th>✅ Real Features Built in LungGuard</th>
    </tr>
  </thead>
  <tbody>

    <tr>
      <td><b>Multimodal AI</b></td>
      <td>
        CNNs, DNNs<br>
        NLP/Text embeddings<br>
        Feature Fusion
      </td>
      <td>
        CNN for Chest X-rays<br>
        DNN for tabular data<br>
        Clinical Text Embeddings<br>
        Fusion Model for combined prediction
      </td>
    </tr>

    <tr>
      <td><b>Explainable AI</b></td>
      <td>
        SHAP<br>
        Grad-CAM
      </td>
      <td>
        SHAP for tabular and text explanations<br>
        Grad-CAM for image visualisation
      </td>
    </tr>

    <tr>
      <td><b>Machine Learning Models</b></td>
      <td>
        SVM, KNN, Naïve Bayes<br>
        Model benchmarking
      </td>
      <td>
        SVM, KNN, Naïve Bayes (Experimental Models)<br>
        Model Benchmarking Lab with performance metrics
      </td>
    </tr>

    <tr>
      <td><b>Data Analytics & Visualisation</b></td>
      <td>
        EDA<br>
        Correlation analysis<br>
        Dashboarding
      </td>
      <td>
        EDA Page: Missing values and type detection<br>
        Correlation heatmaps and boxplots<br>
        Visualisation Gallery: Pie, Radar, Treemap
      </td>
    </tr>

    <tr>
      <td><b>Model Evaluation</b></td>
      <td>
        ROC Curves<br>
        Calibration<br>
        Performance metrics
      </td>
      <td>
        ROC curves and calibration plots<br>
        Benchmarking radar charts<br>
        AUC and F1 evaluation
      </td>
    </tr>

    <tr>
      <td><b>Forecasting & Simulation</b></td>
      <td>
        Time series modelling<br>
        Scenario simulation
      </td>
      <td>
        ARIMA time series forecasting<br>
        Societal impact simulation<br>
        Patient workflow simulation
      </td>
    </tr>

    <tr>
      <td><b>Ethics & Risk</b></td>
      <td>
        Bias detection<br>
        Ethical frameworks
      </td>
      <td>
        Risk/Bias scanner (imbalance, privacy)<br>
        BCS Code of Conduct page
      </td>
    </tr>

    <tr>
      <td><b>System Engineering</b></td>
      <td>
        Modular programming<br>
        Deployment architecture
      </td>
      <td>
        Streamlit app with modular pages<br>
        Python `models/`, `utils/`, `pages/` structure<br>
        Checkpoint loading via Google Drive
      </td>
    </tr>

    <tr>
      <td><b>Clustering & Advanced Methods</b></td>
      <td>
        Unsupervised learning
      </td>
      <td>
        DBSCAN, Agglomerative, KMeans clustering
      </td>
    </tr>

    <tr>
      <td><b>Reporting & Outputs</b></td>
      <td>
        Report generation<br>
        Data export
      </td>
      <td>
        PDF report generation with ReportLab<br>
        Exportable analytics outputs
      </td>
    </tr>

    <tr>
      <td><b>User Experience</b></td>
      <td>
        Interactive UI design
      </td>
      <td>
        Real-time prediction interface<br>
        File upload and patient selection<br>
        Interactive dashboards
      </td>
    </tr>

  </tbody>
</table>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("© 2025 LungGuard | System Capabilities by Tony Livins")
