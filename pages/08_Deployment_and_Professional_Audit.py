import streamlit as st
import pandas as pd
import numpy as np
import datetime

st.set_page_config(page_title="Deployment & Professional Audit", layout="wide")
st.title("🚦 Deployment & Professional Audit Dashboard")

st.markdown("""
This dashboard simulates real-world AI deployment with automated monitoring and a professional audit checklist.
- Track for data drift, model decay, and compliance.
- Complete an audit for GDPR, NHS, FDA/EMA, and ICO readiness.
""")

# --- 1. Simulate Model/Data Drift Monitoring ---
st.header("1. Deployment Monitoring: Drift & Alerts")

uploaded_file = st.file_uploader(
    "Upload production data for monitoring (CSV)", 
    type=["csv"], 
    help="Upload new production data to simulate model monitoring for drift and outlier detection."
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded production data: {df.shape[0]} rows × {df.shape[1]} columns.")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Simulate drift: compare to old (reference) mean/std (dummy values for this example)
    st.subheader("Feature Drift Simulation")
    st.caption("Shows significant changes in key statistics (mean/std) vs. a reference, indicating drift or operational issues.")
    reference_stats = {col: {"mean": 50, "std": 10} for col in df.select_dtypes(np.number).columns}
    drift_report = ""
    for col in df.select_dtypes(np.number).columns:
        prod_mean = df[col].mean()
        prod_std = df[col].std()
        ref_mean = reference_stats[col]["mean"]
        ref_std = reference_stats[col]["std"]
        mean_drift = abs(prod_mean - ref_mean) / (ref_std + 1e-6)
        std_drift = abs(prod_std - ref_std) / (ref_std + 1e-6)
        if mean_drift > 0.5 or std_drift > 0.5:
            drift_report += f"- **{col}**: Mean drift = {mean_drift:.2f}, Std drift = {std_drift:.2f} (Potential issue)\n"
    if drift_report:
        st.warning("Possible drift detected:\n" + drift_report)
    else:
        st.success("No significant drift detected in numeric features.")

    # Outlier/alert simulation (simple threshold)
    st.subheader("Outlier Alerts")
    alert_count = 0
    for col in df.select_dtypes(np.number).columns:
        outliers = df[(df[col] < df[col].mean() - 3*df[col].std()) | (df[col] > df[col].mean() + 3*df[col].std())]
        if not outliers.empty:
            st.error(f"Outlier detected in **{col}**: {len(outliers)} case(s) (Check for errors or new risks!)")
            alert_count += len(outliers)
    if alert_count == 0:
        st.success("No critical outliers detected.")

    st.markdown("---")

else:
    st.info("Upload simulated production data to activate monitoring.")

# --- 2. Professional Audit Checklist ---
st.header("2. Professional Audit Checklist (GDPR, NHS, ICO, FDA/EMA)")

st.markdown("""
**Complete this audit before deployment to demonstrate compliance and readiness.**
""")

audit_questions = [
    "Have you removed or anonymized all direct and indirect patient identifiers (GDPR/NHS)?",
    "Is data storage encrypted and access-controlled?",
    "Do you have patient or data subject consent for use in this system?",
    "Is there a documented process for handling data breaches or privacy incidents?",
    "Has the model been tested for bias and fairness across key groups?",
    "Is model explainability (e.g., SHAP/Grad-CAM) available to clinicians/stakeholders?",
    "Is model performance monitored for drift or decay in production?",
    "Can users/patients opt out or request deletion (data subject rights)?",
    "Is the AI system and decision process documented for regulatory/clinical audit?",
    "Is a clinician or responsible professional identified as accountable for AI output?",
    "Have you reviewed compliance with all relevant local/international guidelines (ICO, FDA, EMA, NHSX, BCS Code)?"
]

if 'audit_submitted' not in st.session_state:
    st.session_state.audit_submitted = False
    st.session_state.audit_answers = None

with st.form("audit_checklist"):
    audit_answers = []
    for idx, q in enumerate(audit_questions):
        audit_answers.append(st.radio(q, ["Yes", "No"], index=None, key=f"audit_{idx}", help="Answer Yes if this compliance requirement is met."))
    submitted_audit = st.form_submit_button("Complete Audit")

if submitted_audit:
    st.session_state.audit_submitted = True
    st.session_state.audit_answers = audit_answers

if st.session_state.audit_submitted and st.session_state.audit_answers:
    audit_answers = st.session_state.audit_answers
    n_yes = sum(ans == "Yes" for ans in audit_answers)
    st.success(f"Audit Compliance Score: {n_yes} / {len(audit_questions)}")
    if n_yes == len(audit_questions):
        st.balloons()
        st.info("Audit complete: All best-practice compliance requirements met. Ready for real-world deployment and clinical/industry audit.")
    elif n_yes >= 8:
        st.info("Good: Most compliance points met, but review any gaps before final deployment.")
    else:
        st.warning("Critical: Gaps remain in compliance—review requirements before deployment!")

    # Generate and enable download of audit report
    report = "Deployment & Professional Audit Checklist\n"
    report += "-"*50 + "\n"
    for q, a in zip(audit_questions, audit_answers):
        report += f"{q}\n - Answer: {a}\n"
    report += f"\nCompliance Score: {n_yes} / {len(audit_questions)}\n"
    report += f"Audited by: {st.session_state.get('user', 'anonymous')}\n"
    report += f"Audit date: {datetime.datetime.now().isoformat()}\n"
    st.download_button(
        "Download Audit Report",
        report,
        file_name="audit_readiness_report.txt",
        mime="text/plain",
        help="Download a full audit/compliance report for documentation or submission."
    )

st.divider()
st.info("""
**Why this matters:**  
Professional deployment and real-world use of AI in health or business requires more than just high model accuracy.  
""")

st.caption("© 2025 Deployment & Professional Audit Dashboard | MSc Applied AI and Data Science, Solent University.")
