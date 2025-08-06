import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="Societal Impact & Scenario Simulation", layout="wide")
st.title("🌍 Societal Impact & Scenario Simulation")

st.markdown("""
Explore and critically reflect on the impact of your AI/data science system.
- Estimate benefits and risks for stakeholders
- Map to NHS, public health, and UN SDGs
- Try “what-if” scenario simulation
- Read patient stories/personas
- View clinical workflow mapping
- Download an impact statement for reporting/appendix
""")

# --- 1. Stakeholder & Population Impact Estimator ---
st.header("1. Stakeholder & Population Impact Estimator")
groups = [
    "Patients (direct benefit)",
    "Clinicians (decision support)",
    "Health System (NHS efficiency)",
    "Underserved or vulnerable populations",
    "Industry/business",
    "Environment",
    "General public (health literacy, awareness)"
]
st.markdown("**Estimate potential reach and impact for each stakeholder group:**")
impact_inputs = {}
for group in groups:
    impact_inputs[group] = st.slider(
        f"{group} (0 = no impact, 100 = transformative impact)",
        min_value=0, max_value=100, value=50, step=5,
        help=f"How much positive change could your solution make for {group.lower()}?"
    )

st.markdown("---")
st.subheader("2. Risks, Accessibility & Equity")
risk_score = st.slider(
    "Overall Risk of Harm (0 = none, 100 = high)",
    min_value=0, max_value=100, value=10, step=5,
    help="Consider possible negative effects, misuse, or unfairness (e.g., bias, digital divide)."
)
access_score = st.slider(
    "Accessibility & Inclusion (0 = exclusive, 100 = universal)",
    min_value=0, max_value=100, value=70, step=5,
    help="Will all relevant groups be able to access and benefit? Digital, language, disability, etc."
)

st.markdown("---")
st.subheader("3. Map to SDGs / NHS / Public Health Priorities")
sdg_options = [
    "Good Health & Well-being (SDG 3)",
    "Reduced Inequalities (SDG 10)",
    "Quality Education (SDG 4)",
    "Gender Equality (SDG 5)",
    "Industry, Innovation & Infrastructure (SDG 9)",
    "Climate Action (SDG 13)",
    "Other/Custom"
]
selected_sdgs = st.multiselect(
    "Which Sustainable Development Goals or priorities are supported?",
    sdg_options,
    default=["Good Health & Well-being (SDG 3)"],
    help="Select all that apply. Supports advanced impact mapping and reporting."
)
custom_impact = st.text_area(
    "Additional Impact or Societal Reflection (optional)",
    help="Describe any other expected impact, e.g., new evidence, public trust, policy, cross-sector benefit."
)

st.markdown("---")
st.subheader("4. Visualise Impact Profile")
st.markdown("**Stakeholder Impact Profile:**")
fig, ax = plt.subplots(figsize=(8, 3))
ax.barh(list(impact_inputs.keys()), list(impact_inputs.values()), color="seagreen")
ax.set_xlim(0, 100)
ax.set_xlabel("Impact Level (0–100)")
st.pyplot(fig)

st.markdown("**Risk & Accessibility Profile:**")
fig2, ax2 = plt.subplots(figsize=(5, 1.5))
ax2.barh(["Risk of Harm"], [risk_score], color="crimson")
ax2.barh(["Accessibility & Inclusion"], [access_score], color="royalblue")
ax2.set_xlim(0, 100)
st.pyplot(fig2)

# --- 5. Scenario Simulation ("What-if") ---
st.divider()
st.header("5. Scenario Simulation ('What-if' Analysis)")
st.markdown("""
Adjust key parameters below to simulate impact under different real-world scenarios.
""")
col1, col2, col3 = st.columns(3)
with col1:
    prevalence = st.slider("Disease Prevalence (%)", 1, 50, 10, help="Base rate of disease in simulated population.")
with col2:
    sensitivity = st.slider("Model Sensitivity (%)", 50, 100, 90, help="True positive rate: detects real disease.")
with col3:
    specificity = st.slider("Model Specificity (%)", 50, 100, 85, help="True negative rate: correctly identifies healthy.")
population = st.number_input("Simulated Population Size", min_value=1000, max_value=100000, value=10000, step=1000)
n_disease = int(population * prevalence / 100)
n_healthy = population - n_disease
tp = int(n_disease * sensitivity / 100)
fn = n_disease - tp
tn = int(n_healthy * specificity / 100)
fp = n_healthy - tn

st.markdown("**Simulation Outcomes:**")
fig_sim, ax_sim = plt.subplots(figsize=(6, 2.5))
labels = ["True Positive (Detected)", "False Negative (Missed)", "True Negative (Healthy)", "False Positive (False Alarm)"]
values = [tp, fn, tn, fp]
colors = ["forestgreen", "crimson", "steelblue", "orange"]
ax_sim.barh(labels, values, color=colors)
ax_sim.set_xlabel("Number of People")
st.pyplot(fig_sim)
st.markdown(f"""
- **Detected cases:** {tp}  
- **Missed cases:** {fn}  
- **Unnecessary alarms:** {fp}  
- **Correctly reassured:** {tn}
""")
st.info("Use this to reflect: What if prevalence increases? Or the model is less accurate? How does this change societal value or risk?")

# --- 6. Patient Stories / Personas ---
st.divider()
st.header("6. Patient Stories & Personas")
patient_stories = [
    {
        "name": "Anna, 47, Non-smoker, Mild Symptoms",
        "story": "Anna experienced a persistent cough. Despite being a non-smoker, she was anxious and her GP referred her for screening. LungGuard flagged her X-ray as low-risk, giving her reassurance and reducing unnecessary follow-ups.",
        "reflection": "AI can reduce anxiety, save healthcare resources, and support patient-centred care—if the model is accurate and inclusive."
    },
    {
        "name": "Tom, 61, Former Smoker, Rural Resident",
        "story": "Tom had limited access to specialist care and delayed seeing a doctor. LungGuard identified a suspicious lesion and flagged him for early intervention, leading to prompt referral and a better prognosis.",
        "reflection": "AI tools can help bridge care gaps for underserved populations but must be validated for fairness and access."
    },
    {
        "name": "Rashida, 55, Asthma, Complex History",
        "story": "Rashida’s chest X-ray showed overlapping symptoms. LungGuard provided a risk score but recommended human review, ensuring her complex case wasn’t missed by automation alone.",
        "reflection": "Clinical oversight and AI together provide safer care, and models must be transparent about uncertainty."
    }
]
for p in patient_stories:
    with st.expander(f"{p['name']}"):
        st.write(f"**Story:** {p['story']}")
        st.info(f"**Critical Reflection:** {p['reflection']}")
st.markdown("These patient stories are **simulated personas** designed for scenario exploration, ethical reflection, and clinical pathway testing within this system.")

# --- 7. Clinical Workflow Mapping ---
st.divider()
st.header("7. Clinical Workflow Mapping")
st.markdown("""
**LungGuard-Enabled Diagnostic Pathway:**
1. Patient presents to GP or clinic
2. Initial assessment (symptoms/history)
3. **Chest X-ray performed**
4. **LungGuard AI analysis applied to X-ray (+ tabular/text data if available)**
5. AI risk score + explainability delivered to clinician
6. Clinician reviews results (with SHAP/Grad-CAM if flagged)
7. Decision: Routine follow-up, further imaging, or specialist referral
8. Patient receives tailored feedback/next steps
""")
st.image(
    "LungGuard_Clinical_Workflow.png",
    caption="LungGuard AI-Enabled Clinical Workflow"
)
st.info("Workflow mapping clarifies where clinical expertise, patient experience, and AI interact, supporting ethical, effective deployment.")


# --- 8. Downloadable Impact Statement ---
st.divider()
st.header("Downloadable Impact Statement")
impact_statement = f"""
Societal Impact & Benefit Statement (Generated {datetime.datetime.now().date()})
=================================================================
**Stakeholder Impact Estimates:**
"""
for group in groups:
    impact_statement += f"\n- {group}: {impact_inputs[group]}/100"
impact_statement += f"\n\n**Risk of Harm:** {risk_score}/100"
impact_statement += f"\n**Accessibility & Inclusion:** {access_score}/100"
impact_statement += "\n\n**SDGs/Public Health Priorities Supported:**\n"
for sdg in selected_sdgs:
    impact_statement += f"- {sdg}\n"
if custom_impact.strip():
    impact_statement += f"\n**Additional Societal Impact:**\n{custom_impact}\n"

impact_statement += f"\n\n---\nScenario Simulation Outcomes:\n- Population: {population}\n- Prevalence: {prevalence}%\n- Sensitivity: {sensitivity}%\n- Specificity: {specificity}%\n"
impact_statement += f"- Detected cases: {tp}\n- Missed cases: {fn}\n- Unnecessary alarms: {fp}\n- Correctly reassured: {tn}\n"
impact_statement += "\n\n---\nExample Patient Stories / Clinical Workflow included (see app for detail).\n"
impact_statement += "\n---\nThis page supports critical reflection on societal, clinical, and ethical benefit, aligned with NHS and SDG."

st.download_button(
    "Download Impact Statement",
    impact_statement,
    file_name="societal_impact_statement.txt",
    mime="text/plain",
    help="Download a complete summary for reporting or appendices."
)

st.caption("© 2025 Societal Impact & Scenario Simulation | MSc Applied AI and Data Science, Solent University.")
