import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="AI Ethics & Risk Assessment", layout="wide")
st.title("🛡️ AI Ethics & Risk Assessment")

st.markdown("""
Assess your dataset and your project's alignment with AI ethics, bias, risk, and transparency.
- Take the interactive quiz.
- Complete the AI explainability & transparency checklist.
- Review the BCS Code of Conduct and AI Principles.
""")

# --- 1. Interactive AI Ethics Quiz ---
st.header("1. AI Ethics & Public Health Quiz")
quiz_questions = [
    {
        "q": "Which of these is an example of algorithmic bias?",
        "options": [
            "A model that underpredicts a disease for one ethnic group.",
            "A model that predicts accurately for all subgroups.",
            "A model that never makes errors."
        ],
        "answer": 0
    },
    {
        "q": "What is the primary principle of 'Data Minimisation' in data ethics?",
        "options": [
            "Collect as much data as possible.",
            "Collect only the data needed for the task.",
            "Never collect personal data."
        ],
        "answer": 1
    },
    {
        "q": "Which of the following is a key pillar of the BCS Code of Conduct?",
        "options": [
            "Maximising profits above all else.",
            "Protecting public interest and privacy.",
            "Hiding model code from audit."
        ],
        "answer": 1
    },
    {
        "q": "What should you do if a dataset contains unbalanced classes (e.g., 90% 'no cancer', 10% 'cancer')?",
        "options": [
            "Ignore it; models can handle it.",
            "Use methods like re-sampling, weighting, or balanced metrics.",
            "Remove the minority class."
        ],
        "answer": 1
    },
    {
        "q": "Which is a privacy risk when using clinical notes?",
        "options": [
            "Notes contain only model predictions.",
            "Notes may include patient names or IDs.",
            "Notes are always anonymous."
        ],
        "answer": 1
    }
]
random.shuffle(quiz_questions)
quiz_score = 0

with st.form("ethics_quiz"):
    st.markdown("**Answer these questions to test your AI ethics knowledge.**")
    user_answers = []
    for i, q in enumerate(quiz_questions):
        user_answers.append(st.radio(q["q"], q["options"], index=None, key=f"q_{i}", help="Select the best answer."))
    submitted_quiz = st.form_submit_button("Submit Quiz")

if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if submitted_quiz:
    st.session_state.quiz_submitted = True
    quiz_score = sum(
        int(user_answers[i] is not None and user_answers[i] == q["options"][q["answer"]])
        for i, q in enumerate(quiz_questions)
    )
if st.session_state.quiz_submitted:
    st.success(f"You scored {quiz_score} out of {len(quiz_questions)}.")
    if quiz_score == len(quiz_questions):
        st.balloons()
    elif quiz_score >= 4:
        st.info("Strong grasp of AI ethics! Review any missed questions for distinction.")
    else:
        st.warning("Consider reviewing the BCS Code and key fairness concepts below.")

st.divider()

# --- 2. AI Explainability & Transparency Checklist ---
st.header("2. AI Explainability & Transparency Checklist")

st.markdown("""
Complete this checklist to assess the transparency and interpretability of your AI model or data science project.
""")

explain_questions = [
    "Is the model's purpose and intended use clearly described to stakeholders?",
    "Is the input data fully documented and available for audit?",
    "Are the features used in the model clearly defined and interpretable?",
    "Can you explain why a given prediction was made (using SHAP, LIME, or similar methods)?",
    "Is the model code or pipeline available for external review?",
    "Have you documented model performance on all key subgroups (e.g., gender, ethnicity)?",
    "Are any post-hoc explanations (e.g., feature importance) provided to users?",
    "Is there a clear process for addressing user or stakeholder queries about predictions?"
]

if 'xai_submitted' not in st.session_state:
    st.session_state.xai_submitted = False
    st.session_state.explain_answers = None

with st.form("xai_checklist"):
    explain_answers = []
    for idx, q in enumerate(explain_questions):
        explain_answers.append(st.radio(q, ["Yes", "No"], index=None, key=f"xai_{idx}", help="Select Yes if this transparency aspect is in place."))
    submitted_xai = st.form_submit_button("Check Compliance")

if submitted_xai:
    st.session_state.xai_submitted = True
    st.session_state.explain_answers = explain_answers

if st.session_state.xai_submitted and st.session_state.explain_answers:
    explain_answers = st.session_state.explain_answers
    n_yes = sum(ans == "Yes" for ans in explain_answers)
    st.success(f"Transparency Score: {n_yes} out of {len(explain_questions)}")
    if n_yes == len(explain_questions):
        st.balloons()
        st.info("Excellent: Your project meets all core transparency and explainability standards.")
    elif n_yes >= 5:
        st.info("Good: Your project is transparent, but review the unchecked points to reach best practice.")
    else:
        st.warning("Caution: Improve your project’s explainability to align with ethical AI and regulatory standards.")

    report = "AI Explainability & Transparency Checklist\n"
    report += "-"*40 + "\n"
    for q, a in zip(explain_questions, explain_answers):
        report += f"{q}\n - Answer: {a}\n"
    report += f"\nTransparency Score: {n_yes} / {len(explain_questions)}\n"
    st.download_button(
        "Download Checklist Report",
        report,
        file_name="transparency_checklist.txt",
        mime="text/plain",
        help="Download your completed transparency checklist for project documentation."
    )

st.divider()

# --- 3. BCS Code of Conduct & AI Principles ---
st.header("3. BCS Code of Conduct & AI Principles (Summary)")
st.markdown("""
**BCS Code of Conduct (Key Points):**
- Act in the public interest and with integrity at all times.
- Respect privacy and confidentiality of personal data.
- Be honest and trustworthy; avoid deception.
- Take due care with sensitive or high-impact systems.
- Disclose conflicts of interest and report unethical practices.
- Maintain professional competence and keep skills up to date.
""")
st.markdown("""
**AI Ethics Principles (for Responsible AI):**
- **Fairness:** Avoid bias, discrimination, and unjust impacts in models.
- **Transparency:** Ensure model decisions can be understood and audited.
- **Accountability:** Take responsibility for system outputs and data practices.
- **Privacy:** Protect data subjects’ identities and sensitive information.
- **Safety:** Ensure AI systems do not cause harm.
""")

st.divider()
st.info("""
**Why this matters:**  
Ethics, fairness, and responsible practice are essential for safe, trustworthy AI in business and public health.  
Understanding and demonstrating compliance is now a core MSc and industry requirement.
""")

st.caption("© 2025 AI Ethics & Risk Assessment | MSc Applied AI and Data Science, Solent University.")
