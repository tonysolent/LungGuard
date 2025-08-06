import streamlit as st
import pandas as pd
import sqlite3
import datetime

st.set_page_config(page_title="Reflection Log & Usage Metrics", layout="wide")
st.title("📖 Reflection Log & Usage Metrics – LungGuard")

# --------------------------------
# 📦 SETUP DATABASE CONNECTION
# --------------------------------
def init_db():
    conn = sqlite3.connect("reflections.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reflections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            week TEXT,
            note TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    return conn

conn = init_db()

# --------------------------------
# 📊 USAGE METRICS (STATIC EXAMPLE)
# --------------------------------
st.subheader("📊 Page Visit Metrics")
usage_data = {
    "Prediction Controls": 84,
    "EDA Page": 47,
    "Visualisation Gallery": 52,
    "Model Playground": 66,
    "Time Series Forecasting": 34,
    "Benchmark Lab": 93,
    "Societal Simulation": 42,
    "Glossary Page": 21,
    "Ethics & Risk Page": 38,
    "Help Page": 19
}
st.bar_chart(pd.Series(usage_data))

st.subheader("🧪 Model Usage Counts")
model_counts = {
    "CNN": 31,
    "DNN": 24,
    "Logistic Regression": 43,
    "Random Forest": 37,
    "Gradient Boosting": 29,
    "KNN": 15,
    "SVM": 18,
    "Naïve Bayes": 9
}
st.bar_chart(pd.Series(model_counts))

# --------------------------------
# 🧠 REFLECTION LOG TABLE
# --------------------------------
st.subheader("🧠 Developer Reflection Log")

# Load from DB
def load_reflections():
    cursor = conn.cursor()
    cursor.execute("SELECT id, week, note, timestamp FROM reflections ORDER BY id DESC")
    return cursor.fetchall()

# Add new reflection
with st.form("new_reflection"):
    col1, col2 = st.columns([1, 3])
    with col1:
        week = st.text_input("Week")
    with col2:
        note = st.text_area("Reflection Note")
    submitted = st.form_submit_button("➕ Add Reflection")
    if submitted and week and note:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("INSERT INTO reflections (week, note, timestamp) VALUES (?, ?, ?)", (week, note, timestamp))
        conn.commit()
        st.success("Reflection added!")

# Display Table with Edit/Delete
reflections = load_reflections()
for r in reflections:
    rid, week, note, ts = r
    with st.expander(f"📌 {week} – {ts}"):
        st.markdown(f"📝 {note}")
        col1, col2, col3 = st.columns([1, 1, 6])
        

        
        # Delete
        with col2:
            if st.button("🗑️ Delete", key=f"delete_{rid}"):
                conn.execute("DELETE FROM reflections WHERE id = ?", (rid,))
                conn.commit()
                st.warning("Reflection deleted.")
                st.experimental_rerun()

st.markdown("---")
st.caption("© 2025 LungGuard – Reflection & Metrics | MSc Applied AI and Data Science")
