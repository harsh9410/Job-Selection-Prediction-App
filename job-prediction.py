import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import os

st.title("📊 Job Selection Prediction App")

try:
    if not os.path.exists("dataset.csv"):
        st.error("🚫 dataset.csv not found. Please upload it to the app directory.")
    else:
        data = pd.read_csv("dataset.csv")

        # Label Encoding
        le_qualification = LabelEncoder()
        le_internship = LabelEncoder()
        le_referral = LabelEncoder()
        le_result = LabelEncoder()

        data["Qualification"] = le_qualification.fit_transform(data["Qualification"])
        data["Internship"] = le_internship.fit_transform(data["Internship"])
        data["Referral"] = le_referral.fit_transform(data["Referral"])
        data["Job_Result"] = le_result.fit_transform(data["Job_Result"])

        X = data[[
            "Qualification", "Internship", "Comm_Skill", "Tech_Skill_Level",
            "Certifications", "Interview_Score", "Resume_Score", "Referral"
        ]]
        y = data["Job_Result"]

        model = LinearRegression()
        model.fit(X, y)

        # User Input
        st.subheader("👤 Enter Candidate Details")
        qualification = st.selectbox("Qualification", le_qualification.classes_)
        internship = st.selectbox("Internship", le_internship.classes_)
        referral = st.selectbox("Referral", le_referral.classes_)
        comm_skill = st.slider("Communication Skill", 0, 10, 5)
        tech_skill = st.slider("Technical Skill Level", 0, 10, 5)
        certifications = st.number_input("Certifications", min_value=0, step=1)
        interview_score = st.slider("Interview Score", 0, 10, 5)
        resume_score = st.slider("Resume Score", 0, 10, 5)

        if st.button("🎯 Predict Job Result"):
            try:
                q = le_qualification.transform([qualification])[0]
                i = le_internship.transform([internship])[0]
                r = le_referral.transform([referral])[0]

                input_data = [[
                    q, i, comm_skill, tech_skill,
                    certifications, interview_score, resume_score, r
                ]]

                prediction = model.predict(input_data)[0]
                rounded = int(round(prediction))

                # Clamp to valid index
                rounded = max(0, min(rounded, len(le_result.classes_) - 1))
                result_label = le_result.inverse_transform([rounded])[0]

                if "Not Selected" in result_label:
                    st.error(f"❌ Predicted Result: {result_label}")
                elif "Selected" in result_label:
                    st.success(f"✅ Predicted Result: {result_label}")
                else:
                    st.info(f"ℹ️ Predicted Result: {result_label}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
except Exception as e:
    st.error(f"🚨 Error loading dataset or training model: {e}")
