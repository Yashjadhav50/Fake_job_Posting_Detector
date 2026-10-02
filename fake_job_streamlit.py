import streamlit as st
import pickle


# ==============================
# Load Saved Model and Vectorizer
# ==============================

with open("fake_job_model.pkl", "rb") as file:
    lr = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🔍",
    layout="centered"
)


# ==============================
# Title
# ==============================

st.title("🔍 Fake Job Posting Detector")

st.write(
    "Enter the details of a job posting below "
    "to check whether it is likely to be real or fraudulent."
)

st.divider()


# ==============================
# Input Fields
# ==============================

title = st.text_input(
    "Job Title",
    placeholder="Example: Data Analyst"
)

company_profile = st.text_area(
    "Company Profile",
    placeholder="Enter company information..."
)

description = st.text_area(
    "Job Description",
    placeholder="Enter the job description..."
)

requirements = st.text_area(
    "Requirements",
    placeholder="Enter required skills and qualifications..."
)

benefits = st.text_area(
    "Benefits",
    placeholder="Enter salary, benefits, perks, etc..."
)


# ==============================
# Prediction
# ==============================

if st.button("🔍 Check Job Posting"):

    if not title and not description and not requirements:

        st.warning("⚠️ Please enter some job posting details.")

    else:

        # Combine text exactly like we did during training
        job_text = (
            title + " " +
            company_profile + " " +
            description + " " +
            requirements + " " +
            benefits
        )

        # Convert text using saved TF-IDF vectorizer
        job_tfidf = vectorizer.transform([job_text])

        # Prediction using saved Logistic Regression model
        prediction = lr.predict(job_tfidf)[0]

        # Prediction probability
        probability = lr.predict_proba(job_tfidf)[0]


        # ==============================
        # Display Result
        # ==============================

        if prediction == 1:

            fake_probability = probability[1] * 100

            st.error("⚠️ Potentially FAKE Job Posting")

            st.write(
                f"Fake Job Probability: **{fake_probability:.2f}%**"
            )

        else:

            real_probability = probability[0] * 100

            st.success("✅ This Job Posting Appears to be REAL")

            st.write(
                f"Real Job Probability: **{real_probability:.2f}%**")


# ==============================
# Footer
# ==============================

st.divider()

st.caption(
    "By Yash |"
    "Fake Job Posting Detection | "
    "TF-IDF + Logistic Regression + Streamlit"
)
