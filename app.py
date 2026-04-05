import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fake Job Posting Detector", layout="wide")

# --- PROFESSIONAL STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button {
        background-color: #4CAF50; color: white; border-radius: 5px;
        width: 100%; height: 3em; font-weight: bold; border: none;
    }
    .css-1aumxhk {
        background-color: #ffffff; border-radius: 10px; padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def clean_text(text):
    """Must be identical to the cleaning used in training."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)       # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)       # keep only letters
    text = re.sub(r"\s+", " ", text).strip()    # collapse whitespace
    return text

# --- LOAD ALL PIPELINES ---
# fake_job_models.pkl is a dict: {model_name: sklearn Pipeline (tfidf + clf)}
@st.cache_resource
def load_all_pipelines():
    try:
        return joblib.load('fake_job_models.pkl')
    except Exception:
        return None

all_pipelines = load_all_pipelines()

# Model descriptions shown beneath the selectbox
MODEL_INFO = {
    "Naive Bayes": "Fast and memory-efficient. Strong recall but tends to over-flag real postings.",
    "Logistic Regression": "Best overall F1. Interpretable coefficients and well-calibrated probabilities.",
    "Random Forest": "Ensemble model. Captures non-linear patterns; slower but robust.",
}

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Analysis Portal")

    # Model selector — only shown when pipelines loaded successfully
    if all_pipelines:
        available_models = list(all_pipelines.keys())
        selected_model = st.selectbox(
            "Select Model",
            options=available_models,
            index=available_models.index("Logistic Regression")
                  if "Logistic Regression" in available_models else 0,
        )
        st.caption(MODEL_INFO.get(selected_model, ""))
    else:
        selected_model = None
        st.error("fake_job_models.pkl not found. Run the notebook or fake_job_models.py first.")

    st.divider()

    job_title = st.text_input("Job Title")
    location = st.text_input("Location")
    description = st.text_area("Job Description", height=300)

    # INPUT VALIDATION: Word Count Check
    word_count = len(re.findall(r'\w+', description))

    if word_count < 20:
        st.warning(f"⚠️ Minimum 20 words required. Current: {word_count}")
        analyze_btn = st.button("RUN FULL PIPELINE", disabled=True)
    else:
        st.success(f"✅ Length sufficient: {word_count} words")
        analyze_btn = st.button("RUN FULL PIPELINE")

# --- MAIN AREA ---
st.title("🛡️ Fake Job Posting Detection System")

if analyze_btn:
    if all_pipelines is None:
        st.error("Backend Error: 'fake_job_models.pkl' not found. Run the notebook or fake_job_models.py first.")
    else:
        pipeline = all_pipelines[selected_model]

        full_text = f"{job_title} {location} {description}"
        cleaned_text = clean_text(full_text)

        # ELEGANT ERROR HANDLING: Check for Out-of-Vocabulary (Gibberish)
        # We check if the TF-IDF step results in a completely empty vector.
        vectorizer = pipeline.named_steps['tfidf']
        transformed_text = vectorizer.transform([cleaned_text])

        if transformed_text.nnz < 20:
            st.error("🚨 **Analysis Failed:** The text entered does not contain enough recognizable vocabulary found in standard job descriptions. Please provide more specific details.")
        else:
            # Pass raw cleaned text — the pipeline handles vectorisation internally.
            prediction = pipeline.predict([cleaned_text])[0]
            probs = pipeline.predict_proba([cleaned_text])[0]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("System Verdict")
                st.caption(f"Model: **{selected_model}**")
                if prediction == 1:
                    st.error("### ⚠️ FRAUDULENT")
                    st.write(f"**Risk Level:** {probs[1]*100:.1f}%")
                    st.progress(probs[1])
                else:
                    st.success("### ✅ LEGITIMATE")
                    st.write(f"**Legitimacy Confidence:** {probs[0]*100:.1f}%")
                    st.progress(probs[0])

            with col2:
                st.subheader("Statistical Breakdown")
                fig = px.pie(
                    values=probs,
                    names=['Legit', 'Fraud'],
                    color_discrete_sequence=['#4CAF50', '#FF4B4B'],
                    hole=0.4,
                )
                st.plotly_chart(fig, use_container_width=True)

elif word_count == 0:
    st.info("Enter job details in the sidebar and click **RUN FULL PIPELINE** to begin analysis.")
