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

# --- LOAD THE FULL PIPELINE ---
@st.cache_resource
def load_backend():
    try:
        return joblib.load('fake_job_model.pkl')
    except:
        return None

backend_pipeline = load_backend()

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Analysis Portal")
    st.info("Using the Multinomial Naive Bayes Engine (98.3% Accuracy)")
    job_title = st.text_input("Job Title")
    location = st.text_input("Location")
    description = st.text_area("Job Description", height=300)
    
    # 1. INPUT VALIDATION: Word Count Check
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
    if backend_pipeline is None:
        st.error("Backend Error: 'fake_job_model.pkl' not found. Run the notebook first.")
    else:
        full_text = f"{job_title} {location} {description}"
        cleaned_text = clean_text(full_text)
        
        # 3. ELEGANT ERROR HANDLING: Check for Out-of-Vocabulary (Gibberish)
        # We check if the TF-IDF transform results in a completely empty vector
        vectorizer = backend_pipeline.named_steps['tfidf']
        transformed_text = vectorizer.transform([cleaned_text])
        
        if transformed_text.nnz < 20:
            st.error("🚨 **Analysis Failed:** The text entered does not contain enough recognizable vocabulary found in standard job descriptions. Please provide more specific details.")
        else:
            # 2. DISPLAY CONFIDENCE SCORES
            prediction = backend_pipeline.predict([transformed_text])[0]
            probs = backend_pipeline.predict_proba([transformed_text])[0]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("System Verdict")
                if prediction == 1:
                    st.error(f"### ⚠️ FRAUDULENT")
                    # Using a progress bar to visually represent risk
                    st.write(f"**Risk Level:** {probs[1]*100:.1f}%")
                    st.progress(probs[1]) 
                else:
                    st.success(f"### ✅ LEGITIMATE")
                    st.write(f"**Legitimacy Confidence:** {probs[0]*100:.1f}%")
                    st.progress(probs[0])
            
            with col2:
                st.subheader("Statistical Breakdown")
                fig = px.pie(values=probs, names=['Legit', 'Fraud'], 
                             color_discrete_sequence=['#4CAF50', '#FF4B4B'],
                             hole=0.4) # Donut chart for professional look
                st.plotly_chart(fig, use_container_width=True)

elif word_count == 0:
    st.info("Enter job details in the sidebar and click **RUN FULL PIPELINE** to begin analysis.")
