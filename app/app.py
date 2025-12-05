import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import time
import re
import torch
import nltk
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Smart Email Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    .urgent-box {
        padding: 20px; background-color: #ffe6e6; border-left: 5px solid #ff4b4b;
        border-radius: 5px; margin-bottom: 20px; color: #b30000;
    }
    .safe-box {
        padding: 20px; background-color: #e6fffa; border-left: 5px solid #00cc96;
        border-radius: 5px; color: #004d38;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 0. SETUP & PREPROCESSING
# ==========================================

# Download NLTK resources (cached)
@st.cache_resource
def download_nltk_resources():
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')

download_nltk_resources()

# Initialize Preprocessing Tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
PRIORITY_TERMS = r'\b(dear|ni|hello|hi|hope|well|escapelong|escapenumber|team|nan|data)\b'

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[\r\n\t\a\b]+', ' ', text)       # Remove escape sequences
    text = re.sub(PRIORITY_TERMS, '', text)          # Remove priority terms
    text = re.sub(r'[^a-z0-9\s]', ' ', text)         # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()         # Remove extra spaces
    
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 1
    ]
    return ' '.join(tokens)


# ==========================================
# 2. MODEL LOADING
# ==========================================
@st.cache_resource
def load_models():
    models = {}
    try:
        # ==========================================
        # 0. AUTO-STITCHER (Fix for GitHub 100MB limit)
        # ==========================================
        import os
        model_path = "./app/quantized_model"
        weights_file = f"{model_path}/quantized_bert.pth"
        
        # If the big file is missing, rebuild it from parts
        if not os.path.exists(weights_file):
            print("🧩 Stitching model parts together...")
            with open(weights_file, 'wb') as output:
                part_num = 0
                while True:
                    part_file = f"{weights_file}.part{part_num}"
                    if not os.path.exists(part_file):
                        break
                    with open(part_file, 'rb') as input_part:
                        output.write(input_part.read())
                    part_num += 1
            print("✅ Model rebuilt successfully!")

        # ==========================================
        # 1. LOAD MODEL (DistilBERT)
        # ==========================================
        from transformers import AutoConfig, AutoTokenizer, DistilBertForSequenceClassification
        import torch

        # Load Config & Tokenizer
        config = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Create Skeleton
        bert_model = DistilBertForSequenceClassification(config)
        
        # Apply Quantization Structure
        torch.quantization.quantize_dynamic(
            bert_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
        )
        
        # Load Weights
        bert_model.load_state_dict(torch.load(weights_file, map_location="cpu"))
        bert_model.eval()
        
        models['bert_tokenizer'] = tokenizer
        models['bert_model'] = bert_model

        # ==========================================
        # 2. LOAD MODEL (Urgency)
        # ==========================================
        svc_model = joblib.load('app/trained_priority_models/poly_svc_model.joblib')
        tfidf_vectorizer = joblib.load('app/trained_priority_models/tfidf_vectorizer.joblib')
        
        models['rf_model'] = svc_model 
        models['tfidf'] = tfidf_vectorizer
        models['device'] = torch.device("cpu")
        
        return models

    except Exception as e:
        st.error(f"Model Loading Failed: {e}")
        return None

# Load the models once
loaded_artifacts = load_models()

# ==========================================
# 3. INFERENCE LOGIC
# ==========================================
def predict_email(text, artifacts,prd=0):
    """
    Orchestrates the prediction using loaded models.
    Fallback to rules if models are None.
    """
    clean_content = clean_text(text)
    
    # --- MOCK MODE (Fallback) ---
    if artifacts is None:
        time.sleep(0.5) # Simulate compute
        # Simple keywords for demo if models fail to load
        if "urgent" in clean_content or "immediately" in clean_content: urg = "High"
        else: urg = "Low"
        
        if "refund" in clean_content: cat = "Request"
        elif "bad" in clean_content: cat = "Complaint"
        else: cat = "Feedback"
        return cat, urg, 0.0

    # --- REAL INFERENCE ---
    try:
        # 1. Predict Category (DistilBERT)
        if prd==1:
            device = artifacts['device']
            inputs = artifacts['bert_tokenizer'](clean_content, return_tensors="pt", truncation=True, padding=True).to(device)
            
            with torch.no_grad():
                outputs = artifacts['bert_model'](**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()
        
        # Mapping from id2label (Make sure this matches your training!)
        # Adjust these labels based on your specific training label2id
            id2label = {0: "Complaint", 1: "Feedback", 2: "Request", 3: "Spam"} 
        # Fallback if your model has different IDs, try to grab from config if possible, 
        # otherwise rely on the hardcoded map above.
            if hasattr(artifacts['bert_model'].config, 'id2label') and artifacts['bert_model'].config.id2label:
                category = artifacts['bert_model'].config.id2label[pred_idx]
            else:
                category = id2label.get(pred_idx, "Unknown")
            return category, confidence
        elif prd==2:
            # 2. Predict Urgency (svc + TFIDF)
            tfidf_text = artifacts['tfidf'].transform([clean_content])
            urgency_pred_idx = artifacts['rf_model'].predict(tfidf_text)[0]
            
            # Reverse mapping from training logic
            # priority_map = {'low': 1, 'medium': 2, 'high': 3}
            reverse_priority_map = {1: "Low", 2: "Medium", 3: "High"}
            urgency = reverse_priority_map.get(urgency_pred_idx, "Medium")

            return urgency
        else:
            device = artifacts['device']
            inputs = artifacts['bert_tokenizer'](clean_content, return_tensors="pt", truncation=True, padding=True).to(device)
            
            with torch.no_grad():
                outputs = artifacts['bert_model'](**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()
        
        # Mapping from id2label (Make sure this matches your training!)
        # Adjust these labels based on your specific training label2id
            id2label = {0: "Complaint", 1: "Feedback", 2: "Request", 3: "Spam"} 
        # Fallback if your model has different IDs, try to grab from config if possible, 
        # otherwise rely on the hardcoded map above.
            if hasattr(artifacts['bert_model'].config, 'id2label') and artifacts['bert_model'].config.id2label:
                category = artifacts['bert_model'].config.id2label[pred_idx]
            else:
                category = id2label.get(pred_idx, "Unknown")
            # 2. Predict Urgency (svc + TFIDF)
            tfidf_text = artifacts['tfidf'].transform([clean_content])
            urgency_pred_idx = artifacts['rf_model'].predict(tfidf_text)[0]
            
            # Reverse mapping from training logic
            # priority_map = {'low': 1, 'medium': 2, 'high': 3}
            reverse_priority_map = {1: "Low", 2: "Medium", 3: "High"}
            urgency = reverse_priority_map.get(urgency_pred_idx, "Medium")

            return urgency,category, confidence

    except Exception as e:
        st.error(f"Inference Error: {e}")
        return "Error", "Error", 0.0

# ==========================================
# 4. SESSION STATE
# ==========================================
if 'email_history' not in st.session_state:
    st.session_state['email_history'] = [
        {"timestamp": "09:00", "text": "System down, need help", "category": "Complaint", "urgency": "High"},
        {"timestamp": "09:15", "text": "Quote for new project", "category": "Request", "urgency": "Medium"},
        {"timestamp": "09:30", "text": "Great service thanks", "category": "Feedback", "urgency": "Low"},
    ]

# ==========================================
# 5. UI LAYOUT
# ==========================================
st.sidebar.title("Smart Classifier")

if loaded_artifacts is None:
    st.sidebar.warning("⚠️ MOCK MODE ACTIVE\n(Models not found in path)")
else:
    st.sidebar.success(f"✅ Models Loaded\nDevice: {loaded_artifacts['device']}")

page = st.sidebar.radio("Navigate", ["Urgency Level","Email Classification","Class & Urgency", "Analytics Dashboard"])
st.sidebar.markdown("---")
confidence_threshold = 0.80

# --- PAGE 1:  ---
if page == "Urgency Level":
    st.title("Customer Support")
    st.markdown("Paste incoming email content below to automatically  prioritize.")
    
    col_input, col_result = st.columns([2, 1])
    
    with col_input:
        email_text = st.text_area("Email Content", height=300, placeholder="Paste email text here...")
        analyze_btn = st.button("Analyze Email", type="primary", use_container_width=True)

    if analyze_btn and email_text:
        urg = predict_email(email_text, loaded_artifacts,2)
        
        # Update History
        st.session_state['email_history'].append({
            "timestamp": datetime.now().strftime("%H:%M"),
            "text": email_text[:50] + "...",
            "category": "",
            "urgency": urg
        })
        
        with col_result:
            st.subheader("Analysis Results")
       

            # Urgency Card
            u_colors = {"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#00cc96"}
            u_bg = u_colors.get(urg, "#ddd")
            
            st.markdown(f"""
            <div style="padding:15px; border:2px solid {u_bg}; border-radius:10px; text-align:center; background: white; color: black;">
                <h4 style="margin:0; color:{u_bg};">Urgency Level</h4>
                <h2 style="margin:0; color:{u_bg};">{urg}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
       
            
            # Action Banner
            if urg == "High":
                st.markdown('<div class="urgent-box">⚡ <b>CRITICAL:</b> Escalate to Tier 2 Support</div>', unsafe_allow_html=True)
        
if page == "Email Classification":
    st.title("Customer Support Email Categorization")
    st.markdown("Paste incoming email content below to automatically categorize and prioritize.")
    
    col_input, col_result = st.columns([2, 1])
    
    with col_input:
        email_text = st.text_area("Email Content", height=300, placeholder="Paste email text here...")
        analyze_btn = st.button("Analyze Email", type="primary", use_container_width=True)

    if analyze_btn and email_text:
        cat,  conf = predict_email(email_text, loaded_artifacts,1)
        
        # Update History
        st.session_state['email_history'].append({
            "timestamp": datetime.now().strftime("%H:%M"),
            "text": email_text[:50] + "...",
            "category": cat,
            "urgency": ""
        })
        
        with col_result:
            st.subheader("Analysis Results")
            
            # Category Card
            st.markdown(f"""
            <div style="padding:15px; border:1px solid #ffa500; border-radius:10px; text-align:center; background: white; color: black;">
                <h4 style="margin:0; color:#555;">Detected Category</h4>
                <h2 style="margin:0; color:#000;">{cat}</h2>
            </div>
            """, unsafe_allow_html=True)

            
           
            
            st.divider()
            
            # Confidence Logic
            if conf > 0:
                st.metric("Model Confidence", f"{conf*100:.1f}%")
                if conf < confidence_threshold:
                    st.caption("⚠️ Confidence below threshold. Human review required.")
            
            # Action Banner
            if cat == "Spam":
                st.markdown('<div class="safe-box">🗑️ <b>SPAM:</b> Move to Junk Folder</div>', unsafe_allow_html=True)

if page == "Class & Urgency":
    st.title("Customer Class & Urgency")
    st.markdown("Paste incoming email content below to automatically categorize and prioritize.")
    
    col_input, col_result = st.columns([2, 1])
    
    with col_input:
        email_text = st.text_area("Email Content", height=300, placeholder="Paste email text here...")
        analyze_btn = st.button("Analyze Email", type="primary", use_container_width=True)

    if analyze_btn and email_text:
        urg,cat, conf = predict_email(email_text, loaded_artifacts)
        
        # Update History
        st.session_state['email_history'].append({
            "timestamp": datetime.now().strftime("%H:%M"),
            "text": email_text[:50] + "...",
            "category": cat,
            "urgency": urg
        })
        
        with col_result:
            st.subheader("Analysis Results")
            
            # Category Card
            st.markdown(f"""
            <div style="padding:15px; border:1px solid #ddd; border-radius:10px; text-align:center; background: white; color: black;">
                <h4 style="margin:0; color:#555;">Detected Category</h4>
                <h2 style="margin:0;">{cat}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("") # Spacer

            # Urgency Card
            u_colors = {"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#00cc96"}
            u_bg = u_colors.get(urg, "#ddd")
            
            st.markdown(f"""
            <div style="padding:15px; border:2px solid {u_bg}; border-radius:10px; text-align:center; background: white; color: black;">
                <h4 style="margin:0; color:{u_bg};">Urgency Level</h4>
                <h2 style="margin:0; color:{u_bg};">{urg}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # Confidence Logic
            if conf > 0:
                st.metric("Model Confidence", f"{conf*100:.1f}%")
                if conf < confidence_threshold:
                    st.caption("⚠️ Confidence below threshold. Human review required.")
            
            # Action Banner
            if urg == "High":
                st.markdown('<div class="urgent-box">⚡ <b>CRITICAL:</b> Escalate to Tier 2 Support</div>', unsafe_allow_html=True)
            elif cat == "Spam":
                st.markdown('<div class="safe-box">🗑️ <b>SPAM:</b> Move to Junk Folder</div>', unsafe_allow_html=True)

# --- PAGE 2: DASHBOARD ---
elif page == "Analytics Dashboard":
    st.title("Operational Dashboard")
    
    df = pd.DataFrame(st.session_state['email_history'])
    
    # Top Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Emails", len(df))
    m2.metric("High Priority", len(df[df['urgency'] == 'High']))
    m3.metric("Complaints", len(df[df['category'] == 'Complaint']))
    m4.metric("Avg Response Time", "2.4 hrs") 
    
    st.divider()
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Category Distribution")
        if not df.empty:
            fig = px.pie(df, names='category', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
            
    with c2:
        st.subheader("Urgency vs Volume")
        if not df.empty:
            fig2 = px.bar(df, x='urgency', color='category', title="Urgency Breakdown")
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Recent Logs")
    st.dataframe(df.iloc[::-1], use_container_width=True)