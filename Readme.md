#  AI-Powered Smart Email Classifier V 0.0.0.001

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-1.2.2-yellow)
![Status](https://img.shields.io/badge/Status-Deployed-success)

An intelligent customer support triage tool that automatically categorizes incoming emails and assigns urgency levels. Designed to help support teams reduce manual sorting time, prioritize critical issues, and improve response efficiency.

---

##  Live Demo
**[Click here to view the deployed app](https://email-class-by-yours-truly.streamlit.app/)**


---

##  Key Features

### 1. Automated Categorization
Classifies emails into four distinct business intent categories using a **Quantized DistilBERT** transformer model:
* 📝 **Complaint:** Issues, failures, or dissatisfaction.
* 💡 **Feedback:** Praise or suggestions.
* ❓ **Request:** Inquiries about pricing, demos, or information.
* 🗑️ **Spam:** Irrelevant or promotional content.

### 2. Urgency Detection
Determines the priority level of an email using a **Support Vector Machine (SVM)** pipeline with TF-IDF vectorization:
* 🔴 **High:** Critical issues requiring immediate attention (e.g., system failure).
* 🟠 **Medium:** Standard requests (e.g., pricing inquiries).
* 🟢 **Low:** General feedback or spam.

### 3. Operational Dashboard
A real-time analytics dashboard powered by **Plotly** to visualize:
* Ticket volume trends.
* Distribution of urgency levels.
* Category breakdowns.

### 4. Smart Preprocessing
Includes a custom NLP cleaning pipeline that handles:
* Lemmatization & Stop-word removal.
* Regex cleaning (removing timestamps, email headers).
* Bias removal (stripping specific "priority terms" during training to prevent overfitting).

---

##  Technical Architecture

### Model Compression & Handling
To adhere to GitHub's 100MB file limit and optimize for Cloud deployment, this project uses a custom architecture:
1.  **Dynamic Quantization:** The DistilBERT model was quantized from FP32 to INT8, reducing size from ~260MB to ~65MB.
2.  **File Splitting:** The model file is physically split into `.part0` and `.part1` to bypass Git constraints.
3.  **Auto-Stitcher:** On application startup, a custom script in `app.py` automatically detects the parts and reassembles them into a functional model in memory.

### Tech Stack
* **Frontend:** Streamlit
* **NLP/ML:** Hugging Face Transformers, PyTorch, Scikit-learn, NLTK
* **Visualization:** Plotly Express
* **Data Handling:** Pandas, Joblib

---

##  Installation & Local Setup

Follow these steps to run the application on your local machine.

 1. Clone the Repository
```
git clone [https://github.com/vickyax/Email-Class-By-Yours-Truly.git](https://github.com/vickyax/Email-Class-By-Yours-Truly.git)
cd Email-Class-By-Yours-Truly
```
2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.
Bash
```
# Windows
python -m venv env
.\env\Scripts\activate

# Mac/Linux
python3 -m venv env
source env/bin/activate
```
3. Install Dependencies

Crucial: This project requires specific library versions to ensure compatibility between transformers and huggingface-hub.
Bash
```
pip install -r project_folder/requirements.txt
```
4. Run the Application

Navigate to the root directory and run:
Bash
```
streamlit run project_folder/app.py
```
The first run make take a few seconds as the Auto-Stitcher reassembles the model.
Project Structure
```
Email-Class-By-Yours-Truly/
│
├── project_folder/               # Main Application Source Code
│   ├── app.py                    # Streamlit Frontend & Application Logic
│   ├── requirements.txt          # Python Dependencies
│   │
│   ├── quantized_model/          # DistilBERT Model (Category)
│   │   ├── config.json
│   │   ├── quantized_bert.pth.part0  # Split Model Part 1
│   │   ├── quantized_bert.pth.part1  # Split Model Part 2
│   │   ├── tokenizer.json
│   │   └── vocab.txt
│   │
│   └── trained_priority_models/  # SVM/RF Model (Urgency)
│       ├── poly_svc_model.joblib
│       └── tfidf_vectorizer.joblib
│
├── .gitignore                    # Git Ignore file (excludes env/)
└── README.md                     # Project Documentation

```

Author: *Vickyax* Built with ❤️ using Streamlit & Python
