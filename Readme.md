
Enterprise Smart Email Classifier

An AI-powered customer support tool that automatically categorizes incoming emails and detects their urgency level. Designed to help support teams prioritize critical issues, filter spam, and route requests efficiently.
Live Demo

Click here to view the deployed app (https://email-class-by-yours-truly.streamlit.app/)
    Key Features

    Dual-Model Architecture:

        Categorization: Uses a Quantized DistilBERT (Transformer) model to classify emails into Complaint, Feedback, Request, or Spam.

        Urgency Detection: Uses a Support Vector Machine (SVM) pipeline with TF-IDF to flag emails as High, Medium, or Low priority.

    Auto-Stitching : automatically reassembling split model files during runtime 60mb + 70mb.

    Real-Time Dashboard: Interactive analytics powered by Plotly to visualize ticket volume, urgency distribution, and response metrics.

    Smart Preprocessing: Includes custom NLP cleaning pipelines (Lemmatization, Stop-word removal, Regex cleaning).

    Installation & Local Setup

Follow these steps to run the application on your local machine.
1. Clone the Repository
Bash

git clone https://github.com/vickyax/Email-Class-By-Yours-Truly.git
cd Email-Class-By-Yours-Truly

2. Create a Virtual Environment (Recommended)
Bash

# Windows
python -m venv env
.\env\Scripts\activate

# Mac/Linux
python3 -m venv env
source env/bin/activate

3. Install Dependencies


pip install -r requirements.txt

4. Run the Application

streamlit run app/app.py



   Project Structure
Plaintext

Email-Class-By-Yours-Truly/
│
├── app/               # Main Application Code
│   ├── app.py                    # Streamlit Frontend & Logic
│   │
│   ├── quantized_model/          # DistilBERT Model (Category)
│   │   ├── config.json
│   │   ├── quantized_bert.pth.part0  # Split file part 1
│   │   ├── quantized_bert.pth.part1  # Split file part 2
│   │   └── tokenizer.json
│   │
│   └── trained_priority_models/  # SVM/RF Model (Urgency)
│       ├── poly_svc_model.joblib
│       └── tfidf_vectorizer.joblib
│
├── requirements.txt              # Dependencies
└── README.md                     # Documentation

    Working
1. The "Auto-Stitcher"

The DistilBERT model is roughly 130MB, which exceeds GitHub's file size limit. We split the model into .part0 and .part1.

    On Deployment: When app.py loads, it checks if the full model exists.

    If Missing: It reads the parts and binary-merges them back into a single quantized_bert.pth file in the system's temporary memory.

2. Quantization

To ensure the app runs fast on CPU-only environments (like the free tier of Streamlit Cloud), the BERT model was Dynamic Quantized.

    Original Size: ~260 MB (FP32)

    Quantized Size: ~65 MB (INT8)

    Accuracy Loss: < 1%

3. Preprocessing Pipeline

Before any prediction, text undergoes:

    Lowercasing & Regex cleaning (removing specific priority keywords to prevent bias).

    Tokenization.

    Stop-word removal (NLTK).

    Lemmatization (converting words to their base root).

   Dependencies

The project relies on a specific "Sweet Spot" of library versions to prevent conflicts between transformers and huggingface-hub:

    streamlit==1.28.0

    transformers==4.35.2

    huggingface-hub==0.19.4

    torch==2.0.1

    scikit-learn==1.2.2 (Required for loading the urgency model)



🤝 Contributing

    Fork the repository.

    Create your feature branch (git checkout -b feature/AmazingFeature).

    Commit your changes (git commit -m 'Add some AmazingFeature').

    Push to the branch (git push origin feature/AmazingFeature).

    Open a Pull Request.

Author: [Vickyax] Built with ❤️ using Streamlit & Python