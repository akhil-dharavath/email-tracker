# FusionMail: Intelligent Email Triaging Assistant

**FusionMail** is a web-based email client designed to reduce inbox anxiety and improve productivity. It uses advanced **Machine Learning (Multimodal Fusion)** to analyze the emotional tone and urgency of your emails in real-time, helping you prioritize what matters most.

![App Screenshot](https://via.placeholder.com/800x400?text=FusionMail+Interface+Preview)

## 🚀 Key Features

*   **🧠 Multimodal Emotion Detection**:
    *   Combines **Semantic Analysis** (DistilBERT) with **Behavioral Features** (punctuation, capitalization, email length).
    *   Detects 4 emotional states: **Angry**, **Anxious**, **Happy**, **Neutral**.
*   **🔥 Smart Urgency Scoring**:
    *   Calculates a dynamic Urgency Score (0-10) for every email.
    *   Prioritizes "Angry" or "Anxious" emails automatically.
*   **📝 AI Summarization**:
    *   Generates concise 1-sentence summaries for long emails using `t5-small`.
    *   Hover over any email to get the gist instantly.
*   **📧 Gmail Integration**:
    *   Seamlessly syncs with your real Gmail inbox using OAuth 2.0.
    *   Parses real timestamps and read/unread status.
*   **🎨 Dynamic UI**:
    *   Modern, "Glassmorphism" design with dark mode.
    *   Real-time search, filtering, and sorting.

## 🛠️ Technology Stack

*   **Frontend**: HTML5, CSS3 (Custom Glassmorphism), pure JavaScript (ES6+).
*   **Backend**: Flask (Python).
*   **Machine Learning**:
    *   **PyTorch**: Custom Fusion Model architecture.
    *   **Transformers (Hugging Face)**: DistilBERT & T5.
    *   **Scikit-Learn**: Behavioral feature scaling.

## 📦 Installation & Setup

### 1. Prerequisites
- Python 3.8+
- A Google Cloud Project with **Gmail API** enabled (for Sync).

### 2. Clone & Install
```bash
# Clone the repository
git clone <your-repo-url>
cd email-tracker

# Create Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Google OAuth Setup (Critical)
To fetch real emails, you need `credentials.json` from Google Cloud:
1.  Go to [Google Cloud Console](https://console.cloud.google.com/).
2.  Enable **Gmail API**.
3.  Configure **OAuth Consent Screen** (Add your email as a "Test User").
4.  Create **OAuth 2.0 Client IDs** (Desktop App).
5.  Download the JSON, rename it to `credentials.json`, and place it in the `backend/` folder.

### 4. Run the Application
```bash
# Start the Flask Backend
python backend/app.py
```
The application will launch at `http://127.0.0.1:9000`.

## 🖥️ Usage Guide

1.  **Sync Gmail**: Click the "Sync Gmail" button in the sidebar. A Google Auth window will open. Grant permissions to allow the app to read your emails.
2.  **View Analysis**: Emails will appear with colored dots:
    *   🔴 **Red**: Angry
    *   🟠 **Orange**: Anxious
    *   🟢 **Green**: Happy
    *   ⚪ **Gray**: Neutral
3.  **Sort by Urgency**: Toggle the "Sort by Urgency" switch to float high-priority emails to the top.
4.  **AI Summary**: Hover over any email card to read the AI-generated summary.

## 📂 Project Structure

```
├── backend/
│   ├── app.py              # Flask API Routes (/api/analyze, /api/sync)
│   ├── gmail_client.py     # Gmail OAuth & Parsing Logic
│   └── credentials.json    # (User config) Google API Credentials
├── frontend/
│   ├── index.html          # Main UI Structure
│   ├── style.css           # Styling
│   └── app.js              # Frontend Logic (Fetch, Render, Interactions)
├── ml/
│   ├── fusion_model.py     # PyTorch Model Architecture (DistilBERT + MLP)
│   ├── feature_extractor.py# Engineering Behavioral Features
│   ├── summarizer.py       # T5 Summarization Wrapper
│   └── train_pipeline.py   # Script to train/save the model
└── requirements.txt        # Python Dependencies
```

## 🚢 Deployment (Render/Heroku)
See [DEPLOY.md](DEPLOY.md) for instructions on deploying this application to Render.
**Note**: Since Render servers are headless, you must perform OAuth locally first and upload `token.json` as a secret file.

---
*Built with ❤️ by FusionMail Team*
