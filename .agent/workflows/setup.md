---
description: How to setup and run the Resume Screening Chatbot
---

Follow these steps to get your chatbot up and running:

1. **Install Python Dependencies**
// turbo
```bash
pip install -r requirements.txt
```

2. **Configure environment variables**
   - Copy `.env.template` to `.env`
   - Open `.env` and replace `your_google_api_key_here` with your actual Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

3. **Run the Streamlit application**
// turbo
```bash
streamlit run app.py
```

4. **Access the Web Interface**
   - Once the command starts, you will see a local URL (e.g., `http://localhost:8501`) in your terminal. 
   - Open this link in your browser to start using the chatbot.
