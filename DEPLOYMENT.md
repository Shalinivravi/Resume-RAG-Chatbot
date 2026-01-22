# Streamlit Cloud Deployment Guide

## 🚀 Quick Deploy

### Step 1: Go to Streamlit Cloud
Visit [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.

### Step 2: Deploy Your App
1. Click **"New app"**
2. Select repository: `Shalinivravi/Resume-RAG-Chatbot`
3. Branch: `main`
4. Main file path: `app.py`
5. Click **"Deploy"**

### Step 3: Add Your API Key
⚠️ **IMPORTANT**: Before the app works, you need to add your Google API Key:

1. In Streamlit Cloud dashboard, click your deployed app
2. Go to **Settings** (⚙️) → **Secrets**
3. Add this:
   ```toml
   GOOGLE_API_KEY = "your-google-api-key-here"
   ```
4. Click **Save**

### Step 4: Get Your Link
Your app will be live at: `https://your-app-name.streamlit.app`

---

## 📝 Notes
- The app auto-deploys whenever you push to GitHub
- Free tier includes 1GB resource limit
- API key is stored securely in Streamlit Cloud
