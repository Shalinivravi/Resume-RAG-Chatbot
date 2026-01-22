---
description: How to create a temporary public link for your chatbot
---

If you want to share your chatbot **right now** without using GitHub or Streamlit Cloud, you can use `localtunnel`.

1. **Start your Streamlit app** in one terminal:
   ```bash
   streamlit run app.py
   ```
   *(Note: By default it runs on port 8501)*

2. **Open a second terminal** and run:
   ```bash
   npx localtunnel --port 8501
   ```

3. **Get your link**:
   - The terminal will output a link like `https://your-app-name.loca.lt`.
   - Anyone with this link can access your chatbot while your computer is running!

> [!TIP]
> This link is temporary and will stop working if you close your terminal. For a permanent link, follow the GitHub + Streamlit Cloud instructions in the `walkthrough.md`.
