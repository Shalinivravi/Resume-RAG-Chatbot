import streamlit as st
import os
import pandas as pd
from rag_engine import RAGEngine
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Resume Screening Chatbot", page_icon="📄", layout="wide")

# Custom CSS for a premium look
st.markdown("""
<style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');

    /* Global Reset & Base Styles */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(139, 92, 246, 0.15) 0%, transparent 40%),
                    radial-gradient(circle at 90% 80%, rgba(59, 130, 246, 0.15) 0%, transparent 40%),
                    #0f172a;
        font-family: 'Inter', sans-serif;
        color: #f8fafc;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Headings Gradient */
    h1 span, h2 span, .gradient-text {
        background: linear-gradient(135deg, #a78bfa 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Glassmorphism Cards */
    .glass-card, .stChatMessage {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.3);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%);
        border: none;
        border-radius: 12px;
        color: white;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }

    .stButton>button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);
    }
    
    /* Inputs */
    .stTextInput>div>div, .stTextArea>div>div {
        background-color: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
    }
    
    .stTextInput>div>div:focus-within, .stTextArea>div>div:focus-within {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2);
    }

    /* Sidebar */
    .stSidebar {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* Toast/Alerts */
    .stToast {
        background-color: rgba(30, 41, 59, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f8fafc !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="gradient-text">AI Resume Screening Chatbot</h1>', unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.2rem; opacity: 0.8;'>Intelligent candidate matching powered by RAG</p>", unsafe_allow_html=True)

# Check for API Key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("❌ **GOOGLE_API_KEY not found.** Please set it in your `.env` file to start.")
    st.info("You can get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).")
    st.stop()

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed" not in st.session_state:
    st.session_state.processed = False

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

# Sidebar for Uploads
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.header("Upload Control")
    uploaded_files = st.file_uploader("Drop PDF resumes here", type="pdf", accept_multiple_files=True)
    
    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("Index Files")
    with col2:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    if process_btn and uploaded_files:
        with st.spinner("Analyzing resumes..."):
            try:
                st.session_state.rag_engine.process_resumes(uploaded_files)
                st.session_state.processed = True
                st.session_state.indexed_files = [f.name for f in uploaded_files]
                st.toast("✅ Resumes indexed successfully!", icon="🚀")
                st.success(f"Indexed {len(uploaded_files)} resumes!")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if st.button("🔄 Reset App", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    if st.session_state.indexed_files:
        st.markdown("---")
        st.markdown("### 📚 Indexed Resumes")
        for file in st.session_state.indexed_files:
            st.caption(f"✅ {file}")

    st.markdown("---")
    with st.expander("🛠️ How it Works"):
        st.markdown("""
        <div class="glass-card" style="padding: 1.5rem;">
            <ol style="list-style-position: inside; margin: 0; padding: 0;">
                <li style="margin-bottom: 0.8rem;"><strong>Extraction</strong>: Text is pulled from PDF resumes.</li>
                <li style="margin-bottom: 0.8rem;"><strong>Chunking</strong>: Resumes are split into small, searchable segments.</li>
                <li style="margin-bottom: 0.8rem;"><strong>Embeddings</strong>: Google Gemini converts text into numerical vectors.</li>
                <li style="margin-bottom: 0.8rem;"><strong>Retrieval</strong>: Relevant context is found using FAISS.</li>
                <li><strong>Generation</strong>: Gemini generates a human-like answer.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("🚀 Built with LangChain & Google Gemini")
st.sidebar.caption("Created by [Antigravity AI](https://github.com/google-deepmind)")

# Main Interface with Tabs
tab1, tab2, tab3 = st.tabs(["💬 Candidate Chat", "🎯 JD Matching", "📊 Resume Dashboard"])

with tab1:
    if st.session_state.processed:
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Ask about candidates (e.g., 'Who knows Python and AWS?')"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            
            # Prepare history for LangChain
            history = [(m["role"], m["content"]) for m in st.session_state.messages]
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_engine.get_response(prompt, chat_history=history)
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("👋 **Welcome!** Please upload and index resumes in the sidebar to start chatting.")
        st.markdown("""
        ### Quick Start:
        1. **Upload**: Drag and drop PDF resumes into the sidebar.
        2. **Index**: Click the 'Index Files' button.
        3. **Chat**: Ask questions here about the candidates found.
        4. **Rank**: Switch to the 'JD Matching' tab to rank candidates against a job description.
        """)

with tab2:
    st.header("🎯 Match Candidates to Job Description")
    st.markdown("Paste a Job Description below to see which candidates fit best.")
    
    jd_input = st.text_area("Paste Job Description here...", height=200)
    
    if st.button("Rank Candidates"):
        if not st.session_state.processed:
            st.warning("Please upload and index resumes first!")
        elif not jd_input:
            st.warning("Please paste a Job Description.")
        else:
            with st.spinner("Analyzing and ranking matches..."):
                try:
                    markdown_res, structured_data = st.session_state.rag_engine.rank_candidates(jd_input)
                    st.markdown("### 🏆 Ranking Results")
                    with st.container(border=True):
                        st.markdown(markdown_res)
                    
                    if structured_data:
                        df = pd.DataFrame(structured_data)
                        st.markdown("### 📈 Summary Table")
                        st.dataframe(df, use_container_width=True)
                        
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Ranking as CSV",
                            data=csv,
                            file_name='candidate_ranking.csv',
                            mime='text/csv',
                        )
                    st.toast("Ranking complete!", icon="🎯")
                except Exception as e:
                    st.error(f"Error during ranking: {str(e)}")

with tab3:
    st.header("📊 Multi-Resume Dashboard")
    if st.session_state.processed:
        if st.button("Generate Talent Overview"):
            with st.spinner("Crunching data..."):
                summary = st.session_state.rag_engine.summarize_resumes()
                st.info(summary)
        
        st.markdown("---")
        st.markdown("### 📋 Uploaded Files")
        cols = st.columns(3)
        for i, file in enumerate(st.session_state.indexed_files):
            cols[i % 3].success(f"📄 {file}")
    else:
        st.info("Upload resumes to see a dashboard analysis.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built with LangChain, FAISS, and Streamlit")
