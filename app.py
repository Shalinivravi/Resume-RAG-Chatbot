import streamlit as st
import os
from rag_engine import rag_engine
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Resume Screening Chatbot", page_icon="📄", layout="wide")

# Custom CSS for a premium look
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .main {
        padding: 2rem;
    }
    .stHeader {
        color: #1e3a8a;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        border-radius: 8px;
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: white;
        border-right: 1px solid #e5e7eb;
    }
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 AI Resume Screening Chatbot")
st.markdown("#### Intelligent candidate matching powered by RAG")

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
                rag_engine.process_resumes(uploaded_files)
                st.session_state.processed = True
                st.session_state.indexed_files = [f.name for f in uploaded_files]
                st.success(f"Indexed {len(uploaded_files)} resumes!")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if st.session_state.indexed_files:
        st.markdown("---")
        st.markdown("### 📚 Indexed Resumes")
        for file in st.session_state.indexed_files:
            st.caption(f"✅ {file}")

    st.markdown("---")
    with st.expander("🛠️ How it Works"):
        st.write("""
        1. **Extraction**: Text is pulled from PDF resumes.
        2. **Chunking**: Resumes are split into small, searchable segments.
        3. **Embeddings**: Google Gemini converts text into numerical vectors.
        4. **Retrieval**: When you ask a question, the most relevant parts are found using FAISS.
        5. **Generation**: Gemini generates a human-like answer based *only* on the retrieved context.
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("🚀 Built with LangChain & Google Gemini")
st.sidebar.caption("Created by [Antigravity AI](https://github.com/google-deepmind)")

# Main Interface with Tabs
tab1, tab2 = st.tabs(["💬 Candidate Chat", "🎯 JD Matching"])

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
                    response = rag_engine.get_response(prompt, chat_history=history)
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("👋 Please upload and index resumes in the sidebar to start chatting.")

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
                    ranking_results = rag_engine.rank_candidates(jd_input)
                    st.markdown("### 🏆 Ranking Results")
                    st.markdown(ranking_results)
                except Exception as e:
                    st.error(f"Error during ranking: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built with LangChain, FAISS, and Streamlit")
