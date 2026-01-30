import streamlit as st
import time

# Page Configuration
st.set_page_config(
    page_title="Resume RAG Chatbot - Demo",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for Premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
    }
    
    .demo-badge {
        background: #ffd700;
        color: #333;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin-left: 10px;
    }
    
    .feature-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📄 AI Resume Screening Chatbot")
st.markdown('<span class="demo-badge">🎬 DEMO MODE</span>', unsafe_allow_html=True)
st.markdown("### 🚀 Intelligent Candidate Matching powered by RAG Technology")
st.markdown("---")

# Project Description
with st.expander("📖 About This Project", expanded=False):
    st.markdown("""
    **Resume RAG Chatbot** is an AI-powered recruitment assistant that helps HR teams and recruiters efficiently screen and match candidates.
    
    ### ✨ Key Features:
    - 📤 **Bulk Resume Upload**: Process multiple PDF resumes simultaneously
    - 🤖 **AI-Powered Search**: Ask natural language questions about candidates
    - 🎯 **Job Matching**: Automatically rank candidates based on job descriptions
    - 💬 **Conversational Interface**: Chat-based interaction for easy querying
    - 🧠 **RAG Technology**: Retrieval-Augmented Generation for accurate, context-aware responses
    
    ### 🛠️ Technology Stack:
    - **LangChain**: For building the RAG pipeline
    - **FAISS**: Vector database for similarity search
    - **Google Gemini**: Advanced language model for embeddings and generation
    - **Streamlit**: Interactive web interface
    """)

# Initialize session state
if "demo_processed" not in st.session_state:
    st.session_state.demo_processed = False

if "demo_messages" not in st.session_state:
    st.session_state.demo_messages = []

if "uploaded_count" not in st.session_state:
    st.session_state.uploaded_count = 0

# Sidebar
with st.sidebar:
    st.markdown("### 📁 Upload Resumes")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple resume PDFs to analyze"
    )
    
    if uploaded_files:
        st.session_state.uploaded_count = len(uploaded_files)
        st.success(f"✅ {len(uploaded_files)} file(s) selected")
        
        with st.expander("📋 Selected Files"):
            for file in uploaded_files:
                st.caption(f"📄 {file.name}")
    
    # Process button
    if st.button("🔄 Process Resumes", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing resumes..."):
                # Simulate processing
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                st.session_state.demo_processed = True
                st.balloons()
                st.success(f"🎉 Successfully processed {len(uploaded_files)} resumes!")
        else:
            st.warning("⚠️ Please upload resumes first!")
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.demo_messages = []
        st.rerun()
    
    # Reset button
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.demo_processed = False
        st.session_state.demo_messages = []
        st.session_state.uploaded_count = 0
        st.rerun()
    
    st.markdown("---")
    
    # Demo info
    st.markdown("### ℹ️ Demo Information")
    st.info("This is a **UI demonstration** with simulated responses. No actual AI processing occurs.")
    
    st.markdown("---")
    st.caption("🚀 Built with LangChain & Streamlit")
    st.caption("Powered by Google Gemini AI")

# Main Content Area - Tabs
tab1, tab2, tab3 = st.tabs(["💬 Candidate Chat", "🎯 JD Matching", "📊 Analytics"])

# Tab 1: Candidate Chat
with tab1:
    if st.session_state.demo_processed:
        st.markdown("### 💬 Ask Questions About Candidates")
        st.caption("Try asking: 'Who has Python experience?' or 'Find candidates with 5+ years experience'")
        
        # Display chat history
        for message in st.session_state.demo_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about candidates (e.g., 'Who knows Python and AWS?')"):
            # Display user message
            st.chat_message("user").markdown(prompt)
            st.session_state.demo_messages.append({"role": "user", "content": prompt})
            
            # Generate demo response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing resumes..."):
                    time.sleep(1)  # Simulate thinking
                    
                    # Demo responses based on common queries
                    demo_response = generate_demo_response(prompt)
                    st.markdown(demo_response)
            
            st.session_state.demo_messages.append({"role": "assistant", "content": demo_response})
    else:
        # Welcome screen
        st.info("👋 **Welcome to Resume RAG Chatbot!**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.markdown("#### 🎯 How It Works")
            st.markdown("""
            1. **Upload** PDF resumes via sidebar
            2. **Process** resumes with AI
            3. **Ask** natural language questions
            4. **Get** intelligent answers instantly
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.markdown("#### 💡 Sample Questions")
            st.markdown("""
            - "Who has Python experience?"
            - "Find ML engineers with 5+ years"
            - "Candidates with cloud certifications?"
            - "Who worked at FAANG companies?"
            """)
            st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: JD Matching
with tab2:
    st.markdown("### 🎯 Job Description Matching")
    st.caption("Paste a job description to rank candidates by relevance")
    
    jd_text = st.text_area(
        "Job Description",
        placeholder="Paste the job description here...\n\nExample:\nLooking for a Senior Python Developer with 5+ years experience in ML/AI, proficient in TensorFlow, PyTorch, and cloud platforms (AWS/GCP).",
        height=200
    )
    
    if st.button("🔍 Rank Candidates", use_container_width=False):
        if not st.session_state.demo_processed:
            st.warning("⚠️ Please upload and process resumes first!")
        elif not jd_text:
            st.warning("⚠️ Please enter a job description!")
        else:
            with st.spinner("Analyzing and ranking candidates..."):
                time.sleep(1.5)
                
                st.success("✅ Ranking complete!")
                
                st.markdown("### 🏆 Top Matches")
                
                # Demo ranking results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🥇 John Smith", "95%", "Top Match")
                    st.caption("✅ Python, ML, AWS, 8 years exp")
                
                with col2:
                    st.metric("🥈 Sarah Johnson", "87%", "Strong Fit")
                    st.caption("✅ Python, TensorFlow, 6 years exp")
                
                with col3:
                    st.metric("🥉 Michael Chen", "79%", "Good Fit")
                    st.caption("✅ ML, PyTorch, 5 years exp")
                
                st.markdown("---")
                
                # Detailed ranking table
                st.markdown("### 📊 Detailed Ranking")
                import pandas as pd
                
                ranking_data = {
                    "Rank": [1, 2, 3, 4, 5],
                    "Candidate": ["John Smith", "Sarah Johnson", "Michael Chen", "Emily Davis", "Robert Wilson"],
                    "Match Score": ["95%", "87%", "79%", "72%", "68%"],
                    "Key Skills": [
                        "Python, ML, AWS, TensorFlow",
                        "Python, TensorFlow, Deep Learning",
                        "ML, PyTorch, Computer Vision",
                        "Python, Data Science, GCP",
                        "Python, NLP, Azure"
                    ],
                    "Experience": ["8 years", "6 years", "5 years", "4 years", "6 years"]
                }
                
                df = pd.DataFrame(ranking_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                st.download_button(
                    "📥 Download Ranking (CSV)",
                    df.to_csv(index=False),
                    "candidate_ranking.csv",
                    "text/csv"
                )

# Tab 3: Analytics Dashboard
with tab3:
    st.markdown("### 📊 Resume Analytics Dashboard")
    
    if st.session_state.demo_processed:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Resumes", st.session_state.uploaded_count or "12", "+3")
        
        with col2:
            st.metric("Avg Experience", "5.2 years", "↑1.2")
        
        with col3:
            st.metric("Top Skill", "Python", "85%")
        
        with col4:
            st.metric("Queries Today", "24", "+12")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔥 Top Skills Found")
            skills_data = {
                "Skill": ["Python", "JavaScript", "AWS", "Machine Learning", "Docker"],
                "Count": [10, 8, 7, 6, 5]
            }
            st.bar_chart(skills_data, x="Skill", y="Count")
        
        with col2:
            st.markdown("#### 📈 Experience Distribution")
            exp_data = {
                "Experience Level": ["1-3 years", "3-5 years", "5-8 years", "8+ years"],
                "Candidates": [3, 4, 3, 2]
            }
            st.bar_chart(exp_data, x="Experience Level", y="Candidates")
        
        st.markdown("---")
        st.markdown("#### 🎓 Education Levels")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("🎓 Bachelor's: 7")
        with col2:
            st.success("🎓 Master's: 4")
        with col3:
            st.warning("🎓 PhD: 1")
        
    else:
        st.info("📊 Upload and process resumes to see analytics")
        st.image("https://cdn-icons-png.flaticon.com/512/2329/2329142.png", width=200)

# Helper function for demo responses
def generate_demo_response(query):
    """Generate contextual demo responses based on the query"""
    query_lower = query.lower()
    
    if "python" in query_lower:
        return """
📌 **Found 3 candidates with Python experience:**

1. **John Smith** - 8 years experience
   - Expert in Python, Django, Flask
   - ML/AI projects with TensorFlow
   - Previously at Google

2. **Sarah Johnson** - 6 years experience  
   - Full-stack Python developer
   - Experience with FastAPI, pandas
   - Data science background

3. **Michael Chen** - 5 years experience
   - Python for ML/Computer Vision
   - PyTorch expert
   - Published researcher

*Note: This is a demo response. Actual implementation would retrieve from indexed resumes.*
"""
    
    elif "experience" in query_lower or "years" in query_lower:
        return """
📌 **Candidates with 5+ years experience:**

1. **John Smith** - 8 years (Senior Software Engineer)
2. **Sarah Johnson** - 6 years (Full Stack Developer)  
3. **Robert Wilson** - 6 years (NLP Specialist)
4. **Michael Chen** - 5 years (ML Engineer)

**Average Experience:** 6.25 years

*Note: This is a demo response. Actual implementation would retrieve from indexed resumes.*
"""
    
    elif "aws" in query_lower or "cloud" in query_lower:
        return """
📌 **Found 2 candidates with AWS/Cloud experience:**

1. **John Smith**
   - AWS Solutions Architect Certified
   - 5+ years with EC2, S3, Lambda
   - Deployed production ML pipelines on AWS

2. **Emily Davis**
   - Google Cloud Platform experience
   - 3 years with GCP, BigQuery
   - Cloud infrastructure automation

*Note: This is a demo response. Actual implementation would retrieve from indexed resumes.*
"""
    
    elif "machine learning" in query_lower or "ml" in query_lower or "ai" in query_lower:
        return """
📌 **Found 4 candidates with ML/AI expertise:**

1. **John Smith** - TensorFlow, Keras, 8 years
2. **Sarah Johnson** - Deep Learning, NLP, 6 years
3. **Michael Chen** - Computer Vision, PyTorch, 5 years
4. **Robert Wilson** - NLP, transformers, 6 years

All candidates have hands-on ML project experience and relevant publications.

*Note: This is a demo response. Actual implementation would retrieve from indexed resumes.*
"""
    
    else:
        return f"""
📌 **Analysis complete for:** "{query}"

Based on the processed resumes, here are relevant matches:

1. **John Smith** - Highly relevant
   - 8 years experience
   - Strong technical background
   - Matches multiple criteria

2. **Sarah Johnson** - Good match
   - 6 years experience  
   - Versatile skill set
   - Team leadership experience

3. **Michael Chen** - Potential fit
   - 5 years experience
   - Specialized expertise
   - Strong academic background

*Note: This is a demo response showing the UI. Actual implementation would use RAG to retrieve real candidate information.*
"""

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Resume RAG Chatbot</strong> - AI-Powered Recruitment Assistant</p>
    <p>🚀 Demo UI | Built with Streamlit, LangChain & Google Gemini</p>
</div>
""", unsafe_allow_html=True)
