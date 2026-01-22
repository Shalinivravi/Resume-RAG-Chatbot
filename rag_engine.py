import os
import tempfile
import json
import re
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class RAGEngine:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        self.vector_store = None

    def process_resumes(self, uploaded_files):
        """
        Extract text from uploaded PDFs, chunk them, and store in FAISS.
        """
        all_documents = []
        
        for uploaded_file in uploaded_files:
            # Save the uploaded file to a temporary file because PyPDFLoader needs a path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                # Add metadata to each document to track its source
                for doc in documents:
                    doc.metadata["source"] = uploaded_file.name
                all_documents.extend(documents)
            finally:
                os.remove(tmp_path)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(all_documents)

        # Create vector store
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        return self.vector_store

    def get_response(self, query: str, chat_history: List = None):
        """
        Retrieve relevant docs and generate a response, considering chat history.
        """
        if not self.vector_store:
            return "Please upload some resumes first."

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        # Define prompts for contextualizing the query
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        # Define prompts for the answer generation
        system_prompt = (
            "You are an expert recruitment assistant. Use the following pieces of retrieved context "
            "to answer the question. If you don't know the answer based on the resumes, "
            "just say that you don't know. Do not hallucinate or make up candidate details. "
            "Always mention the candidate name if available in the source.\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        response = rag_chain.invoke({"input": query, "chat_history": chat_history or []})
        return response["answer"]

    def rank_candidates(self, job_description: str):
        """
        Rank all indexed candidates against a job description.
        Returns both a markdown response and a list of candidate scores for export.
        """
        if not self.vector_store:
            return "Please upload some resumes first.", []

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})
        
        system_prompt = (
            "You are an expert HR Analyst. You will receive a Job Description and pieces of retrieved context from several resumes. "
            "Your task is to rank the candidates based on how well they match the Job Description.\n\n"
            "Analyze the resumes and provide a ranked list of candidates.\n"
            "For each candidate, provide:\n"
            "- Name\n"
            "- Match Score (0-100)\n"
            "- Reasoning (bullet points)\n\n"
            "After the list, provide a JSON-formatted list of candidates with keys: 'name', 'score', 'reasoning_summary'. "
            "Example JSON: [{\"name\": \"John Doe\", \"score\": 85, \"reasoning_summary\": \"Strong Python skills\"}]\n\n"
            "Job Description: {jd}\n\n"
            "Retrieved Resume Context:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
        chain = create_stuff_documents_chain(self.llm, prompt)
        
        docs = retriever.invoke(job_description)
        response = chain.invoke({"context": docs, "jd": job_description})
        
        # Simple extraction of markdown vs JSON part
        markdown_part = response
        structured_data = []
        
        json_match = re.search(r'\[\s*{.*}\s*\]', response, re.DOTALL)
        if json_match:
            try:
                structured_data = json.loads(json_match.group())
                markdown_part = response[:json_match.start()].strip()
            except:
                pass
                
        return markdown_part, structured_data

    def summarize_resumes(self):
        """
        Generate a quick summary of all indexed candidates.
        """
        if not self.vector_store:
            return "No resumes indexed."

        # Retrieve chunks with broad search to get a sense of everything
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 15})
        docs = retriever.invoke("Summarize all candidates and their key skills.")
        
        system_prompt = (
            "Provide a high-level summary of all candidates found in the provided resume snippets. "
            "Group them by their primary expertise. Be concise."
            "\n\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
        chain = create_stuff_documents_chain(self.llm, prompt)
        
        return chain.invoke({"context": docs})

# Note: No singleton instance here. Instantiate in app.py within session_state.
