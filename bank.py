import os
import shutil
from dotenv import load_dotenv

import streamlit as st

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI

# -----------------------------------
# LOAD ENV VARIABLES
# -----------------------------------
load_dotenv()

# -----------------------------------
# STREAMLIT CONFIG (CLEAN UI)
# -----------------------------------
st.set_page_config(
    page_title="Knowledge Base Chatbot",
    page_icon="💬",
    layout="wide"
)

# -----------------------------------
# ONLY DESCRIPTION (NO EXTRA TITLE)
# -----------------------------------
st.markdown("""
# 💬 Knowledge Base Chatbot

This is a smart **knowledge base chatbot** that allows you to ask questions about your uploaded documents, such as bank statements, financial records, or any PDF files.

It uses **OpenAI GPT-4.1-mini** and **vector search (FAISS)** to find relevant information from your documents and generate accurate answers based only on the provided context.

To use this app, place your PDF files inside the `docs/` folder and ask questions below.
""")

# -----------------------------------
# PATHS
# -----------------------------------
docs_folder = "docs"
index_path = "bank_faiss_index"

# -----------------------------------
# RESET VECTOR DB
# -----------------------------------
if st.button("🔄 Reset Knowledge Base"):
    if os.path.exists(index_path):
        shutil.rmtree(index_path)

    st.cache_resource.clear()
    st.success("Knowledge base cleared. Please refresh or ask a question.")

# -----------------------------------
# FORCE RELOAD
# -----------------------------------
force_reload = st.checkbox("Force rebuild knowledge base")

# -----------------------------------
# LOAD VECTORSTORE
# -----------------------------------
@st.cache_resource
def load_vectorstore(force_reload: bool):

    documents = []

    for file in os.listdir(docs_folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(docs_folder, file)

            loader = PyMuPDFLoader(file_path)
            docs = loader.load()

            for d in docs:
                d.metadata["source_file"] = file

            documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    if os.path.exists(index_path) and not force_reload:

        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(index_path)

    return vectorstore


# -----------------------------------
# VECTORSTORE
# -----------------------------------
vectorstore = load_vectorstore(force_reload)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# -----------------------------------
# LLM
# -----------------------------------
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

# -----------------------------------
# USER INPUT (CLEAN UI)
# -----------------------------------
question = st.text_input("Ask a question from your documents:")

if st.button("Ask Question"):

    if not question.strip():
        st.warning("Please enter a question")

    else:

        with st.spinner("Searching knowledge base..."):

            retrieved_docs = retriever.invoke(question)

            context = "\n\n".join(doc.page_content for doc in retrieved_docs)

            prompt = f"""
You are a knowledge base assistant.

Answer ONLY using the provided context.

Rules:
- Do NOT hallucinate
- Use exact values from document
- If not found, say: "I don't know based on the document."

Context:
{context}

Question:
{question}
"""

            response = llm.invoke(prompt)

            st.subheader("Answer")
            st.success(response.content)