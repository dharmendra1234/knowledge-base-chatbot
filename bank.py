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
#load_dotenv()


if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# -----------------------------------
# STREAMLIT CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Bank Statement Chatbot",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Multi-Bank Statement PDF Chatbot")

# -----------------------------------
# PATHS
# -----------------------------------
docs_folder = "docs"
index_path = "bank_faiss_index"

# -----------------------------------
# RESET VECTOR DB
# -----------------------------------
if st.button("🔄 Reset Vector DB (Delete & Rebuild)"):
    if os.path.exists(index_path):
        shutil.rmtree(index_path)

    st.cache_resource.clear()
    st.success("Vector DB cleared. Please rerun or ask a question.")

# -----------------------------------
# FORCE REBUILD OPTION
# -----------------------------------
force_reload = st.checkbox("Force rebuild from PDFs")

# -----------------------------------
# LOAD VECTORSTORE FUNCTION
# -----------------------------------
@st.cache_resource
def load_vectorstore(force_reload: bool):

    # -----------------------------------
    # LOAD ALL PDFs FROM FOLDER
    # -----------------------------------
    documents = []

    for file in os.listdir(docs_folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(docs_folder, file)

            loader = PyMuPDFLoader(file_path)
            docs = loader.load()

            # Add metadata for traceability
            for d in docs:
                d.metadata["source_file"] = file

            documents.extend(docs)

    # -----------------------------------
    # SHOW RAW TEXT (DEBUG)
    # -----------------------------------
    with st.expander("📄 Extracted PDF Text"):
        for i, doc in enumerate(documents[:5]):  # limit preview
            st.write(doc.page_content)

    # -----------------------------------
    # SPLIT TEXT
    # -----------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    docs = text_splitter.split_documents(documents)

    # -----------------------------------
    # SHOW CHUNKS (DEBUG)
    # -----------------------------------
    with st.expander("🧩 Chunks Preview"):
        st.write(f"Total Chunks: {len(docs)}")

        for i, doc in enumerate(docs[:10]):  # limit preview
            st.markdown(f"### Chunk {i+1}")
            st.write(doc.page_content)
            st.divider()

    # -----------------------------------
    # EMBEDDINGS
    # -----------------------------------
    embeddings = OpenAIEmbeddings()

    # -----------------------------------
    # LOAD OR BUILD FAISS
    # -----------------------------------
    if os.path.exists(index_path) and not force_reload:

        st.info("Loading existing FAISS index...")

        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:

        st.info("Creating NEW FAISS index...")

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(index_path)

    return vectorstore


# -----------------------------------
# INIT VECTORSTORE
# -----------------------------------
vectorstore = load_vectorstore(force_reload)

# -----------------------------------
# RETRIEVER
# -----------------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# -----------------------------------
# LLM
# -----------------------------------
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

# -----------------------------------
# USER INPUT
# -----------------------------------
question = st.text_input("Ask a question about the bank statements:")

# -----------------------------------
# ASK BUTTON
# -----------------------------------
if st.button("Ask Question"):

    if not question.strip():
        st.warning("Please enter a question")

    else:

        with st.spinner("Searching documents..."):

            # -------------------------
            # RETRIEVE
            # -------------------------
            retrieved_docs = retriever.invoke(question)

            # -------------------------
            # SHOW RETRIEVED CHUNKS
            # -------------------------
            with st.expander("🔍 Retrieved Chunks"):
                for i, doc in enumerate(retrieved_docs):
                    st.markdown(f"### Chunk {i+1}")
                    st.write(doc.page_content)
                    st.caption(f"Source: {doc.metadata.get('source_file', 'unknown')}")
                    st.divider()

            # -------------------------
            # BUILD CONTEXT
            # -------------------------
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)

            # -------------------------
            # PROMPT
            # -------------------------
            prompt = f"""
You are a bank statement assistant.

Answer ONLY using the provided context.

Rules:
- Do NOT hallucinate numbers
- Return exact values as written
- Do not calculate unless asked
- If not found, say: "I don't know based on the document."

Context:
{context}

Question:
{question}
"""

            # -------------------------
            # GET RESPONSE
            # -------------------------
            response = llm.invoke(prompt)

            # -------------------------
            # OUTPUT
            # -------------------------
            st.subheader("✅ Answer")
            st.success(response.content)