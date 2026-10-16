
import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# ============================================================
# CONFIGURATION
# ============================================================

import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "qwen/qwen3.8-27b"  # Replace with your actual Groq model ID

INDEX_PATH = "faiss_store_groq"


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="PKBot News Research Tool",
    page_icon="📰",
    layout="wide"
)

st.title("📰 PKBot News Research Tool")
st.caption("Powered by Groq, LangChain and FAISS")


# ============================================================
# LOCAL EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# ============================================================
# GROQ LLM
# ============================================================

def get_llm():
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=700,
        max_retries=2,
        timeout=60
    )


# ============================================================
# URL INPUTS
# ============================================================

st.subheader("Step 1: Enter News Article URLs")

url1 = st.text_input("News URL 1")
url2 = st.text_input("News URL 2")
url3 = st.text_input("News URL 3")

urls = [
    url.strip()
    for url in [url1, url2, url3]
    if url.strip()
]


# ============================================================
# PROCESS URLS
# ============================================================

if st.button("🚀 Process URLs", type="primary"):

    if not urls:
        st.warning("Please enter at least one news URL.")

    elif not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY":
        st.error("Add your Groq API key in the configuration section.")

    else:
        try:
            with st.spinner("Processing news articles..."):

                loader = UnstructuredURLLoader(
                    urls=urls,
                    continue_on_failure=True
                )

                documents = loader.load()

                documents = [
                    doc for doc in documents
                    if doc.page_content and doc.page_content.strip()
                ]

                if not documents:
                    st.error(
                        "No article text was extracted. "
                        "Check your URLs and try again."
                    )
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )

                chunks = splitter.split_documents(documents)

                embeddings = get_embeddings()

                vectorstore = FAISS.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )

                vectorstore.save_local(INDEX_PATH)

            st.session_state["news_processed"] = True

            st.success(
                f"Processed {len(documents)} article(s) "
                f"and created {len(chunks)} text chunks."
            )

        except Exception as e:
            st.error(f"Error processing URLs: {e}")


# ============================================================
# ASK QUESTIONS
# ============================================================

st.divider()
st.subheader("Step 2: Ask Questions")

question = st.text_input(
    "Your question",
    placeholder="What are the main points in these news articles?"
)

if st.button("💬 Get Answer", type="primary"):

    index_file = os.path.join(INDEX_PATH, "index.faiss")

    if not os.path.exists(index_file):
        st.warning("Please process your URLs first.")

    elif not question.strip():
        st.warning("Please enter a question.")

    elif not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY":
        st.error("Please configure your Groq API key.")

    else:
        try:
            with st.spinner("Groq is generating your answer..."):

                embeddings = get_embeddings()

                # Load only the FAISS index created by this app.
                vectorstore = FAISS.load_local(
                    INDEX_PATH,
                    embeddings,
                    allow_dangerous_deserialization=True
                )

                llm = get_llm()

                qa_chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(
                        search_kwargs={"k": 4}
                    )
                )

                result = qa_chain.invoke({
                    "question": question.strip()
                })

            st.subheader("Answer")
            st.write(result.get("answer", "No answer returned."))

            sources = result.get("sources", "")

            if sources:
                st.subheader("Sources")
                st.write(sources)

        except Exception as e:
            st.error(f"Error generating answer: {e}")
            st.info(
                "Check your Groq API key, model availability, "
                "internet connection, and API rate limits."
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("About RockyBot")
    st.write("**LLM:** Groq")
    st.write(f"**Model:** {GROQ_MODEL}")
    st.write("**Vector store:** FAISS")
    st.write("**Embeddings:** Local Hugging Face model")
