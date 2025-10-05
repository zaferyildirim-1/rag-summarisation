import os
from pathlib import Path
import tempfile
import time
from datetime import datetime

import streamlit as st

# Langchain / embeddings / vectorstore
from langchain_community.document_loaders import pypdfium2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

from rag_bert.prompts_summarize import SUMMARY_PROMPT, COMBINE_PROMPT
from rag_bert.summarize_pipeline import (
    build_vectorstore_from_documents,
    create_qa_chain,
    load_pdf_documents,
)


st.set_page_config(page_title="RAG Summarization", layout="wide")


def sidebar_config():
    st.sidebar.header("🔧 Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

    with st.sidebar.expander("Advanced Options", expanded=False):
        k = st.slider("Number of retrieved chunks", 1, 15, 5)
        temperature = st.slider("Response creativity", 0.0, 1.0, 0.1, step=0.1)
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=2000, value=1000)
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=200)
        llm_model = st.selectbox("LLM model", ["gpt-4o-mini", "gpt-4o"], index=0)
        persist_dir = st.text_input("Persist directory (optional)", value="./chroma_store")
        use_persist = st.checkbox("Persist vectorstore to disk", value=False)

    return {
        "api_key": api_key,
        "k": k,
        "temperature": temperature,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "llm_model": llm_model,
        "persist_dir": persist_dir,
        "use_persist": use_persist,
    }


cfg = sidebar_config()

if cfg["api_key"]:
    os.environ["OPENAI_API_KEY"] = cfg["api_key"]


st.title("📄 RAG Summarization")
st.markdown("Upload a PDF (or paste text) and generate a summary using Retrieval-Augmented Generation.")

# Upload / paste area
col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload a PDF to summarize", type=["pdf"])
    paste_text = st.text_area("Or paste text to summarize (optional)", height=120)

with col2:
    st.metric("Session start", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# container for status info
status = st.empty()

# Build vectorstore if upload or paste exists
if uploaded or paste_text:
    if st.button("🔍 Process Document"):
        # Use stepwise progress UI
        progress = st.progress(0)
        status.text("Starting processing...")
        try:
            if uploaded:
                # save temp file and load
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                status.text("Loading PDF pages...")
                progress.progress(10)
                docs = load_pdf_documents(tmp_path)
                os.unlink(tmp_path)
            else:
                status.text("Preparing text document...")
                progress.progress(10)
                # LangChain expects Document-like objects; the splitter will accept plain text
                docs = [paste_text]

            status.text("Splitting into chunks...")
            progress.progress(30)
            from rag_bert.summarize_pipeline import split_documents, create_vectorstore_from_chunks

            chunks = split_documents(docs, chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"])
            progress.progress(60)

            status.text("Creating embeddings and vectorstore...")
            if cfg.get("use_persist"):
                vs = create_vectorstore_from_chunks(chunks, persist_directory=cfg.get("persist_dir"))
            else:
                vs = create_vectorstore_from_chunks(chunks, persist_directory=None)

            progress.progress(90)
            st.session_state.vectorstore = vs
            progress.progress(100)
            status.text("")
            st.success("✅ Document processed and vectorstore created")
        except Exception as e:
            status.text("")
            st.error(f"Error processing document: {e}")

if "vectorstore" in st.session_state:
    st.markdown("---")
    st.subheader("Generate summary")

    chain_type = st.selectbox("Chain type", ["map_reduce", "refine", "stuff"], index=0)
    length = st.selectbox("Summary length", ["short", "medium", "long"], index=1)

    if st.button("📝 Create Summary"):
        with st.spinner("Generating summary..."):
            try:
                qa = create_qa_chain(
                    st.session_state.vectorstore,
                    k=cfg["k"],
                    temperature=cfg["temperature"],
                    llm_model=cfg["llm_model"],
                    chain_type=chain_type,
                )

                prompt_instruction = {
                    "short": "Provide a concise summary in 3-5 sentences.",
                    "medium": "Provide a summary with key bullets and a short paragraph (approx 150-300 words).",
                    "long": "Provide a detailed summary with sections and important citations (approx 600-1000 words).",
                }[length]

                result = qa.invoke({"query": prompt_instruction})
                if isinstance(result, dict):
                    summary = result.get("result") or str(result)
                    sources = result.get("source_documents", [])
                else:
                    summary = str(result)
                    sources = []

                st.session_state.last_summary = {
                    "summary": summary,
                    "sources": sources,
                }

                st.markdown("### Summary")
                st.write(summary)

                # Download options
                col_d1, col_d2 = st.columns([1, 1])
                with col_d1:
                    st.download_button("⬇️ Download .txt", summary, file_name="summary.txt")
                with col_d2:
                    st.download_button("⬇️ Download .md", summary, file_name="summary.md")

                if sources:
                    with st.expander("📚 Sources / Evidence", expanded=False):
                        for i, s in enumerate(sources[:6]):
                            st.markdown(f"**Source {i+1}:**")
                            try:
                                st.text(s.page_content[:1000])
                            except Exception:
                                st.text(str(s))

                # Follow-up QA box
                st.markdown("---")
                st.subheader("Ask a follow-up question")
                follow_q = st.text_input("Ask a question about the summary or document", key="follow_q")
                if st.button("❓ Ask", key="follow_qa") and follow_q:
                    try:
                        follow_qa = create_qa_chain(
                            st.session_state.vectorstore,
                            k=cfg["k"],
                            temperature=cfg["temperature"],
                            llm_model=cfg["llm_model"],
                        )
                        res = follow_qa.invoke({"query": follow_q})
                        if isinstance(res, dict):
                            reply = res.get("result") or str(res)
                        else:
                            reply = str(res)
                        st.markdown("**Answer:**")
                        st.write(reply)
                    except Exception as e:
                        st.error(f"Error answering follow-up question: {e}")

            except Exception as e:
                st.error(f"Error generating summary: {e}")

    if "last_summary" in st.session_state:
        st.markdown("---")
        st.download_button("⬇️ Download summary", st.session_state.last_summary["summary"], file_name="summary.txt")

else:
    st.info("Upload a PDF or paste text and click 'Process Document' to begin")
