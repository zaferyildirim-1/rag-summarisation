# --- must be at the very top ---
import os
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "poll"   # avoid inotify ENOSPC
os.environ["STREAMLIT_SERVER_RUNONSAVE"] = "false"        # reduce watch activity
import os
import tempfile
import time
from datetime import datetime

import streamlit as st

# Try to import the rag_bert package modules (package layout) and fallback to top-level filenames
try:
    from rag_bert.prompts_summarize import (
        CHUNK_SUMMARY_PROMPT,
        COMBINE_SUMMARY_PROMPT,
        REFINE_PROMPT,
        INSTRUCTION_SHORT,
        INSTRUCTION_MEDIUM,
        INSTRUCTION_LONG,
    )
    from rag_bert.summarize_pipeline import (
        read_pdf,
        split_documents,
        build_chroma,
        load_persisted_chroma,
        summarize_with_chain,
    )
except Exception:
    # fallback: if files are top-level modules (rag_bert_prompts_summarize.py etc.)
    try:
        from rag_bert_prompts_summarize import (
            CHUNK_SUMMARY_PROMPT,
            COMBINE_SUMMARY_PROMPT,
            REFINE_PROMPT,
            INSTRUCTION_SHORT,
            INSTRUCTION_MEDIUM,
            INSTRUCTION_LONG,
        )
        from rag_bert_summarize_pipeline import (
            read_pdf,
            split_documents,
            build_chroma,
            load_persisted_chroma,
            summarize_with_chain,
        )
    except Exception:
        st.error("Could not import local prompt/pipeline modules. Ensure `rag_bert` package or top-level modules exist.")
        raise

# Wrap the optional import for PDF loader to give clearer instructions if missing
try:
    # used by read_pdf inside pipeline; imported here for early error messages if absent
    from langchain_community.document_loaders import PyPDFium2Loader  # noqa: F401
except Exception as e:
    st.error(
        "Missing runtime dependency `langchain-community` or `pypdfium2` required for PDF loading.\n"
        "Please ensure your `requirements.txt` includes langchain-community and pypdfium2 and redeploy.\n"
        f"Original import error: {e}"
    )
    raise

st.set_page_config(page_title="RAG Summarization", layout="wide")
st.title("📄 RAG Summarization")
st.markdown("Upload a PDF (or paste text) and generate a summary using Retrieval-Augmented Generation.")

# Sidebar config
with st.sidebar:
    st.header("🔧 Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    with st.expander("Advanced Options", expanded=False):
        k = st.slider("Number of retrieved chunks", 1, 15, 5)
        temperature = st.slider("Response creativity", 0.0, 1.0, 0.0, step=0.1)
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=2000, value=1000)
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=200)
        llm_model = st.selectbox("LLM model", ["gpt-4o-mini", "gpt-4o"], index=0)
        persist_dir = st.text_input("Persist directory (optional)", value="./chroma_store")
        use_persist = st.checkbox("Persist vectorstore to disk", value=False)

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Main UI
col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload a PDF to summarize", type=["pdf"])
    paste_text = st.text_area("Or paste text to summarize (optional)", height=150)
with col2:
    st.metric("Session start", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

status = st.empty()

# Process document
if uploaded or paste_text:
    if st.button("🔍 Process Document"):
        progress = st.progress(0)
        status.text("Starting processing...")
        try:
            if uploaded:
                # Save and read pdf
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                status.text("Loading PDF pages...")
                progress.progress(10)
                docs = read_pdf(tmp_path)
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            else:
                status.text("Preparing text document...")
                progress.progress(10)
                # wrap plain text in a Document-like structure expected by splitter/loaders
                docs = [paste_text]

            status.text("Splitting into chunks...")
            progress.progress(30)
            chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            progress.progress(60)

            status.text("Creating embeddings and vectorstore...")
            if use_persist:
                vs = build_chroma(chunks, persist_directory=persist_dir)
            else:
                vs = build_chroma(chunks, persist_directory=None)

            progress.progress(100)
            st.session_state.vectorstore = vs
            status.text("")
            st.success("✅ Document processed and vectorstore created")
        except Exception as e:
            status.text("")
            st.error(f"Error processing document: {e}")

# Summarize
if "vectorstore" in st.session_state:
    st.markdown("---")
    st.subheader("Generate summary")
    chain_type = st.selectbox("Chain type", ["map_reduce", "refine", "stuff"], index=0)
    length = st.selectbox("Summary length", ["short", "medium", "long"], index=1)

    if st.button("📝 Create Summary"):
        with st.spinner("Generating summary..."):
            try:
                instruction_map = {
                    "short": INSTRUCTION_SHORT,
                    "medium": INSTRUCTION_MEDIUM,
                    "long": INSTRUCTION_LONG,
                }
                instruction = instruction_map[length]
                summary, sources = summarize_with_chain(
                    st.session_state.vectorstore,
                    method=chain_type,
                    instruction=instruction,
                    llm_model=llm_model,
                    temperature=temperature,
                    k=k,
                )

                st.session_state.last_summary = {"summary": summary, "sources": sources}
                st.markdown("### Summary")
                st.write(summary)

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

                # Follow-up QA
                st.markdown("---")
                st.subheader("Ask a follow-up question")
                follow_q = st.text_input("Ask a question about the summary or document", key="follow_q")
                if st.button("❓ Ask", key="follow_qa") and follow_q:
                    try:
                        # reuse summarize pipeline's QA fallback
                        qa_summary, _ = summarize_with_chain(
                            st.session_state.vectorstore,
                            method="retrieval",
                            instruction=follow_q,
                            llm_model=llm_model,
                            temperature=temperature,
                            k=k,
                        )
                        st.markdown("**Answer:**")
                        st.write(qa_summary)
                    except Exception as e:
                        st.error(f"Error answering follow-up question: {e}")
            except Exception as e:
                st.error(f"Error generating summary: {e}")
else:
    st.info("Upload a PDF or paste text and click 'Process Document' to begin")
