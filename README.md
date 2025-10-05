# RAG Summarizer Streamlit

Minimal Streamlit app and pipeline for RAG-based summarization. This scaffold includes:

- `streamlit_app_summarization.py` - the Streamlit app UI
- `rag_bert_summarize_pipeline.py` - pipeline helpers (split, build vectorstore, QA chain)
- `rag_bert_prompts_summarize.py` - prompt templates

Run locally with:

```bash
pip install -r requirements.txt
streamlit run streamlit_app_summarization.py
```

Note: Set your OpenAI API key in the app sidebar. Chroma persistence directory can be configured in the sidebar as well.
