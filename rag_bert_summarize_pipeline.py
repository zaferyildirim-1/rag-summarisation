from typing import List, Optional, Tuple
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Prefer load_summarize_chain if available
try:
    from langchain.chains.summarize import load_summarize_chain
except Exception:
    load_summarize_chain = None

from rag_bert.prompts_summarize import (
    CHUNK_SUMMARY_PROMPT,
    COMBINE_SUMMARY_PROMPT,
    REFINE_PROMPT,
    INSTRUCTION_SHORT,
    INSTRUCTION_MEDIUM,
    INSTRUCTION_LONG,
)


def read_pdf(path: str):
    """
    Load PDF and return a list of LangChain Document objects (pages).
    """
    loader = PyPDFium2Loader(path)
    return loader.load()


def split_documents(docs, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    Returns a list of Document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    return chunks


def create_embeddings(model_name: Optional[str] = None):
    """
    Create and return embedding function. If you want a specific model, pass model_name.
    """
    if model_name:
        return OpenAIEmbeddings(model=model_name)
    return OpenAIEmbeddings()


def build_chroma(chunks, persist_directory: Optional[str] = None, embeddings=None):
    """
    Build a Chroma vectorstore from chunks. Optionally persist to a directory.
    Returns a Chroma instance.
    """
    if embeddings is None:
        embeddings = create_embeddings()

    if persist_directory:
        vs = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
        try:
            vs.persist()
        except Exception:
            pass
    else:
        vs = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vs


def load_persisted_chroma(persist_directory: str, embeddings: Optional[OpenAIEmbeddings] = None):
    """
    Load an existing Chroma store from disk.
    """
    if embeddings is None:
        embeddings = create_embeddings()
    vs = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vs


def create_llm(model: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: Optional[int] = None):
    """
    Create a ChatOpenAI instance. Handles different ChatOpenAI constructor arg names.
    """
    # Try common param names. This keeps compatibility across langchain-openai versions.
    kwargs = {"temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    # prefer 'model' then fallback to 'model_name' if needed
    try:
        llm = ChatOpenAI(model=model, **kwargs)
        return llm
    except TypeError:
        # older versions used model_name
        llm = ChatOpenAI(model_name=model, **kwargs)
        return llm


def create_summarize_chain(llm, chain_type: str = "map_reduce"):
    """
    Return a summarization chain instance if available; otherwise None.
    """
    if load_summarize_chain is not None:
        try:
            return load_summarize_chain(llm, chain_type=chain_type, verbose=False)
        except Exception:
            return None
    return None


def create_qa_chain_from_vectorstore(vs: Chroma, llm_model: str = "gpt-4o-mini", temperature: float = 0.0, k: int = 5):
    retriever = vs.as_retriever(search_kwargs={"k": k})
    llm = create_llm(model=llm_model, temperature=temperature)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True)


def summarize_with_chain(vs: Chroma, method: str = "map_reduce", instruction: str = INSTRUCTION_MEDIUM,
                         llm_model: str = "gpt-4o-mini", temperature: float = 0.0, k: int = 5) -> Tuple[str, List]:
    """
    High level: try load_summarize_chain for map_reduce/refine/stuff; fallback to RetrievalQA for quick summary.
    Returns (summary_text, source_documents_or_chunks)
    """
    llm = create_llm(model=llm_model, temperature=temperature)
    chain = create_summarize_chain(llm, chain_type=method)

    retriever = vs.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(instruction)

    # If chain exists, use it on the docs
    if chain is not None:
        try:
            # Common interfaces: chain.run(docs) or chain.invoke({"input_documents": docs})
            if hasattr(chain, "run"):
                result = chain.run(docs)
            else:
                # some versions accept invoke with dict
                result = chain.invoke({"input_documents": docs})
            if isinstance(result, dict):
                summary = result.get("output_text") or result.get("result") or str(result)
            else:
                summary = str(result)
            return summary, docs
        except Exception:
            # fall through to fallback
            pass

    # Fallback: RetrievalQA -> use instruction as query
    qa = create_qa_chain_from_vectorstore(vs, llm_model=llm_model, temperature=temperature, k=k)
    out = qa({"query": instruction})
    if isinstance(out, dict):
        summary = out.get("result") or out.get("answer") or str(out)
        sources = out.get("source_documents", [])
    else:
        summary = str(out)
        sources = []
    return summary, sources
