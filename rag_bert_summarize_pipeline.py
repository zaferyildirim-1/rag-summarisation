from typing import List, Any, Optional
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


def load_pdf_documents(path: str) -> List[Any]:
    """Load a PDF from path and return a list of documents (LangChain format)."""
    pages = PyPDFium2Loader(path).load()
    return pages


def split_documents(docs: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split documents into smaller chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    return chunks


def create_vectorstore_from_chunks(
    chunks: List[Any], persist_directory: Optional[str] = None, embeddings: Optional[OpenAIEmbeddings] = None
):
    """Create a Chroma vectorstore from pre-split chunks. Optionally persist to disk."""
    if embeddings is None:
        embeddings = OpenAIEmbeddings()

    if persist_directory:
        # Create or load a persistent Chroma store
        vs = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
        try:
            vs.persist()
        except Exception:
            # Some Chroma wrappers auto-persist; ignore if not available
            pass
    else:
        vs = Chroma.from_documents(documents=chunks, embedding=embeddings)

    return vs


def build_vectorstore_from_documents(docs: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200, persist_directory: Optional[str] = None):
    """Convenience wrapper: split then create vectorstore. Returns the vectorstore."""
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs = create_vectorstore_from_chunks(chunks, persist_directory=persist_directory)
    return vs


def load_persisted_vectorstore(persist_directory: str, embeddings: Optional[OpenAIEmbeddings] = None):
    """Load an existing persisted Chroma vectorstore from disk."""
    if embeddings is None:
        embeddings = OpenAIEmbeddings()
    # Chroma can be instantiated with a persist_directory
    vs = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vs


def create_qa_chain(vs: Chroma, k: int = 5, temperature: float = 0.1, llm_model: str = "gpt-4o-mini", chain_type: str = "map_reduce"):
    retriever = vs.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model=llm_model, temperature=temperature)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type=chain_type,
        return_source_documents=True,
    )
    return qa
