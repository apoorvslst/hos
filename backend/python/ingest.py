import os
import hashlib
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# ─── Shared Embeddings (singleton) ──────────────────────────────────────────
_embeddings = None

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _embeddings


def manual_split_text(text, chunk_size=1000, chunk_overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - chunk_overlap)
    return chunks


def ingest_docs(file_path: str, persist_directory: str = "vectordb"):
    """Loads a document and stores it in ChromaDB using FastEmbed."""
    print(f"Loading document: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Manual splitting to avoid LangChain's torch-dependent splitters
    chunks = manual_split_text(text)
    docs = [Document(page_content=chunk, metadata={"source": file_path}) for chunk in chunks]
    print(f"Split into {len(docs)} chunks.")

    embeddings = _get_embeddings()
    
    print("Creating vector database...")
    if os.path.exists(persist_directory):
        print(f"Updating existing database at {persist_directory}...")
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name="medical_docs"
        )
        vectordb.add_documents(docs)
    else:
        print(f"Creating new database at {persist_directory}...")
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name="medical_docs"
        )
    print(f"Vector DB created and persisted at {persist_directory}")
    return vectordb


# ─── Per-User Conversation Ingestion ────────────────────────────────────────

def _user_collection_name(user_id: str) -> str:
    """Generate a safe collection name from user_id."""
    h = hashlib.md5(user_id.encode()).hexdigest()[:12]
    return f"user_{h}"


def _user_persist_dir(user_id: str) -> str:
    """Return the per-user ChromaDB persist directory."""
    h = hashlib.md5(user_id.encode()).hexdigest()[:12]
    base = os.path.join(os.path.dirname(__file__), "vectordb_users", f"user_{h}")
    os.makedirs(base, exist_ok=True)
    return base


def ingest_conversation(user_id: str, text: str):
    """
    Ingest a single Q&A turn into the user's personal ChromaDB collection.
    Called after each /ask response to build per-user memory.
    """
    if not text or not text.strip():
        return

    persist_dir = _user_persist_dir(user_id)
    collection_name = _user_collection_name(user_id)
    embeddings = _get_embeddings()

    # Chunk the conversation turn (usually small, but be safe)
    chunks = manual_split_text(text, chunk_size=500, chunk_overlap=50)
    docs = [
        Document(
            page_content=chunk,
            metadata={
                "source": "conversation",
                "user_id": user_id,
                "type": "conversation_turn"
            }
        )
        for chunk in chunks
    ]

    try:
        if os.path.exists(persist_dir) and any(os.listdir(persist_dir)):
            vectordb = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            vectordb.add_documents(docs)
        else:
            Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_name=collection_name
            )
        print(f"🧬 Ingested conversation turn for {user_id} ({len(chunks)} chunks)")
    except Exception as e:
        print(f"⚠️ Conversation ingestion error for {user_id}: {e}")


def get_user_retriever(user_id: str, k: int = 3):
    """
    Return a retriever for the user's personal ChromaDB collection.
    Returns None if the user has no conversation history yet.
    """
    persist_dir = _user_persist_dir(user_id)
    collection_name = _user_collection_name(user_id)
    
    if not os.path.exists(persist_dir) or not any(os.listdir(persist_dir)):
        return None
    
    try:
        embeddings = _get_embeddings()
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        return vectordb.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        print(f"⚠️ Could not load user retriever for {user_id}: {e}")
        return None


if __name__ == "__main__":
    sample_file = "healt_book.txt"
    if os.path.exists(sample_file):
        ingest_docs(sample_file)
    else:
        print(f"File {sample_file} not found.")
