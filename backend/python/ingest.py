"""
ingest.py — MongoDB Atlas Vector Search storage & retrieval.

Replaces ChromaDB with MongoDB Atlas Vector Search:
- ingest_docs()          → embed healt_book.txt into 'medical_vectors' collection
- ingest_conversation()  → embed user Q&A turns into 'user_vectors' collection
- get_static_context()   → $vectorSearch on medical_vectors
- get_user_context()     → $vectorSearch on user_vectors (filtered by user_id)
"""

import os
import hashlib
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from pymongo import MongoClient

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://apoorv13:wmxfzy5ZPQJY5P7L@cluster0.dzdexwp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)

_client = MongoClient(MONGO_URI)
_db = _client.get_default_database("hos")
_medical_vectors = _db["medical_vectors"]
_user_vectors = _db["user_vectors"]


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


# ─── Static Medical Docs Ingestion ──────────────────────────────────────────

def ingest_docs(file_path: str):
    """
    Read a text file, chunk it, compute embeddings, and store in MongoDB
    'medical_vectors' collection. Skips if collection already has data.
    """
    # Check if already ingested
    existing_count = _medical_vectors.count_documents({})
    if existing_count > 0:
        print(f"✅ medical_vectors already has {existing_count} documents. Skipping ingestion.")
        return existing_count

    print(f"📄 Loading document: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chunks = manual_split_text(text)
    print(f"✂️  Split into {len(chunks)} chunks. Computing embeddings...")

    embeddings_model = _get_embeddings()

    # Batch embed all chunks
    vectors = embeddings_model.embed_documents(chunks)
    print(f"🧮 Computed {len(vectors)} embeddings (dim={len(vectors[0])})")

    # Build MongoDB documents
    docs = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        docs.append({
            "text": chunk,
            "embedding": vector,
            "source": file_path,
            "chunk_index": i,
            "created_at": datetime.utcnow()
        })

    # Batch insert
    BATCH_SIZE = 500
    for start in range(0, len(docs), BATCH_SIZE):
        batch = docs[start:start + BATCH_SIZE]
        _medical_vectors.insert_many(batch)
        print(f"   📦 Inserted batch {start // BATCH_SIZE + 1}/{(len(docs) + BATCH_SIZE - 1) // BATCH_SIZE}")

    print(f"✅ Ingested {len(docs)} chunks into MongoDB 'medical_vectors'")
    return len(docs)


# ─── Per-User Conversation Ingestion ────────────────────────────────────────

def ingest_conversation(user_id: str, text: str):
    """
    Embed a single Q&A turn and store in MongoDB 'user_vectors' collection.
    Called async after each /ask response to build per-user memory.
    """
    if not text or not text.strip():
        return

    chunks = manual_split_text(text, chunk_size=500, chunk_overlap=50)
    embeddings_model = _get_embeddings()
    vectors = embeddings_model.embed_documents(chunks)

    docs = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        docs.append({
            "text": chunk,
            "embedding": vector,
            "user_id": user_id,
            "source": "conversation",
            "type": "conversation_turn",
            "created_at": datetime.utcnow()
        })

    try:
        _user_vectors.insert_many(docs)
        print(f"🧬 Ingested conversation turn for {user_id} ({len(chunks)} chunks)")
    except Exception as e:
        print(f"⚠️ Conversation ingestion error for {user_id}: {e}")


# ─── Vector Search Queries ──────────────────────────────────────────────────

def get_static_context(query: str, k: int = 3) -> str:
    """
    Search 'medical_vectors' using MongoDB Atlas $vectorSearch.
    Returns concatenated text of top-k matches.
    """
    embeddings_model = _get_embeddings()
    query_vector = embeddings_model.embed_query(query)

    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "medical_vectors_index",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": k * 20,
                    "limit": k
                }
            },
            {
                "$project": {
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"},
                    "_id": 0
                }
            }
        ]
        results = list(_medical_vectors.aggregate(pipeline))
        if results:
            return "\n\n".join([r["text"] for r in results])
        else:
            print("⚠️ No vector search results from medical_vectors")
            return ""
    except Exception as e:
        print(f"⚠️ Static vector search error: {e}")
        # Fallback: text search
        return _fallback_text_search(_medical_vectors, query, k)


def get_user_context(query: str, user_id: str, k: int = 3) -> str:
    """
    Search 'user_vectors' using MongoDB Atlas $vectorSearch filtered by user_id.
    Returns concatenated text of top-k matches.
    """
    # Check if user has any vectors
    if _user_vectors.count_documents({"user_id": user_id}) == 0:
        return ""

    embeddings_model = _get_embeddings()
    query_vector = embeddings_model.embed_query(query)

    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "user_vectors_index",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": k * 20,
                    "limit": k,
                    "filter": {"user_id": user_id}
                }
            },
            {
                "$project": {
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"},
                    "_id": 0
                }
            }
        ]
        results = list(_user_vectors.aggregate(pipeline))
        if results:
            return "\n\n".join([r["text"] for r in results])
        return ""
    except Exception as e:
        print(f"⚠️ User vector search error for {user_id}: {e}")
        return _fallback_text_search(_user_vectors, query, k, {"user_id": user_id})


def _fallback_text_search(collection, query: str, k: int = 3, extra_filter: dict = None) -> str:
    """
    Fallback if Atlas Vector Search index is not yet created.
    Uses simple text matching on the 'text' field.
    """
    try:
        keywords = query.lower().split()[:5]
        regex_pattern = "|".join(keywords)
        filter_query = {"text": {"$regex": regex_pattern, "$options": "i"}}
        if extra_filter:
            filter_query.update(extra_filter)
        results = list(collection.find(filter_query, {"text": 1, "_id": 0}).limit(k))
        if results:
            print(f"📋 Fallback text search returned {len(results)} results")
            return "\n\n".join([r["text"] for r in results])
        return ""
    except Exception as e:
        print(f"⚠️ Fallback search error: {e}")
        return ""


# ─── CLI Entry Point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_file = "healt_book.txt"
    if os.path.exists(sample_file):
        count = ingest_docs(sample_file)
        print(f"\n🎉 Done! {count} chunks stored in MongoDB Atlas.")
        print(f"\n⚠️  NEXT STEP: Create Atlas Vector Search indexes!")
        print(f"   Go to MongoDB Atlas → Database → Browse Collections → 'hos'")
        print(f"   → Click 'Search Indexes' tab → Create index")
        print(f"   See implementation_plan.md for the exact JSON definitions.")
    else:
        print(f"File {sample_file} not found.")
