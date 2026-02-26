"""
conversation_store.py — Per-user MongoDB conversation storage + context builder.

Stores each Q&A turn in a per-user document in the 'user_conversations' collection.
Provides methods to fetch recent context and turn counts for the follow-up loop.
"""

import os
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://apoorv13:wmxfzy5ZPQJY5P7L@cluster0.dzdexwp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)

_client = AsyncIOMotorClient(MONGO_URI)
_db = _client.get_default_database("test")
_conversations = _db["user_conversations"]


def _resolve_user_id(user_email: str = None, user_phone: str = None) -> str:
    """Resolve user identity: email > phone > anonymous."""
    if user_email and user_email.strip():
        return user_email.strip().lower()
    if user_phone and user_phone.strip():
        return user_phone.strip()
    return "anonymous"


async def save_turn(
    user_id: str,
    eng_query: str,
    eng_answer: str,
    clinical_data: dict = None,
    session_id: str = None
):
    """Save a single Q&A turn into the user's conversation document."""
    turn = {
        "timestamp": datetime.utcnow(),
        "user_message": eng_query,
        "ai_response": eng_answer,
        "symptoms": (clinical_data or {}).get("symptoms", []),
        "severity": (clinical_data or {}).get("severity", 5),
        "summary": (clinical_data or {}).get("summary", ""),
        "session_id": session_id or datetime.utcnow().strftime("%Y%m%d")
    }

    await _conversations.update_one(
        {"user_id": user_id},
        {
            "$push": {"turns": turn},
            "$set": {"updated_at": datetime.utcnow()},
            "$setOnInsert": {"user_id": user_id, "created_at": datetime.utcnow()}
        },
        upsert=True
    )
    print(f"💾 ConvStore: saved turn for {user_id}")


async def get_recent_context(user_id: str, limit: int = 10) -> str:
    """
    Fetch the last N turns for a user and format them as context text.
    This text is injected into the RAG prompt so the LLM has conversation history.
    """
    doc = await _conversations.find_one({"user_id": user_id})
    if not doc or not doc.get("turns"):
        return ""

    recent_turns = doc["turns"][-limit:]
    context_lines = []
    for t in recent_turns:
        context_lines.append(f"Patient: {t['user_message']}")
        context_lines.append(f"Doctor: {t['ai_response']}")
        if t.get("symptoms"):
            context_lines.append(f"  [Symptoms noted: {', '.join(t['symptoms'])}]")
    
    return "\n".join(context_lines)


async def get_turn_count(user_id: str) -> int:
    """Return how many turns this user has had (for follow-up loop control)."""
    doc = await _conversations.find_one({"user_id": user_id})
    if not doc or not doc.get("turns"):
        return 0
    return len(doc["turns"])


async def get_session_turn_count(user_id: str, session_id: str = None) -> int:
    """Return turn count for the current session only."""
    if not session_id:
        session_id = datetime.utcnow().strftime("%Y%m%d")
    
    doc = await _conversations.find_one({"user_id": user_id})
    if not doc or not doc.get("turns"):
        return 0
    
    return sum(1 for t in doc["turns"] if t.get("session_id") == session_id)
