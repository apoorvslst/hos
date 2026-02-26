import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from pathlib import Path
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()


class MedicalRAGService:
    """
    Dynamic RAG service with merged retrieval via MongoDB Atlas Vector Search:
    - Static knowledge base (medical_vectors collection)
    - Per-user conversation history (user_vectors collection)
    - Severity-driven follow-up symptom loop
    """

    def __init__(self):
        # 1. Initialize LLM (Llama 3 via Groq)
        self.llm = ChatGroq(
            temperature=0.1,
            model_name="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        # 2. Auto-ingest medical docs if MongoDB collection is empty
        from ingest import ingest_docs, _medical_vectors
        count = _medical_vectors.count_documents({})
        if count == 0:
            print("⚠️ medical_vectors is empty. Auto-ingesting healt_book.txt...")
            health_file = os.path.join(os.path.dirname(__file__), "healt_book.txt")
            if os.path.exists(health_file):
                ingest_docs(health_file)
            else:
                print(f"❌ Error: {health_file} not found. RAG service will have no medical context.")
        else:
            print(f"✅ medical_vectors has {count} documents ready.")

        # 3. System Prompt — Full-body Physician with Follow-Up Loop
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert physician assistant covering all areas of medicine — general health, internal medicine, cardiology, neurology, orthopedics, dermatology, gastroenterology, gynecology, pediatrics, and more.

You will receive a patient's symptoms along with their past conversation history and medical context from a knowledge base.

CRITICAL BEHAVIOR RULES:
1. VOICE-OPTIMIZED: Your output will be spoken directly via Text-to-Speech. Use a warm, simple, direct conversational tone. Do NOT use markdown, asterisks, bullet points, or special characters.
2. EXTREME BREVITY: Keep responses under 3-4 short sentences for the main answer.
3. NO FLUFF: Do not use filler phrases like "Thank you for sharing" or "I understand your concern". Get straight to the point.
4. DIRECT INFORMATION: Provide direct actionable information. Do not include disclaimers.

DIAGNOSTIC FOLLOW-UP PROTOCOL:
- After answering, you MUST ask exactly ONE specific follow-up question targeting the next most relevant symptom to narrow down possible conditions and diseases based on the medical knowledge base.
- Frame follow-up questions as simple yes/no or short-answer questions the patient can easily respond to.
- Examples: "Are you also experiencing any headaches or dizziness?" or "How long have you had this symptom?"
- Your goal is to progressively narrow down from many possible conditions to the most likely ones through systematic questioning.

{followup_instruction}

CONTEXT PRIORITY (most important first):
1. PATIENT'S PAST CONVERSATIONS — these are the most valuable for personalized diagnosis
2. MEDICAL KNOWLEDGE BASE — use this for clinical accuracy
3. CURRENT SYMPTOMS — evaluate against both sources above

RESPONSE STRUCTURE:
1. Brief assessment of the reported symptom or condition (1-2 sentences)
2. One specific, actionable recommendation
3. ONE follow-up question to gather more diagnostic information"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", """PATIENT'S PAST CONVERSATION CONTEXT:
{user_context}

MEDICAL KNOWLEDGE BASE:
{static_context}

PATIENT DATA:
{patient_data}

CURRENT QUESTION:
{question}

PHYSICIAN RESPONSE:""")
        ])

    def ask_stream(
        self,
        query: str,
        patient_data: str = "None provided",
        chat_history: list = None,
        user_id: str = None,
        conversation_context: str = "",
        turn_count: int = 0
    ):
        """
        Merged RAG retrieval via MongoDB Atlas Vector Search:
        static medical knowledge + per-user conversation history.
        Streams the response with follow-up questions based on severity/turn count.
        """
        if chat_history is None:
            chat_history = []

        # 1. Retrieve from STATIC knowledge base (MongoDB Atlas Vector Search)
        from ingest import get_static_context, get_user_context
        static_context = get_static_context(query, k=3)

        # 2. Retrieve from PER-USER vector history (if available)
        user_context_from_vectordb = ""
        if user_id and user_id != "anonymous":
            try:
                user_context_from_vectordb = get_user_context(query, user_id, k=3)
            except Exception as e:
                print(f"⚠️ User retriever error: {e}")

        # 3. Merge user context: MongoDB conversation context + vectorDB user context
        merged_user_context = ""
        if conversation_context:
            merged_user_context += f"--- Recent Conversation History ---\n{conversation_context}\n\n"
        if user_context_from_vectordb:
            merged_user_context += f"--- Related Past Interactions ---\n{user_context_from_vectordb}"

        if not merged_user_context:
            merged_user_context = "No previous interactions with this patient."

        # 4. Determine follow-up instruction based on turn count
        if turn_count >= 15:
            followup_instruction = (
                "IMPORTANT: This is a concluding turn. You have gathered enough information. "
                "Provide a FINAL SUMMARY of probable conditions based on all symptoms discussed. "
                "Do NOT ask any more follow-up questions. End with a clear recommendation to consult a doctor "
                "with this summary for verification."
            )
        elif turn_count >= 10:
            followup_instruction = (
                "You are in the final rounds of symptom gathering. Ask only if the symptom is critical "
                "for narrowing down the diagnosis. If you have enough information, provide a concluding summary."
            )
        else:
            followup_instruction = (
                "Continue the diagnostic process. Ask a targeted follow-up question about the next "
                "most relevant symptom to narrow down possible conditions."
            )

        # 5. Streaming Generation
        generation_chain = self.rag_prompt | self.llm | StrOutputParser()

        for chunk in generation_chain.stream({
            "chat_history": chat_history,
            "static_context": static_context,
            "user_context": merged_user_context,
            "question": query,
            "patient_data": patient_data,
            "followup_instruction": followup_instruction
        }):
            # VOICE-OPTIMIZATION: Strip markdown artifacts
            clean_chunk = chunk.replace("*", "").replace("#", "").replace("- ", "")
            yield clean_chunk

    def get_context_and_sources(self, query: str):
        from ingest import get_static_context
        context = get_static_context(query, k=3)
        sources = ["MongoDB Atlas Vector Search"]
        return context, sources


if __name__ == "__main__":
    service = MedicalRAGService()
    print("MedicalRAGService initialized successfully.")
    # Quick test
    answer = ""
    for chunk in service.ask_stream("I have a persistent headache and nausea"):
        answer += chunk
    print(f"Answer: {answer}")
