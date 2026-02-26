import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
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
    Dynamic RAG service with merged retrieval:
    - Static knowledge base (health_book.txt vectorDB)
    - Per-user conversation history (dynamic vectorDB)
    - Severity-driven follow-up symptom loop
    """

    def __init__(self, persist_directory: str = "vectordb"):
        # 1. Initialize LLM (Llama 3 via Groq)
        self.llm = ChatGroq(
            temperature=0.1,
            model_name="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        # 2. Initialize Static Vector DB
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        index_exists = os.path.exists(persist_directory) and any(os.listdir(persist_directory))
        if not index_exists:
            print(f"⚠️ VectorDB at {persist_directory} not found or empty. Initializing from health_book.txt...")
            from ingest import ingest_docs
            health_file = "healt_book.txt"
            if os.path.exists(health_file):
                ingest_docs(health_file, persist_directory)
            else:
                print(f"❌ Error: {health_file} not found. RAG service will have no medical context.")

        self.vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name="medical_docs"
        )
        self.static_retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})

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
- After answering, you MUST ask exactly ONE specific follow-up question targeting the next most relevant symptom to narrow down possible conditions and diseases possible accordinh to chroma.sqlite under  directory  vectordb + vectordb_users.
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
        Merged RAG retrieval: static knowledge + per-user history.
        Streams the response with follow-up questions based on severity/turn count.
        """
        if chat_history is None:
            chat_history = []
            
        # 1. Retrieve from STATIC knowledge base
        static_docs = self.static_retriever.invoke(query)
        static_context = "\n\n".join([d.page_content for d in static_docs])
        
        # 2. Retrieve from PER-USER vector DB (if available)
        user_context_from_vectordb = ""
        if user_id and user_id != "anonymous":
            try:
                from ingest import get_user_retriever
                user_retriever = get_user_retriever(user_id, k=3)
                if user_retriever:
                    user_docs = user_retriever.invoke(query)
                    user_context_from_vectordb = "\n\n".join([d.page_content for d in user_docs])
            except Exception as e:
                print(f"⚠️ User retriever error: {e}")

        # 3. Merge user context: MongoDB conversation context + vectorDB user context
        #    MongoDB context (recent turns) takes priority as it's more detailed
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
        docs = self.static_retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])
        sources = list(set([d.metadata.get("source", "Unknown") for d in docs]))
        return context, sources





if __name__ == "__main__":
    service = MedicalRAGService()
    print("MedicalRAGService initialized successfully.")
    # Quick test
    answer = ""
    for chunk in service.ask_stream("I have a persistent headache and nausea"):
        answer += chunk
    print(f"Answer: {answer}")
