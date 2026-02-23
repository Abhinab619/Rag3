import os
import time
import logging
import hashlib
from typing import Dict, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

try:
    from src.token_tracker import get_tracking_callbacks
except ImportError:
    from token_tracker import get_tracking_callbacks

# Import your brains
try:
    from src.vectorstore import search_faiss
except ImportError:
    search_faiss = None

try:
    from src.sqldb import search_live_events
except ImportError:
    search_live_events = None

# --- QUIET MODE CONFIGURATION ---
# 1. Set global logging to WARNING (hides INFO messages)
logging.basicConfig(
    level=logging.WARNING, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# 2. Silence the specific noisy libraries you saw in your terminal
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("src.embedding").setLevel(logging.WARNING)
logging.getLogger("src.vectorstore").setLevel(logging.WARNING)
logging.getLogger("__main__").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

load_dotenv()

# Simple in-memory cache
_response_cache: Dict[str, dict] = {}
CACHE_MAX_SIZE = 100

def _get_cache_key(query: str, context: str) -> str:
    """Generate a cache key from query and context."""
    combined = f"{query}:{context[:500]}"
    return hashlib.md5(combined.encode()).hexdigest()

class SmartAgent:
    def __init__(self, use_cache: bool = True, max_retries: int = 3):
        """
        Initialize the Smart Agent with caching and anti-repetition logic.
        """
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key="AIzaSyCczUh_aVhX9QEnOyaD5xjmtstQGWMPk8k",
            temperature=0.0
        )
        
        self.use_cache = use_cache
        self.max_retries = max_retries
        
        self.event_keywords = {
            "tomorrow", "today", "date", "when", "event", "camp", 
            "happening", "schedule", "upcoming", "next week", "this week"
        } 

    def _classify_query(self, query: str) -> str:
        """Classify query intent - 'event' or 'scheme'."""
        query_lower = query.lower()
        if any(word in query_lower for word in self.event_keywords):
            return "event"
        return "scheme"

    def _call_llm_with_retry(self, chain, inputs: dict, retries: int = None) -> Optional[str]:
        """Call LLM with exponential backoff retry logic."""
        retries = retries or self.max_retries
        for attempt in range(retries):
            try:
                callbacks = get_tracking_callbacks(caller="SmartAgent._call_llm_with_retry")
                response = chain.invoke(inputs, config={"callbacks": callbacks})
                return response.content
            except Exception as e:
                wait_time = 2 ** attempt
                # logger.warning(f"LLM call failed: {e}") # Uncomment to debug errors
                if attempt < retries - 1:
                    time.sleep(wait_time)
                else:
                    return "I'm having trouble connecting to the brain right now. Please try again."

    def get_answer(self, query: str, district: str = None) -> dict:
        """
        Get an answer based on the user's query.
        """
        query_type = self._classify_query(query)
        context = ""
        source = ""
        
        if query_type == "event" and search_live_events:
            try:
                raw_data = search_live_events(query, district)
                context = f"LIVE EVENTS DATA:\n{raw_data}"
                source = "LiveDB"
            except Exception as e:
                logger.error(f"Database query failed: {e}")
                context = "No event data found."
                source = "Error"
        
        else:
            if search_faiss:
                raw_data = search_faiss(query, k=5)
                context = f"OFFICIAL SCHEME RULES:\n{raw_data}"
                source = "PDF"
            else:
                context = "No PDF search module found."
                source = "System"

        # Check Cache
        if self.use_cache:
            cache_key = _get_cache_key(query, context)
            if cache_key in _response_cache:
                return _response_cache[cache_key]

        # Construct Prompt
        system_prompt = """You are a helpful government caseworker.
RULES:
1. Answer ONLY based on the context provided.
2. If the answer is not in the context, say "I don't find that information in the documents."
3. Keep it short, bulleted, and clear.
4. Do not repeat the same closing sentence multiple times.

CONTEXT:
{context}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

        chain = prompt | self.llm
        
        answer = self._call_llm_with_retry(chain, {"context": context, "question": query})
        
        # Double check: Remove repeated closing phrases if they slip through
        if answer and "Feel free to ask" in answer:
            parts = answer.split("Feel free to ask")
            if len(parts) > 2:
                # Keep only the first occurrence
                answer = parts[0] + "Feel free to ask" + parts[1]

        result = {
            "answer": answer,
            "source": source
        }
        
        # Update Cache
        if self.use_cache:
            if len(_response_cache) >= CACHE_MAX_SIZE:
                oldest_key = next(iter(_response_cache))
                del _response_cache[oldest_key]
            _response_cache[cache_key] = result
        
        return result

# Entry point for testing
if __name__ == "__main__":
    agent = SmartAgent()
    print("\n--- Bot Ready (Quiet Mode) ---")
    
    while True:
        user_q = input("\nYou: ")
        if user_q.lower() in ["quit", "exit"]:
            break
        
        response = agent.get_answer(user_q)
        print(f"Bot: {response['answer']}")