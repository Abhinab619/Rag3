import os
import logging
from typing import Literal
from pydantic import BaseModel, Field 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

try:
    from src.token_tracker import get_tracking_callbacks, get_current_request_id
except ImportError:
    from token_tracker import get_tracking_callbacks, get_current_request_id

# Configure logging
logger = logging.getLogger(__name__)

# Define Categories (The Output Options)
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["scheme", "service", "office", "general"] = Field(
        ...,
        description="Given a user question, choose which datasource is most relevant.",
    )

# Define the Router Class
class PipelineRouter:
    """
    Routes queries to the appropriate RAG pipeline based on intent.
    """
    
    def __init__(self, vector_store_search_func):
        self.search_func = vector_store_search_func
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key="AIzaSyCczUh_aVhX9QEnOyaD5xjmtstQGWMPk8k",
            temperature=0
        )

        # --- UPDATED STRICT PROMPT ---
        system = """You are a classification tool. 
        Classify the user's query into EXACTLY one of these categories:
        
        1. "scheme" (Eligibility, money, benefits, PM Kisan, Ayushman)
        2. "service" (Certificates, documents, forms, caste certificate)
        3. "office" (Locations, addresses, phone numbers, contact info)
        4. "general" (Greetings, other)
        
        OUTPUT RULES:
        - You MUST output ONLY the single word category name: scheme, service, office, or general
        - Do not output anything else, just the word
        - If the user asks for an address, the answer is "office".
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{question}"),
        ])
        
        self.router_chain = self.prompt | self.llm

    def get_intent(self, query: str) -> str:
        """Decides where the query should go. Rule-based first, LLM fallback for ambiguous."""
        try:
            q_lower = query.lower()
            
            # --- RULE-BASED ROUTING (no LLM call) ---
            
            # Office queries
            if any(w in q_lower for w in ["office", "block", "hospital", "address", "location", "phone", "contact", "timing", "working hours"]):
                # But if asking about a specific scheme's office, route to scheme
                if any(w in q_lower for w in ["pension", "scheme", "yojana", "kisan", "awas", "ujjwala"]):
                    return "scheme"
                return "office"
            
            # Service queries (certificates, registrations)
            if any(w in q_lower for w in ["certificate", "registration", "form", "caste", "income", "domicile", "birth", "death", "ration card"]):
                return "service"
            
            # Scheme queries (benefits, eligibility, etc.)
            if any(w in q_lower for w in [
                "scheme", "yojana", "eligible", "eligibility", "benefit", "pension",
                "subsidy", "loan", "scholarship", "kisan", "awas", "ujjwala",
                "amount", "how much", "documents required", "how to apply"
            ]):
                return "scheme"
            
            # Greetings
            if any(w in q_lower for w in ["hello", "hi", "thanks", "thank you", "bye", "help"]):
                return "general"

            # Fallback: LLM for truly ambiguous queries
            req_id = get_current_request_id()
            callbacks = get_tracking_callbacks(caller="PipelineRouter.get_intent", request_id=req_id)
            result = self.router_chain.invoke(
                {"question": query},
                config={"callbacks": callbacks}
            )
            # Extract the category from LLM response (should be a single word)
            intent = result.content.strip().lower()
            # Clean up in case of extra punctuation
            intent = ''.join(c for c in intent if c.isalpha())
            print(f"[Router] LLM Intent: {intent}")
            # Validate intent is one of the valid options
            if intent in ["scheme", "service", "office", "general"]:
                return intent
            return "general"
            
        except Exception as e:
            logger.error(f"[Router] Routing failed: {e}")
            return "general"

    def route_and_retrieve(self, query: str) -> dict:
        intent = self.get_intent(query)
        
        # Define Metadata Filters
        filter_dict = {}
        if intent == "schema":
            filter_dict = {"type": "schema"}
        elif intent == "services":
            filter_dict = {"type": "services"}
        elif intent == "offices":
            filter_dict = {"type": "office"}

        print(f"[Router] Searching with filter: {filter_dict}")
        
        context = self.search_func(query, k=5, filter=filter_dict)
        
        return {
            "context": context,
            "intent": intent
        }