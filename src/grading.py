from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import os

try:
    from src.token_tracker import get_tracking_callbacks, get_current_request_id
except ImportError:
    from token_tracker import get_tracking_callbacks, get_current_request_id

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeRelevance(BaseModel):
    """Binary score for document relevance to query."""
    binary_score: str = Field(description="Document is relevant to the query, 'yes' or 'no'")

class RAGGrader:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key="AIzaSyCczUh_aVhX9QEnOyaD5xjmtstQGWMPk8k",
            temperature=0.0
        )

        self.hallucination_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key="AIzaSyCczUh_aVhX9QEnOyaD5xjmtstQGWMPk8k",
            temperature=0.0
        )
         
        self.hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a STRICT fact-checker for a government chatbot.
            
Your job: Check if EVERY claim in the 'generation' is EXPLICITLY stated in the 'facts'.

RULES:
1. If the generation mentions a scheme name, that EXACT scheme name must appear in the facts.
2. If the generation lists documents required, those documents must be listed in the facts.
3. The generation must NOT contain any information from your internal knowledge.
4. If the generation says "Not available" or admits uncertainty, that is GROUNDED.
5. If the generation provides specific details (addresses, phone numbers, amounts) not in facts, it is NOT grounded.
6. Check NUMERICAL VALUES carefully — amounts, ages, percentages must match EXACTLY.
7. Check OFFICE NAMES and ADDRESSES — they must match the facts exactly.

Answer 'yes' ONLY if every factual claim can be traced to the facts below.
Answer 'no' if ANY detail seems to come from outside the facts.

RESPOND WITH ONLY 'yes' or 'no', nothing else.

facts: {documents}

generation: {generation}
"""),
            ("human", "Is every claim in the answer explicitly supported by the facts above?")
        ])
        
        self.hallucination_chain = self.hallucination_prompt | self.hallucination_llm

        self.relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a grader assessing the relevance of a retrieved document to a user question.
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
            
            Retrieved document: {document}
            User question: {question}
            
            RESPOND WITH ONLY 'yes' or 'no', nothing else.
            """),
            ("human", "Is the document relevant?")
        ])
        
        self.relevance_chain = self.relevance_prompt | self.llm

    def is_answer_grounded(self, answer: str, context: str, query: str = "") -> bool:
        """Check if the answer is supported by the context with pre-checks."""
        
        # Empty context 
        if not context or len(context.strip()) < 50:
            return False  # No real context, can't be grounded
        
        # Topic keyword check 
        # Extract key terms from the query and check if ANY appear in context
        if query:
            query_words = [w.lower() for w in query.split() if len(w) > 3]
            context_lower = context.lower()
            matches = sum(1 for w in query_words if w in context_lower)
            if matches == 0:
                # No keywords from the question appear in context - definitely hallucinating
                return False
        
        # LLM CHECK 
        try:
            req_id = get_current_request_id()
            callbacks = get_tracking_callbacks(caller="RAGGrader.is_answer_grounded", request_id=req_id)
            score = self.hallucination_chain.invoke(
                {"documents": context, "generation": answer},
                config={"callbacks": callbacks}
            )
            # Parse plain text response
            response_text = score.content.strip().lower()
            return "yes" in response_text
        except Exception:
            return False  # Fail CLOSED now (safer)

    def is_document_relevant(self, document_text: str, question: str) -> bool:
        """Check if a document is relevant to the question."""
        try:
            req_id = get_current_request_id()
            callbacks = get_tracking_callbacks(caller="RAGGrader.is_document_relevant", request_id=req_id)
            score = self.relevance_chain.invoke(
                {"document": document_text, "question": question},
                config={"callbacks": callbacks}
            )
            # Parse plain text response
            response_text = score.content.strip().lower()
            return "yes" in response_text
        except Exception:
            return True


class ClaimVerifier:
    """
    Extracts individual claims from an answer and verifies each one
    against the source context independently. Catches subtle hallucinations
    that whole-answer grading misses.
    """

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key="AIzaSyCczUh_aVhX9QEnOyaD5xjmtstQGWMPk8k",
            temperature=0.0
        )

        self.extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract every factual claim from this answer as a JSON list of strings.
Each claim should be a single, atomic fact (one number, one name, one condition).
Do NOT include formatting instructions or subjective statements.
Example: ["PM Kisan gives ₹6000 per year", "Aadhaar card is required", "Apply at Block Office Patna"]
Output ONLY the JSON list, nothing else."""),
            ("human", "{answer}")
        ])

        self.verify_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fact-checker. Is this specific claim EXPLICITLY supported by the source data below?
            
Source data:
{context}

Claim to verify: {claim}

Answer ONLY 'yes' if the claim can be directly traced to the source data.
Answer 'no' if the claim contains ANY detail not in the source data."""),
            ("human", "Is the claim supported? Answer 'yes' or 'no'.")
        ])

    def verify_claims(self, answer: str, context: str) -> dict:
        """
        Returns dict with:
          - 'verified_claims': list of claims that passed
          - 'failed_claims': list of claims that failed
          - 'all_passed': bool
        """
        import json as _json

        req_id = get_current_request_id()

        try:
            extract_callbacks = get_tracking_callbacks(caller="ClaimVerifier.extract_claims", request_id=req_id)
            claims_response = (self.extract_prompt | self.llm).invoke(
                {"answer": answer},
                config={"callbacks": extract_callbacks}
            )
            claims_text = claims_response.content.strip()
            if claims_text.startswith("```"):
                claims_text = claims_text.split("```")[1]
                if claims_text.startswith("json"):
                    claims_text = claims_text[4:]
            claims = _json.loads(claims_text)
        except Exception:
            return {"verified_claims": [], "failed_claims": [], "all_passed": True}

        if not claims or not isinstance(claims, list):
            return {"verified_claims": [], "failed_claims": [], "all_passed": True}

        verified = []
        failed = []

        for claim in claims:
            try:
                verify_callbacks = get_tracking_callbacks(caller="ClaimVerifier.verify_claim", request_id=req_id)
                result = (self.verify_prompt | self.llm).invoke(
                    {"context": context, "claim": claim},
                    config={"callbacks": verify_callbacks}
                )
                if "yes" in result.content.lower():
                    verified.append(claim)
                else:
                    failed.append(claim)
            except Exception:
                failed.append(claim)

        return {
            "verified_claims": verified,
            "failed_claims": failed,
            "all_passed": len(failed) == 0
        }
