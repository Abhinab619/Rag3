import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

try:
    from src.grading import RAGGrader, ClaimVerifier
    from src.router import PipelineRouter
    from src.vectorstore import (
        search_faiss, search_faiss_with_scores,
        hybrid_search, format_docs_to_string
    )
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.getLogger("src.token_tracker").setLevel(logging.INFO)

MODEL_NAME = "llama-3.3-70b-versatile"      
CONFIDENCE_THRESHOLD = 1.2                   


class DeepLinker:
    def __init__(self):
        self.data_map = {}
        data_path = BASE_DIR / "data"

        for json_file in data_path.glob("**/*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data = [data]

                for item in data:
                    if "scheme_id" in item:
                        self.data_map[item["scheme_id"]] = item
                    if "office_id" in item:
                        self.data_map[item["office_id"]] = item
                    if "service_id" in item:
                        self.data_map[item["service_id"]] = item
            except Exception:
                continue

        print(f"Linker Ready: Indexed {len(self.data_map)} records.")
 
    def _format_office(self, oid: str) -> str:
        """Format an office record as full plain text."""
        office = self.data_map.get(oid)
        if not office:
            return f"Office {oid}: Not found in database"
        lines = []
        lines.append(f"Office Name: {office.get('office_name', 'N/A')}")
        if office.get("district"):
            lines.append(f"District: {office['district']}")
        if office.get("address"):
            lines.append(f"Address: {office['address']}")
        if office.get("contact_number"):
            lines.append(f"Contact: {office['contact_number']}")
        if office.get("working_hours"):
            lines.append(f"Working Hours: {office['working_hours']}")
        return "\n".join(lines)

    def _format_service(self, sid: str) -> str:
        """Format a service record as full plain text."""
        service = self.data_map.get(sid)
        if not service:
            return f"Service {sid}: Not found"
        name = service.get("service_name") or service.get("service_name_en") or "N/A"
        lines = []
        lines.append(f"Service Name: {name}")
        if service.get("service_type"):
            lines.append(f"Type: {service['service_type']}")
        if service.get("description"):
            lines.append(f"Description: {service['description']}")
        # Processing time
        sla = service.get("sla_processing_time")
        if sla:
            std = sla.get("standard_days")
            tatkal = sla.get("tatkal_days")
            time_parts = []
            if std is not None:
                time_parts.append(f"Standard: {std} days")
            if tatkal is not None:
                time_parts.append(f"Tatkal: {tatkal} days")
            if time_parts:
                lines.append(f"Processing Time: {', '.join(time_parts)}")
        # Documents required for this service
        docs = service.get("documents_required")
        if docs:
            doc_names = []
            for d in docs:
                doc_name = d.get("doc_name", "N/A")
                mandatory = "Required" if d.get("is_mandatory") else "Optional"
                doc_names.append(f"{doc_name} ({mandatory})")
            lines.append(f"Documents Needed: {'; '.join(doc_names)}")
        # Responsible offices (resolved to names)
        office_ids = service.get("responsible_office_ids") or []
        if office_ids:
            office_names = []
            for o_id in office_ids:
                o = self.data_map.get(o_id)
                if o:
                    office_names.append(f"{o.get('office_name', o_id)} ({o.get('address', 'N/A')})")
                else:
                    office_names.append(o_id)
            lines.append(f"Available At: {'; '.join(office_names)}")
        return "\n".join(lines)

    def enrich_from_scheme(self, scheme_id: str, focus: str = "scheme") -> str:
        """Return only the relevant sections based on user intent (focus)."""
        if scheme_id not in self.data_map:
            return "[ERROR: Scheme not found]"

        scheme = self.data_map[scheme_id]
        lines = []

        # Always include scheme name
        lines.append(f"SCHEME: {scheme.get('scheme_name', 'N/A')} (ID: {scheme_id})")

        # --- Benefits (for: scheme, benefits) ---
        if focus in ("scheme", "benefits"):
            benefits = scheme.get("benefits") or {}
            if benefits:
                lines.append(f"Benefit Type: {benefits.get('benefit_type', 'N/A')}")
                if benefits.get("max_amount_inr"):
                    lines.append(f"Max Amount: ₹{benefits['max_amount_inr']}")
                if benefits.get("frequency"):
                    lines.append(f"Frequency: {benefits['frequency']}")
                if benefits.get("description"):
                    lines.append(f"Details: {benefits['description']}")

        # --- Eligibility (for: scheme, eligibility) ---
        if focus in ("scheme", "eligibility"):
            elig = scheme.get("eligibility_criteria") or {}
            if elig:
                conditions = elig.get("special_conditions") or []
                age_parts = []
                if elig.get("age_min") is not None:
                    age_parts.append(f"Min Age: {elig['age_min']}")
                if elig.get("age_max") is not None:
                    age_parts.append(f"Max Age: {elig['age_max']}")
                if age_parts:
                    lines.append(f"Eligibility Age: {', '.join(age_parts)}")
                if elig.get("residency_requirement"):
                    lines.append(f"Residency: {elig['residency_requirement']}")
                if elig.get("employment_status"):
                    lines.append(f"Employment: {elig['employment_status']}")
                if conditions:
                    lines.append(f"Conditions: {'; '.join(conditions)}")

        # --- Office (for: scheme, office) ---
        if focus in ("scheme", "office"):
            managing_oid = scheme.get("managing_office_id")
            if managing_oid:
                lines.append(f"Managing Office:\n{self._format_office(managing_oid)}")

        # --- Application Process (for: scheme, office) ---
        if focus in ("scheme", "office"):
            app_process = scheme.get("application_process") or {}
            stages = app_process.get("stages") or []
            if stages:
                lines.append("Application Steps:")
                for step in stages:
                    step_text = f"  Step {step.get('step', '?')}: {step.get('action', 'N/A')}"
                    oid = step.get("designated_office_id")
                    if oid:
                        step_text += f"\n    {self._format_office(oid)}"
                    lines.append(step_text)

        # --- Documents (for: scheme, documents) ---
        if focus in ("scheme", "documents"):
            docs = scheme.get("documents_required") or []
            if docs:
                lines.append("Documents Required:")
                for doc in docs:
                    mandatory = "Required" if doc.get("is_mandatory") else "Optional"
                    doc_text = f"  - {doc.get('doc_name', 'N/A')} ({mandatory})"
                    sid = doc.get("linked_service_id")
                    if sid:
                        doc_text += f"\n    {self._format_service(sid)}"
                    lines.append(doc_text)

        # --- Services (for: service) ---
        if focus == "service":
            service_ids = set()
            if scheme.get("application_service_id"):
                service_ids.add(scheme["application_service_id"])
            for doc in scheme.get("documents_required", []):
                if doc.get("linked_service_id"):
                    service_ids.add(doc["linked_service_id"])
            if service_ids:
                lines.append("Related Services:")
                for sid in service_ids:
                    lines.append(f"  {self._format_service(sid)}")
            else:
                lines.append("No linked services found for this scheme.")

        return "\n".join(lines)
    
class StaticBot:
    def __init__(self, use_hybrid_search: bool = False, max_history: int = 5, enable_claim_verification: bool = False):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key="AIzaSyCczUh_aVhX9QEnOyaD5xjmtstQGWMPk8k",
            temperature=0.0
        )
        self.max_history=max_history
        self.chat_history=[]     
        self.last_scheme_id = None

        try:
            self.grader = RAGGrader()
            print("RAGGrader initialized.")
        except Exception as e:
            print(f"Warning: RAGGrader failed to initialize: {e}")
            self.grader = None

        self.claim_verifier = None
        if enable_claim_verification:
            try:
                self.claim_verifier = ClaimVerifier()
                print("ClaimVerifier initialized.")
            except Exception as e:
                print(f"Warning: ClaimVerifier failed to initialize: {e}")
                self.claim_verifier = None

        search_func = hybrid_search if use_hybrid_search else search_faiss
        self.pipeline = PipelineRouter(search_func)
        self.linker = DeepLinker()

    def clear_memory(self):
        """ clear memory history"""
        self.chat_history=[]
        print("Bot history clear")  
         
    def detect_focus(self, query: str) -> str:
        q = query.lower()
        if "eligible" in q or "eligibility" in q:
            return "eligibility"
        if "document" in q or "certificate" in q or "certificates" in q:
            return "documents"
        if "service" in q:
            return "service"
        if "benefit" in q or "amount" in q or "how much" in q:
            return "benefits"
        if "where" in q or "office" in q or "address" in q or "contact" in q:
            return "office"
        return "scheme"

    def is_too_generic(self, query: str) -> bool:
        q = query.lower().strip()
        generic_patterns = [
            r"^tell me about scheme$",
            r"^tell me about schemes$",
            r"^tell me about government scheme$",
            r"^scheme$",
            r"^government scheme$",
            r"^tell me about$"
        ]
        return any(re.match(p, q) for p in generic_patterns)

    def is_generic_scheme_query(self, query: str) -> bool:
        q = query.lower().strip()
        return q in [
            "tell me about scheme",
            "tell me about schemes",
            "scheme",
            "government scheme",
            "govt scheme"
        ]

    def _generate_query_variants(self, query: str) -> List[str]:
        """Rule-based variants — zero extra LLM calls."""
        variants = []
        q = query.strip()
        if not q.endswith("?"):
            variants.append(q + "?")
        if "scheme" not in q.lower():
            variants.append(q + " government scheme")
        return variants[:2]

 
    def _multi_query_retrieve(self, user_query: str) -> dict:
        """Retrieve using original + optional rephrased queries."""
    
        main_result = self.pipeline.route_and_retrieve(user_query)
        all_docs = list(main_result["context"]) if main_result["context"] else []

        # Only use variants if no scheme memory yet
        if not self.last_scheme_id:
            variants = self._generate_query_variants(user_query)
            print(f"[Multi-Query] Generated {len(variants)} variant(s)")
        else:
            variants = []
            print("[Multi-Query] Skipped (scheme memory active)")

        seen_contents = {doc.page_content for doc in all_docs}

        for variant in variants:
            try:
                variant_docs = search_faiss(variant, k=3)
                for doc in variant_docs:
                    if doc.page_content not in seen_contents:
                        all_docs.append(doc)
                        seen_contents.add(doc.page_content)
            except Exception:
                continue

        print(f"[Multi-Query] Total unique docs: {len(all_docs)}")

        return {
            "context": all_docs,
            "intent": main_result["intent"]
        }


    def _check_retrieval_confidence(self, query: str) -> Tuple[bool, float]:
        """
        Check if the top FAISS result is within our confidence threshold.
        FAISS L2 distance: lower = more similar.
        Returns (is_confident, score).
        """
        results = search_faiss_with_scores(query, k=1)
        if not results:
            return False, float('inf')

        doc, score = results[0]
        is_confident = score <= CONFIDENCE_THRESHOLD
        print(
            f"[Confidence] Top score: {score:.4f} "
            f"(threshold: {CONFIDENCE_THRESHOLD}) "
            f"-> {'PASS' if is_confident else 'FAIL'}"
        )
        return is_confident, score
        


    def _verify_answer(self, answer: str, context: str, query: str) -> Tuple[str, bool]:
        """
        Verify the generated answer is grounded in context using RAGGrader.
        If hallucination is detected, RETRY with a stricter prompt.
        If retry also fails, return the raw data directly (zero hallucination).
        Returns (final_answer, was_grounded).
        """
        if not self.grader:
            return answer, True

        is_grounded = self.grader.is_answer_grounded(
            answer=answer,
            context=context,
            query=query
        )

        if is_grounded:
            print("[Verification] Answer is grounded in context.")
            return answer, True

       
        print("[Verification] FAILED. Retrying with constrained prompt...")
        strict_answer = self._regenerate_strict(context, query)

        is_grounded_retry = self.grader.is_answer_grounded(
            answer=strict_answer,
            context=context,
            query=query
        )

        if is_grounded_retry:
            print("[Verification] Retry answer is grounded.")
            return strict_answer, True

       
        print("[Verification] Retry also failed. Returning raw data fallback.")
        fallback = (
            "Here is the information directly from the database:\n\n"
            f"{context}\n\n"
            "*This is raw data from the database. Please verify with official sources.*"
        )
        return fallback, False

    def _regenerate_strict(self, context: str, query: str) -> str:
        """Regenerate answer with an ultra-strict extraction-only prompt."""
        context_escaped = context.replace("{", "{{").replace("}", "}}")
        strict_prompt = f"""You are a data extraction tool. Extract ONLY the relevant facts from the DATA below to answer the question.

RULES:
- Copy facts EXACTLY as they appear in the DATA. Do not rephrase.
- Do NOT add any information not present in the DATA.
- If the answer is not in the DATA, say "Not available in database".
- Format as bullet points with **bold** labels.

DATA:
{context_escaped}

QUESTION: {query}"""

        try:
            messages = [("human", strict_prompt)]
            response = (
                ChatPromptTemplate.from_messages(messages) | self.llm
            ).invoke({})
            return response.content
        except Exception as e:
            return f"Error generating answer: {e}"


    def contains_scheme_name(self, query: str):
        """
        Return matching scheme_id if user mentions a scheme name.
        PASS 1: Match against original query (exact, acronym, keyword).
        PASS 2: Only if PASS 1 fails, try with abbreviation expansion.
        """
        q = query.lower()

        result = self._match_scheme_in_text(q)
        if result:
            return result

        abbreviations = {
            "pm": "pradhan mantri",
            "cm": "mukhyamantri",
        }
        q_expanded = q
        for abbr, full in abbreviations.items():
            q_expanded = re.sub(rf'\b{abbr}\b', full, q_expanded)

        if q_expanded != q:
            result = self._match_scheme_in_text(q_expanded)
            if result:
                return result

        return None

    def _match_scheme_in_text(self, q: str):
        """Core matching logic: exact → acronym → keyword scoring."""
        best_match = None
        best_score = 0

        for item in self.linker.data_map.values():
            if "scheme_id" not in item or "scheme_name" not in item:
                continue

            scheme_name = item["scheme_name"].lower()
            clean_name = scheme_name.split("(")[0].strip()

            if clean_name in q:
                return item["scheme_id"]

            query_words = [w for w in q.split() if len(w) >= 2 and w not in {
                "tell", "me", "about", "what", "is", "the", "how", "to",
                "can", "do", "get", "for", "and", "of", "in", "my"
            }]
            query_phrase = " ".join(query_words)
            if len(query_phrase) >= 4 and query_phrase in clean_name:
                return item["scheme_id"]

            if "(" in scheme_name and ")" in scheme_name:
                short_name = scheme_name.split("(")[1].split(")")[0]
                
                is_acronym = short_name.replace("-", "").replace(" ", "").isupper() and len(short_name) <= 10
                if is_acronym and short_name.lower() in q:
                    return item["scheme_id"]

            stop_words = {"yojana", "scheme", "abhiyan", "the", "of", "for", "and", "a", "an"}
            scheme_words = [w for w in clean_name.split() if w not in stop_words and len(w) >= 2]

            if scheme_words:
                matches = sum(1 for w in scheme_words if w in q)
                score = matches / len(scheme_words)

                if matches >= 2 and score >= 0.5 and score > best_score:
                    best_score = score
                    best_match = item["scheme_id"]

        return best_match

    def _is_greeting_or_chitchat(self, query: str) -> bool:
        q = query.lower().strip().rstrip("!?.,")
        greetings = [
            "hello", "hi", "hey", "hii", "hiii", "good morning", "good afternoon",
            "good evening", "good night", "namaste", "namaskar", "thanks", "thank you",
            "thankyou", "ok", "okay", "bye", "goodbye", "good bye", "welcome",
            "how are you", "what's up", "sup", "yo", "hola"
        ]
        return q in greetings

    def _is_follow_up(self, query: str) -> bool:
        """
        Detect if a query is a follow-up about the current scheme,
        or a completely new topic. Logic:
        
        1. If query has ASPECT keywords (documents, eligibility, etc.) → follow-up
           (regardless of length or phrasing)
        2. If query introduces a NEW SUBJECT ("tell me about X") → NOT follow-up
           (unless X is itself an aspect like "eligibility")
        3. Very short queries (≤ 4 words) with no clear subject → follow-up
        4. Everything else → NOT follow-up (safer to search fresh)
        """
        q = query.lower().strip()

        aspect_keywords = [
            "document", "documents", "eligible", "eligibility", "criteria",
            "apply", "application", "process", "step", "steps",
            "office", "address", "contact", "phone", "number",
            "benefit", "benefits", "amount", "money", "how much",
            "where", "when", "required", "needed", "mandatory",
            "certificate", "qualification", "age", "limit",
            "ration card", "aadhaar", "aadhar", "income",
            "i don't have", "i dont have", "do i need", "can i get",
            "am i", "how to", "how do i"
        ]

        has_aspect = any(kw in q for kw in aspect_keywords)

        new_topic_starters = [
            "tell me about", "what is", "explain", "describe",
            "information about", "details about", "info about",
            "i want to know about", "know about"
        ]

        has_new_topic_starter = any(starter in q for starter in new_topic_starters)

        if has_aspect:
            return True

        if has_new_topic_starter:
            return False

        if len(q.split()) <= 4:
            return True

        return False


    def chat(self, user_query: str) -> str:

        return self._chat_inner(user_query)

    def _chat_inner(self, user_query: str) -> str:
        print(f"\n{'='*50}")
        print(f"Processing: '{user_query}'")
        print(f"{'='*50}")

        if self._is_greeting_or_chitchat(user_query):
            return (
                "Hello! 👋 I'm a Government Scheme Assistant for Bihar.\n\n"
                "I can help you with:\n"
                "- **Government Schemes** (eligibility, benefits, documents, application process)\n"
                "- **Office Information** (address, contact, working hours)\n"
                "- **Services** (certificates, registrations)\n\n"
                "Please ask about a specific scheme, office, or service to get started!"
            )

        if self.is_too_generic(user_query):
            return (
                "Please specify the scheme name or the purpose "
                "(for example: health, farmer, marriage, education)."
            )

        if self.is_generic_scheme_query(user_query):
            return (
                "Multiple government schemes are available. "
                "Please specify the scheme name or purpose "
                "(e.g., health, education, farmer)."
            )
        
        focus = self.detect_focus(user_query)

        
        is_follow_up = self.last_scheme_id and self._is_follow_up(user_query)
        direct_scheme_id = self.contains_scheme_name(user_query)

        
        if is_follow_up and not direct_scheme_id:
            scheme_id = self.last_scheme_id
            print(f"[Memory] Follow-up detected, using last scheme: {scheme_id}")

        elif is_follow_up and direct_scheme_id:
            
            scheme_id = direct_scheme_id
            self.last_scheme_id = scheme_id
            print(f"[Direct Match + Follow-up] Switching to: {scheme_id}")

        elif direct_scheme_id:
            scheme_id = direct_scheme_id
            self.last_scheme_id = scheme_id
            print(f"[Direct Match] Found scheme: {scheme_id} — skipping vector search.")

        else:
            combined_query = user_query

            is_confident, _ = self._check_retrieval_confidence(combined_query)

            if not is_confident:
                return (
                    "I'm not confident I have relevant information for your query. "
                    "Please try being more specific about the scheme or service name."
                )

            route_result = self._multi_query_retrieve(combined_query)
            raw_docs = route_result["context"]

            if not raw_docs:
                return "I found no records matching your query."

            scheme_id = raw_docs[0].metadata.get("scheme_id")
            self.last_scheme_id = scheme_id

            if not scheme_id:
                return "Scheme details are not available. Please specify the scheme name."

            context_str = (
                format_docs_to_string(raw_docs) if isinstance(raw_docs, list)
                else str(raw_docs)
            )

            if self.grader:
                is_relevant = self.grader.is_document_relevant(
                    document_text=context_str,
                    question=user_query
                )
                if not is_relevant:
                    return (
                        "I found some data, but it doesn't appear relevant "
                        "to your question. Please try rephrasing."
                    )
                print("[Relevance] Retrieved docs are relevant.")


        final_context_str = self.linker.enrich_from_scheme(scheme_id, focus=focus)
        final_context_escaped = final_context_str.replace("{", "{{").replace("}", "}}")


        """History """
        history_str = ""
        if self.chat_history:
            history_str = "### PREVIOUS CONTEXT:\n"
            for turn in self.chat_history[-5:]:
                scheme_info = f" for scheme: {turn.get('scheme_name', 'Unknown')} ({turn.get('scheme_id', 'N/A')})"
                history_str += f"User asked about: {turn['summary']}{scheme_info}\n"

        system_prompt = """You are a government scheme data FORMATTER. You convert raw database records into clean, readable answers.

ABSOLUTE RESTRICTIONS:
- You are a FORMATTER, not an expert. You have ZERO knowledge of any government scheme.
- Every single fact in your answer MUST have a direct source in the DATA section below.
- Do NOT use phrases like "you can also", "it is recommended", "typically", "usually", "generally".
- Do NOT provide URLs, website names, portal links, or helpline numbers unless they appear EXACTLY in DATA.
- Do NOT rephrase monetary amounts. If DATA says ₹6000, write ₹6000. Do NOT convert to words.
- Do NOT infer, assume, or extrapolate. If DATA does not contain a piece of information, say "Not available in database".
- Do NOT add eligibility conditions, documents, or steps that are not listed in DATA.

FORMAT RULES:
- NEVER output raw JSON, code blocks, or data dumps.
- ONLY output a clean, human-readable formatted answer.
- Use markdown: **bold** for labels/headings, bullet points for lists.
- Cite relevant source IDs (e.g., Scheme ID, Office ID) where appropriate.
- APPLICATION STEPS: For every step in the 'application_process', you MUST check if a 'designated_office_id' is mentioned. If so, find that Office ID's Name and Address in the DATA section and include it next to that specific step.

HISTORY USAGE: Always check the conversation history to understand follow-up questions (e.g., "I don't have a ration card" means they need it for the scheme discussed previously).

REMEMBER: If it is NOT in the DATA, it does NOT exist. Never fill gaps with your own knowledge."""


        human_prompt = f"""Answer the user's question using ONLY the data below.

USER INTENT: {focus}
{history_str}
FORMAT RULES:
- If intent is "scheme": Present **Scheme Name**, **Benefits**, **Eligibility Criteria**, **Application Process** (Listing Name and Address for each designated office), **Managing Office**, and **Required Documents**. Cite the scheme ID.
- If intent is "documents": List ONLY the required documents as bullet points.
- If intent is "office": Give ONLY **Office Name**, **Address**, **Working Hours**, and **Contact Number**. Cite the office ID.
- If intent is "eligibility": List the eligibility criteria and ask the user relevant questions. Do NOT decide eligibility for them.

DATA:
{final_context_escaped}

QUESTION: {user_query}

Provide a clean, well-formatted answer. Cite source IDs. Do NOT output raw JSON."""

        messages = [("system", system_prompt), ("human", human_prompt)]

        try:
            # Remove callbacks and tracker usage
            response = (
                ChatPromptTemplate.from_messages(messages) | self.llm
            ).invoke({})
            answer = response.content
        except Exception as e:
            return f"System Error: {e}"

        answer, was_grounded = self._verify_answer(
            answer=answer,
            context=final_context_str,
            query=user_query
        )

        if was_grounded and self.claim_verifier:
            print("[ClaimVerifier] Running claim-level verification...")
            claim_result = self.claim_verifier.verify_claims(answer, final_context_str)
            if not claim_result["all_passed"] and claim_result["failed_claims"]:
                failed = claim_result["failed_claims"]
                print(f"[ClaimVerifier] {len(failed)} claim(s) could not be verified.")
                answer += (
                    "\n\n---\n"
                    "**⚠️ The following could not be verified against the database:**\n"
                )
                for fc in failed:
                    answer += f"- {fc}\n"
            else:
                print("[ClaimVerifier] All claims verified.")

        scheme_name = self.linker.data_map.get(scheme_id, {}).get("scheme_name", "Unknown")
        self.chat_history.append({
            "summary": focus,
            "scheme_id": scheme_id,
            "scheme_name": scheme_name
        })

        # Keep only last N turns
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]

        
        return answer


def main():
    print("SCHEME ASSISTANT (ANTI-HALLUCINATION MODE)")
    print(f"Model: {MODEL_NAME}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print("-" * 50)

    try:
        bot = StaticBot()
    except Exception as e:
        print(e)
        return

    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        if q:
            print(f"Bot: {bot.chat(q)}")
        if q.lower() == "clear":
            bot.chat_history.clear()
            bot.last_scheme_id = None
            print("Bot: Memory cleared.")
            continue


if __name__ == "__main__":
    main()