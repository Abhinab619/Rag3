import json
import os
import logging
from typing import List, Dict, Any
from pathlib import Path
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class LinkedLoader:
    """
    A specialized loader that joins Scheme <-> Service <-> Office data
    into a single, fully enriched text block for RAG.
    """
    def __init__(self, data_dir: str):
        self.data_path = Path(data_dir)
        self.schemes = []  # Fixed: changed 'schem' to 'schemes'
        self.services_map = {}
        self.offices_map = {}

    def _load_json(self, filename: str) -> List[Dict]:
        """Safe JSON loader."""
        path = self.data_path / filename
        if not path.exists():
            logger.warning(f"Missing file: {filename}")
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return []

    def load_data(self):
        """Load and index the raw data."""
        # Load Raw Files
        raw_schemes = self._load_json("schemes.json") 
        raw_services = self._load_json("services.json")
        raw_offices = self._load_json("offices.json")

        # Index Services and Offices by ID for O(1) lookup
        self.services_map = {srv['service_id']: srv for srv in raw_services}
        self.offices_map = {off['office_id']: off for off in raw_offices}
        self.schemes = raw_schemes
        
        logger.info(f"Loaded {len(self.schemes)} schemes, {len(self.services_map)} services, {len(self.offices_map)} offices.")

    def _format_office(self, office_id: str) -> str:
        """Fetch office details and format them as a string."""
        if office_id not in self.offices_map:
            return f"(Office details not found for ID: {office_id})"
        
        off = self.offices_map[office_id]
        loc = off.get('location', {})
        addr = loc.get('address_line', 'N/A')
        landmark = loc.get('landmark', '')
        phone = ", ".join(off.get('contacts', []))
        hours = off.get('timings', {}).get('working_hours', 'N/A')
        
        return (f"\n    -> LOCATION: {off.get('office_name_en')}\n"
                f"       ADDRESS: {addr}, {landmark}\n"
                f"       TIMINGS: {hours}\n"
                f"       PHONE: {phone}")

    def _format_service(self, service_id: str) -> str:
        """Fetch service details and link to the responsible office."""
        if service_id not in self.services_map:
            return "" # If linked service doesn't exist, ignore it
            
        srv = self.services_map[service_id]
        
        # Find responsible offices for this service
        office_texts = []
        for off_id in srv.get('responsible_office_ids', []):
            office_texts.append(self._format_office(off_id))
            
        office_block = "\n".join(office_texts) if office_texts else "    -> Office: Online or Not Specified"
        
        return (f"\n  * REQUIREMENT: {srv.get('service_name_en')}\n"
                f"    DESCRIPTION: {srv.get('description')}\n"
                f"    SLA/TIME: {srv.get('sla_processing_time', {}).get('standard_days', 'N/A')} days\n"
                f"{office_block}")

    def create_documents(self) -> List[Document]:
        """
        The Master Logic: Creates one 'Super Document' per Scheme.
        """
        documents = []
        
        # Fixed: Iterate over 'self.schemes', not 'self.schema'
        for sc in self.schemes:
            # A. Basic Info
            content = [
                f"SCHEME: {sc.get('scheme_name')} ({sc.get('scheme_id')})",
                f"CATEGORY: {sc.get('category')}",
                f"BENEFITS: {sc.get('benefits', {}).get('description') or sc.get('benefits', {}).get('max_amount_inr')}",
                f"ELIGIBILITY: {json.dumps(sc.get('eligibility_criteria', {}), indent=2)}"
            ]
            
            # B. Enrich Documents Required (Link to Services -> Link to Offices)
            docs_req_text = ["\n--- DOCUMENTS & WHERE TO GET THEM ---"]
            has_docs = False
            
            for doc in sc.get('documents_required', []):
                doc_name = doc.get('doc_name', 'Unknown Document')
                linked_id = doc.get('linked_service_id')
                
                if linked_id:
                    # RECURSIVE ENRICHMENT: Go fetch the service and office data
                    enriched_info = self._format_service(linked_id)
                    docs_req_text.append(enriched_info)
                else:
                    docs_req_text.append(f"\n  * {doc_name} (Standard Document)")
                has_docs = True
                
            if has_docs:
                content.extend(docs_req_text)

            # C. Enrich Application Process (Link directly to Offices)
            app_process_text = ["\n--- APPLICATION PROCESS ---"]
            stages = sc.get('application_process', {}).get('stages', [])
            
            for stage in stages:
                step_desc = f"Step {stage.get('step')}: {stage.get('action')}"
                
                # Check if this specific step happens at a specific office
                designated_off_id = stage.get('designated_office_id')
                if designated_off_id:
                    office_info = self._format_office(designated_off_id)
                    step_desc += office_info
                
                app_process_text.append(step_desc)
                
            content.extend(app_process_text)

            # Create the final huge text chunk
            full_text = "\n".join(content)
            
            documents.append(Document(
                page_content=full_text,
                metadata={
                    "source": "Linked_Database",
                    "scheme_id": sc.get('scheme_id'),
                    "type": "enriched_scheme"
                }
            ))
            
        return documents

# Usage Helper
def load_linked_documents(data_dir: str) -> List[Document]:
    loader = LinkedLoader(data_dir)
    loader.load_data()
    return loader.create_documents()