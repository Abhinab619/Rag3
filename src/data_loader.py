import json
import logging
from pathlib import Path
from typing import List, Any, Dict

# LangChain Community Loaders
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader, 
    Docx2txtLoader
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CUSTOM JSON LOADER CLASS
class JSONLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def load(self) -> List[Document]:
        """Load JSON file and convert to Documents based on type."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading JSON file {self.file_path}: {e}")
            return []
            
        documents = []
        
        # Ensure data is a list
        if isinstance(data, dict):
            data = [data]
        
        for record in data:
            # Determine type of record and process accordingly
            if "scheme_id" in record:
                doc = self._process_scheme(record)
            elif "service_id" in record:
                doc = self._process_service(record)
            elif "office_id" in record:
                doc = self._process_office(record)
            else:
                continue
                
            documents.append(doc)
            
        return documents

    def _process_scheme(self, record: Dict) -> Document:
        """Convert scheme JSON to retrieval text format."""
        
        # Handle documents_required
        docs_required = record.get('documents_required', [])
        if docs_required and isinstance(docs_required[0], dict):
            docs_list = [d.get('doc_name', str(d)) for d in docs_required]
        else:
            docs_list = docs_required
        
        # Handle application_process stages
        app_process = record.get('application_process', {})
        stages = app_process.get('stages', [])
        steps_text = "\n".join([f"- Step {s.get('step', i+1)}: {s.get('action', '')}" for i, s in enumerate(stages)]) if stages else "None"
        
        # Build synonyms text
        synonyms = record.get('synonyms', [])
        synonyms_text = f"SYNONYMS/ALIASES: {', '.join(synonyms)}" if synonyms else ""
        
        content = f"""TYPE: scheme
SCHEME_ID: {record.get('scheme_id')}
SCHEME_NAME: {record.get('scheme_name')}
{synonyms_text}
CATEGORY: {record.get('category', 'General')}
IS_ACTIVE: {record.get('is_active', True)}

ELIGIBILITY:
{self._format_dict(record.get('eligibility_criteria', record.get('eligibility', {})))}

BENEFITS:
{self._format_dict(record.get('benefits', {}))}

DOCUMENTS REQUIRED:
{self._format_list(docs_list)}

HOW TO APPLY:
{steps_text}

TAGS: {", ".join(record.get('tags', []))}"""
        
        return Document(
            page_content=content,
            metadata={
                "source": self.file_path,
                "type": "scheme",
                "record_id": record.get('scheme_id'),
                "category": record.get('category'),
                "doc_type": "scheme" 
            }
        )

    def _process_service(self, record: Dict) -> Document:
        """Convert service JSON to retrieval text format."""
        
        docs_required = record.get('documents_required', [])
        if docs_required and isinstance(docs_required[0], dict):
            docs_list = [d.get('doc_name', str(d)) for d in docs_required]
        else:
            docs_list = docs_required
            
        office_ids = record.get('responsible_office_ids', record.get('office_ids', []))
        
        sla = record.get('sla_processing_time', {})
        sla_text = f"Processing Time: {sla.get('standard_days', 'N/A')} days" if sla else ""
        
        content = f"""TYPE: service
SERVICE_ID: {record.get('service_id')}
SERVICE_NAME: {record.get('service_name_en')}
SERVICE_TYPE: {record.get('service_type', 'General')}
DESCRIPTION: {record.get('description')}
{sla_text}

RESPONSIBLE_OFFICES: {", ".join(office_ids)}

DOCUMENTS_REQUIRED:
{self._format_list(docs_list)}"""

        return Document(
            page_content=content,
            metadata={
                "source": self.file_path,
                "type": "service",
                "record_id": record.get('service_id'),
                "service_type": record.get('service_type'),
                "doc_type": "service"
            }
        )

    def _process_office(self, record: Dict) -> Document:
        """Convert office JSON to retrieval text format."""
        
        location = record.get('location', {})
        address = f"{location.get('address_line', 'N/A')}, Landmark: {location.get('landmark', 'N/A')}" if location else record.get('address', 'N/A')
        
        timings = record.get('timings', record.get('office_timing', {}))
        working_hours = timings.get('working_hours', 'N/A') if isinstance(timings, dict) else str(timings)
        closed_on = ", ".join(timings.get('closed_on', [])) if isinstance(timings, dict) else ""
        
        contacts = record.get('contacts', [])
        contacts_text = ", ".join(str(c) for c in contacts) if contacts else "N/A"
        
        content = f"""TYPE: office
OFFICE_ID: {record.get('office_id')}
OFFICE_NAME: {record.get('office_name_en')}
OFFICE_TYPE: {record.get('office_type', 'Government Office')}
DISTRICT: {record.get('district')}
DESCRIPTION: {record.get('description', '')}

ADDRESS: {address}
CONTACTS: {contacts_text}

WORKING HOURS: {working_hours}
CLOSED ON: {closed_on}"""

        return Document(
            page_content=content,
            metadata={
                "source": self.file_path,
                "type": "office",
                "record_id": record.get('office_id'),
                "district": record.get('district'),
                "doc_type": "office"
            }
        )

    def _format_list(self, data: List) -> str:
        return "\n".join([f"- {item}" for item in data]) if data else "None"

    def _format_dict(self, data: Dict) -> str:
        lines = []
        for k, v in data.items():
            if isinstance(v, list):
                val = ", ".join(map(str, v))
            elif isinstance(v, dict):
                val = str(v)
            else:
                val = str(v)
            lines.append(f"- {k.replace('_', ' ').title()}: {val}")
        return "\n".join(lines) if lines else "None"


# MAIN LOADER FUNCTION

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    # 1. PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files")
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file.name}: {e}")

    # 2. TXT files
    txt_files = list(data_path.glob('**/*.txt'))
    print(f"[DEBUG] Found {len(txt_files)} TXT files")
    for txt_file in txt_files:
        try:
            loader = TextLoader(str(txt_file), encoding='utf-8')
            loaded = loader.load()
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load TXT {txt_file.name}: {e}")

    # 3. CSV files
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"[DEBUG] Found {len(csv_files)} CSV files")
    for csv_file in csv_files:
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load CSV {csv_file.name}: {e}")

    # 4. Excel files
    xlsx_files = list(data_path.glob('**/*.xlsx'))
    print(f"[DEBUG] Found {len(xlsx_files)} Excel files")
    for xlsx_file in xlsx_files:
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded = loader.load()
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Excel {xlsx_file.name}: {e}")

    # 5. Word files
    docx_files = list(data_path.glob('**/*.docx'))
    print(f"[DEBUG] Found {len(docx_files)} Word files")
    for docx_file in docx_files:
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Word {docx_file.name}: {e}")

    # 6. JSON files (USING CUSTOM CLASS ABOVE)
    json_files = list(data_path.glob('**/*.json'))
    print(f"[DEBUG] Found {len(json_files)} JSON files")
    for json_file in json_files:
        print(f"[DEBUG] Loading JSON: {json_file.name}")
        try:
            # Use the local custom JSONLoader class
            loader = JSONLoader(str(json_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} JSON docs from {json_file.name}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load JSON {json_file.name}: {e}")

    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents

# Example usage
if __name__ == "__main__":
    # Create a dummy data dir for testing if needed
    import os
    if os.path.exists("data"):
        docs = load_all_documents("data")
        print(f"Loaded {len(docs)} documents.")
    else:
        print("Data directory not found. Run from project root.")