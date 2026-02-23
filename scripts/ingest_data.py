import sys
import os
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.append(project_root)


DATA_PATH = os.path.join(project_root, "data")
FAISS_INDEX_PATH = os.path.join(project_root, "data", "faiss_index")

# Import from your custom modules
from src.linked_loader import load_linked_documents
from src.processing import TextProcessor
from src.vectorstore import add_to_faiss, get_index_stats, clear_index

# Import JSONLoader from data_loader (since we merged them)
try:
    from src.data_loader import JSONLoader
except ImportError:
    # Fallback if you still have the separate file
    from src.json_loader import JSONLoader

# 3. Import LangChain loaders
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, 
    UnstructuredExcelLoader, Docx2txtLoader
)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_loader_for_file(file_path: Path):
    """Return the appropriate LangChain loader based on file extension."""
    ext = file_path.suffix.lower()
    str_path = str(file_path)
    
    if ext == ".pdf":
        return PyPDFLoader(str_path)
    elif ext == ".txt":
        return TextLoader(str_path, encoding='utf-8')
    elif ext == ".csv":
        return CSVLoader(str_path)
    elif ext in [".xlsx", ".xls"]:
        return UnstructuredExcelLoader(str_path)
    elif ext == ".docx":
        return Docx2txtLoader(str_path)
    elif ext == ".json":
        return JSONLoader(str_path)
    else:
        return None

def process_batch(file_batch: List[Path], processor: TextProcessor) -> int:
    """
    Loads, cleans, chunks, and saves a batch of files.
    """
    raw_docs = []
    
    for file_path in file_batch:
        try:
            loader = get_loader_for_file(file_path)
            if loader:
                docs = loader.load()
                # Add source metadata if missing
                for doc in docs:
                    if "source" not in doc.metadata:
                        doc.metadata["source"] = str(file_path)
                raw_docs.extend(docs)
            else:
                logger.warning(f"Unsupported file type: {file_path}")
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
            continue

    if not raw_docs:
        return 0

    # Chunk Documents
    chunks = processor.chunk_documents(raw_docs, deduplicate=True)
    
    if not chunks:
        return 0

    success = add_to_faiss(chunks)
    return len(chunks) if success else 0


def main(clear_existing: bool = False, batch_size: int = 10) -> bool:
    start_time = datetime.now()
    
    print("\n" + "=" * 60)
    print("ROBUST DATA INGESTION PIPELINE")
    print("=" * 60)

    # 1. Manage Existing Index
    if clear_existing:
        print("Creating fresh index (Clearing old data)...")
        clear_index()
    else:
        stats = get_index_stats()
        print(f"Appending to existing index. Current size: {stats.get('document_count', 0)} vectors.")

    # 2. Initialize Processor
    processor = TextProcessor(chunk_size=800, chunk_overlap=150)
    total_chunks_added = 0

    # --- PHASE 1: LINKED DATA (High Priority) ---
    print("\n--- Phase 1: Ingesting Linked Knowledge (Schemes/Services/Offices) ---")
    try:
        linked_docs = load_linked_documents(DATA_PATH)
        
        if linked_docs:
            print(f"Generated {len(linked_docs)} enriched scheme records.")
            
            linked_chunks = processor.chunk_documents(linked_docs, deduplicate=True)
            
            if add_to_faiss(linked_chunks):
                print(f"Successfully added {len(linked_chunks)} linked knowledge vectors.")
                total_chunks_added += len(linked_chunks)
            else:
                logger.error("Failed to add linked documents to FAISS.")
        else:
            print("No linked data found (checked schemes.json, services.json, offices.json).")

    except Exception as e:
        logger.error(f"Error during Linked Data phase: {e}")

    # --- PHASE 2: GENERAL FILES (PDFs, etc.) ---
    print("\n--- Phase 2: Ingesting General Documents ---")
    
    data_dir = Path(DATA_PATH)
    all_files = []
    extensions = ['*.pdf', '*.txt', '*.csv', '*.xlsx', '*.docx', '*.json']
    
    # Exclude files handled in Phase 1
    excluded_files = {"schemes.json", "services.json", "offices.json", "schema.json"}

    for ext in extensions:
        found = list(data_dir.rglob(ext))
        for f in found:
            if f.name not in excluded_files:
                all_files.append(f)
            else:
                logger.info(f"Skipping {f.name} (Handled in Phase 1)")

    if not all_files:
        print("No other documents found to process.")
    else:
        print(f"Found {len(all_files)} general documents.")
        
        total_files_processed = 0
        
        for i in range(0, len(all_files), batch_size):
            batch = all_files[i : i + batch_size]
            print(f"\nProcessing General Batch {i//batch_size + 1} ({len(batch)} files)...")
            
            chunks_added = process_batch(batch, processor)
            total_chunks_added += chunks_added
            total_files_processed += len(batch)
            
            print(f"Batch complete. Added {chunks_added} chunks.")

    # Final Stats
    duration = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print(f"Total Vectors in DB: {total_chunks_added}")
    print(f"Time Taken: {duration:.2f} seconds")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust Document Ingestion")
    parser.add_argument("--clear", action="store_true", help="Delete existing index and start fresh")
    parser.add_argument("--batch", type=int, default=10, help="Number of files to process at once (default: 10)")
    
    args = parser.parse_args()
    
    main(clear_existing=args.clear, batch_size=args.batch)