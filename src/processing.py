import re
import hashlib
import logging
from typing import List, Set
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        """
        Initializes the processor with optimized chunking settings.
        
        Args:
            chunk_size (int): Max characters per chunk. Increased to 800 to keep tables intact.
            chunk_overlap (int): Characters to overlap.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._seen_hashes: Set[str] = set()  # For deduplication

    def get_content_hash(self, text: str) -> str:
        """Generate a hash for text content to detect duplicates."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _is_table_row(self, line: str) -> bool:
        """
        Heuristic to check if a line might be part of a table.
        Checks for multiple tabs, pipes, or wide gaps.
        """
        if "|" in line and line.count("|") > 2:
            return True
        if line.count("\t") > 2:
            return True
        # Check for multiple wide spaces (common in PDF tables)
        if len(re.findall(r' {3,}', line)) > 1:
            return True
        return False

    def clean_text(self, text: str) -> str:
        """
        Advanced cleaning logic that PRESERVES table structures.
        """
        if not text:
            return ""
        
        # 1. Fix broken words at line breaks (e.g., "de-\nvelopment")
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # 2. Remove page headers/footers
        text = re.sub(r'Page\s*\d+\s*(of\s*\d+)?', '', text, flags=re.IGNORECASE)
        
        # 3. Remove watermarks
        text = re.sub(r'CONFIDENTIAL|DRAFT|INTERNAL USE ONLY', '', text, flags=re.IGNORECASE)
        
        # 4. Standardize list markers
        text = re.sub(r'[•●○▪▫◦]', '- ', text)
        
        # 5. Fix Unicode artifacts
        text = text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2013', '-').replace('\u2014', '-')
        
        # 6. PROCESSING LINES (The Table Fix)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
                
            if self._is_table_row(line):
                # PRESERVE: Keep internal spacing for tables, just strip ends
                cleaned_lines.append(line.rstrip()) 
            else:
                # COLLAPSE: Normal text gets spaces normalized
                clean_line = re.sub(r'[ \t]+', ' ', stripped)
                cleaned_lines.append(clean_line)
        
        # Reassemble
        text = '\n'.join(cleaned_lines)
        
        # 7. Normalize multiple newlines (max 2) to preserve paragraph structure
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    def extract_metadata_from_content(self, text: str, existing_metadata: dict) -> dict:
        """
        Extract additional metadata from document content safely.
        """
        metadata = existing_metadata.copy() if existing_metadata else {}
        
        try:
            # Try to extract document title
            lines = text.strip().split('\n')
            for line in lines[:5]:  
                line = line.strip()
                if line and len(line) < 200: 
                    if not line.endswith(('.', ',', ';')) and line[0].isupper():
                        metadata['extracted_title'] = line
                        break
            
            # Extract dates
            date_pattern = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'
            dates = re.findall(date_pattern, text)
            if dates:
                metadata['extracted_dates'] = dates[:3]
            
            # Detect document type
            text_lower = text.lower()
            if any(w in text_lower for w in ['scheme', 'yojana', 'programme', 'benefit']):
                metadata['doc_type'] = 'scheme'
            elif any(w in text_lower for w in ['notification', 'order', 'circular', 'gazette']):
                metadata['doc_type'] = 'notification'
            elif any(w in text_lower for w in ['form', 'application', 'annexure']):
                metadata['doc_type'] = 'form'
            else:
                metadata['doc_type'] = 'general'
                
        except Exception as e:
            logger.warning(f"Metadata extraction warning: {e}")
        
        return metadata

    def chunk_documents(self, documents: List[Document], deduplicate: bool = True) -> List[Document]:
        """
        Takes raw documents, cleans them, and splits them into searchable chunks.
        """
        logger.info(f"Processing {len(documents)} documents...")
        
        cleaned_docs = []
        for doc in documents:
            try:
                cleaned_content = self.clean_text(doc.page_content)
                
                # Skip empty documents
                if not cleaned_content or len(cleaned_content) < 50:
                    continue
                
                # Enrich metadata
                enriched_metadata = self.extract_metadata_from_content(cleaned_content, doc.metadata)
                
                new_doc = Document(
                    page_content=cleaned_content,
                    metadata=enriched_metadata
                )
                cleaned_docs.append(new_doc)
            except Exception as e:
                logger.error(f"Error cleaning document {doc.metadata.get('source', 'unknown')}: {e}")
                continue
        
        logger.info(f"Cleaned {len(cleaned_docs)} documents.")
        
        # Split into chunks
        # Adjusted separators to keep paragraphs and tables together
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "], 
            add_start_index=True,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(cleaned_docs)
        
        # Deduplication
        if deduplicate:
            unique_chunks = []
            for chunk in chunks:
                content_hash = self.get_content_hash(chunk.page_content)
                if content_hash not in self._seen_hashes:
                    self._seen_hashes.add(content_hash)
                    chunk.metadata['content_hash'] = content_hash
                    unique_chunks.append(chunk)
            chunks = unique_chunks
        
        # Add chunk-level metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['char_count'] = len(chunk.page_content)
        
        logger.info(f"Final output: {len(chunks)} unique chunks.")
        return chunks

    def reset_deduplication(self):
        """Reset the deduplication hash set."""
        self._seen_hashes.clear()
        logger.info("Deduplication cache cleared.")


if __name__ == "__main__":
    # Test with a table-like structure
    dummy_text = """
    Page 1
    
    INCOME CRITERIA:
    Category      | Limit (Rs) | Status
    General       | 2,00,000   | Active
    SC/ST         | 5,00,000   | Active
    
    Please apply online.
    """
    
    doc = Document(page_content=dummy_text, metadata={"source": "test.pdf"})
    processor = TextProcessor()
    chunks = processor.chunk_documents([doc])
    
    print("--- Processed Text ---")
    print(chunks[0].page_content)