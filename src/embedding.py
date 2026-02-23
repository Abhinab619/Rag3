import os
import logging
import torch
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingPipeline:
    def __init__(
        self, 
        model_name: str = "models/gemini-embedding-001",
        chunk_size: int = 800,   # Matches processing.py for table preservation
        chunk_overlap: int = 150 # Matches processing.py
    ):
        """
        Initializes the Embedding Pipeline with robust settings.
        
        Args:
            model_name: The Google Generative AI model to use.
            chunk_size: Size of text chunks (800 preserves small tables).
            chunk_overlap: Overlap to keep context between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        
        logger.info(f"Loading embedding model: {model_name}...")
        
        try:
            self.model = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key="AIzaSyCczUh_aVhX9QEnOyaD5xjmtstQGWMPk8k"
            )
            logger.info(f"Model {model_name} loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load embedding model: {e}")
            logger.critical("Please check your Google API key configuration.")
            raise e

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of documents into smaller chunks with enhanced metadata.
        NOTE: This is a backup chunker. Ideally, use src.processing.py first.
        """
        if not documents:
            return []
            
        logger.info(f"Splitting {len(documents)} documents (Backup Method)...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            # Prioritize paragraph breaks, then sentences
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
            length_function=len,
        )
        
        chunks = splitter.split_documents(documents)
        
        # Enrich metadata for each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks_from_doc"] = len(chunks)
            chunk.metadata["embedding_model"] = self.model_name
            
            if "source" not in chunk.metadata:
                chunk.metadata["source"] = "Unknown"
        
        logger.info(f"Generated {len(chunks)} chunks.")
        return chunks

    def get_model(self):
        """
        Returns the embedding model object (Required for vector stores).
        """
        return self.model


# Example usage for testing
if __name__ == "__main__":
    try:
        pipeline = EmbeddingPipeline()
        
        # Test embedding
        test_text = "What are the eligibility criteria for pension scheme?"
        embedding = pipeline.get_model().embed_query(test_text)
        print(f"Success! Generated embedding vector of size: {len(embedding)}")
        
    except Exception as e:
        print(f"Test failed: {e}")