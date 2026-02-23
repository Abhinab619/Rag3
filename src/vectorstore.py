import os
import logging
from typing import List, Tuple, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define path for FAISS index
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "faiss_index")

# imports for optional dependencies
_embedding_pipeline = None
_bm25_retriever = None
_reranker = None


def _get_embedding_pipeline():
    """Lazy load embedding pipeline."""
    global _embedding_pipeline
    if _embedding_pipeline is None:
        from src.embedding import EmbeddingPipeline
        _embedding_pipeline = EmbeddingPipeline()
    return _embedding_pipeline


def _get_reranker():
    """Lazy load cross-encoder reranker."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder reranker...")
            _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Reranker loaded successfully.")
        except ImportError:
            logger.warning("sentence-transformers not installed. Reranking disabled.")
            _reranker = False  # Mark as unavailable
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}")
            _reranker = False
    return _reranker if _reranker else None


def get_db_connection() -> Optional[FAISS]:
    """
    Attempts to load the existing FAISS index from disk.
    Returns None if the folder doesn't exist yet.
    """
    pipeline = _get_embedding_pipeline()
    embedding_model = pipeline.get_model()

    if os.path.exists(DB_PATH):
        try:
            db = FAISS.load_local(
                DB_PATH, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
            # logger.info(f"Loaded FAISS index from {DB_PATH}")
            return db
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return None
    return None


def add_to_faiss(chunks: List[Document]) -> bool:
    """
    Adds chunks to FAISS with deduplication check.
    If DB exists, it appends. If not, it creates a new one.
    """
    if not chunks:
        logger.warning("No chunks to add.")
        return False

    pipeline = _get_embedding_pipeline()
    embedding_model = pipeline.get_model()
    
    # Load existing DB
    db = get_db_connection()

    logger.info(f"Adding {len(chunks)} chunks to FAISS...")

    try:
        if db:
            # Append to existing DB
            db.add_documents(chunks)
        else:
            # Create new DB from scratch
            db = FAISS.from_documents(chunks, embedding_model)

        # Save everything to disk
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        db.save_local(DB_PATH)
        logger.info(f"Index saved to {DB_PATH}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to add documents to FAISS: {e}")
        return False


def _rerank_results(query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
    """
    Re-rank documents using cross-encoder for better relevance.
    """
    reranker = _get_reranker()
    
    if not reranker or len(documents) <= top_k:
        return documents[:top_k]
    
    try:
        # Create query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get reranker scores
        scores = reranker.predict(pairs)
        
        # Sort by score (higher is better for cross-encoder)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k documents
        reranked = [doc for doc, score in scored_docs[:top_k]]
        logger.debug(f"Reranked {len(documents)} documents, returning top {top_k}")
        return reranked
        
    except Exception as e:
        logger.warning(f"Reranking failed: {e}. Returning original order.")
        return documents[:top_k]


def _bm25_search(query: str, documents: List[Document], top_k: int = 10) -> List[Document]:
    """
    Perform BM25 keyword search on documents.
    """
    try:
        from rank_bm25 import BM25Okapi
        
        # Tokenize documents
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get scores
        scores = bm25.get_scores(tokenized_query)
        
        # Sort by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:top_k]]
        
    except ImportError:
        logger.warning("rank_bm25 not installed. BM25 search disabled.")
        return []
    except Exception as e:
        logger.warning(f"BM25 search failed: {e}")
        return []


def format_docs_to_string(documents: List[Document], max_chars: int = 400) -> str:
    if not documents:
        return "No relevant documents found."
    context = ""
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        content = doc.page_content[:max_chars]  # Truncate!
        context += f"\n[{i}] {os.path.basename(source)}\n{content}\n"
    return context


def search_faiss(
    query: str, 
    k: int = 5,
    use_mmr: bool = True,
    use_reranking: bool = True,
    score_threshold: float = None,
    fetch_k: int = 10,
    filter: Optional[dict] = None
) -> List[Document]:
    """
    Enhanced search returning Document objects (ROBUST MODE).
    """
    db = get_db_connection()
    
    if not db:
        logger.error("FAISS Index not found.")
        return []

    try:
        if use_mmr:
            results = db.max_marginal_relevance_search(
                query, 
                k=k if not use_reranking else fetch_k,
                fetch_k=fetch_k * 2,
                lambda_mult=0.7,
                filter=filter
            )
        else:
            if score_threshold is not None:
                results_with_scores = db.similarity_search_with_score(query, k=fetch_k, filter=filter)
                results = [
                    doc for doc, score in results_with_scores 
                    if score <= score_threshold
                ][:k if not use_reranking else fetch_k]
            else:
                results = db.similarity_search(query, k=fetch_k if use_reranking else k, filter=filter)
        
        if use_reranking and len(results) > k:
            results = _rerank_results(query, results, top_k=k)
            
        return results
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []


def search_faiss_with_scores(
    query: str,
    k: int = 5,
    fetch_k: int = 20,
    filter: Optional[dict] = None
) -> List[Tuple[Document, float]]:
    """
    Search FAISS and return documents WITH their L2 distance scores.
    Lower score = more similar. Used for retrieval confidence checking.
    """
    db = get_db_connection()
    
    if not db:
        logger.error("FAISS Index not found.")
        return []

    try:
        results = db.similarity_search_with_score(query, k=fetch_k, filter=filter)
        return results[:k]
    except Exception as e:
        logger.error(f"Search with scores error: {str(e)}")
        return []


def hybrid_search(
    query: str,
    k: int = 5,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3
) -> List[Document]:
    """
    Hybrid search returning Document objects (ROBUST MODE).
    """
    db = get_db_connection()
    
    if not db:
        logger.error("FAISS Index not found.")
        return []
    
    try:
        # Get vector search results with scores
        vector_results = db.similarity_search_with_score(query, k=20)
        
        if not vector_results:
            return []
        
        # Normalize vector scores
        max_dist = max(score for _, score in vector_results) or 1
        vector_scores = {
            doc.page_content: (1 - score / max_dist) * vector_weight 
            for doc, score in vector_results
        }
        
        # Get BM25 results
        all_docs = [doc for doc, _ in vector_results]
        bm25_results = _bm25_search(query, all_docs, top_k=20)
        
        # Score BM25 results
        bm25_scores = {}
        for i, doc in enumerate(bm25_results):
            bm25_scores[doc.page_content] = (1 - i / len(bm25_results)) * bm25_weight
        
        # Combine scores
        combined_scores = {}
        doc_map = {doc.page_content: doc for doc, _ in vector_results}
        
        for content in set(list(vector_scores.keys()) + list(bm25_scores.keys())):
            combined_scores[content] = (
                vector_scores.get(content, 0) + 
                bm25_scores.get(content, 0)
            )
        
        # Sort by combined score
        sorted_contents = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        # Get top k documents
        final_docs = [doc_map[content] for content in sorted_contents[:k] if content in doc_map]
        
        # Apply reranking
        final_docs = _rerank_results(query, final_docs, top_k=k)
        
        return final_docs
        
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        return []

# MISSING UTILITY FUNCTIONS ADDED BELOW 

def get_index_stats() -> dict:
    """
    Get statistics about the current FAISS index.
    """
    db = get_db_connection()
    
    if not db:
        return {"status": "No index found"}
    
    try:
        # Get document count
        doc_count = db.index.ntotal
        return {
            "status": "active",
            "document_count": doc_count,
            "index_path": DB_PATH
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def clear_index() -> bool:
    """
    Delete the FAISS index (use with caution).
    """
    import shutil
    
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            logger.info(f"Cleared FAISS index at {DB_PATH}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return False
    return True