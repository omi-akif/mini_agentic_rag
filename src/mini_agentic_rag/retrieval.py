import os
from typing import List
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from mini_agentic_rag.ingestion import get_embeddings_model

# Load environment variables
load_dotenv()

def get_vector_store(collection_name: str = "agentic_rag_knowledge") -> QdrantVectorStore:
    """
    Connects to the Qdrant vector store.
    """
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not all([url, api_key]):
        raise ValueError("Missing Qdrant environment variables. Please check your .env file.")

    embeddings_model = get_embeddings_model()
    
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings_model,
        collection_name=collection_name,
        url=url,
        api_key=api_key,
    )

def retrieve_context(query: str, k: int = 3) -> List[Document]:
    """
    Retrieves the most relevant document chunks for a given query.
    """
    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k=k)

if __name__ == "__main__":
    # Quick test
    test_query = "What is this document about?"
    print(f"Testing retrieval for query: '{test_query}'")
    
    try:
        results = retrieve_context(test_query)
        print(f"Retrieved {len(results)} chunks:")
        for i, doc in enumerate(results):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
    except Exception as e:
        print(f"Error during retrieval: {e}")
