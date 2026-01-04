import os
import argparse
import sys
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

def load_documents(source_path: str) -> List[Document]:
    """
    Loads documents from a file or directory.
    Supports .txt, .pdf, and .csv files.
    """
    documents = []
    
    if os.path.isfile(source_path):
        files = [source_path]
    elif os.path.isdir(source_path):
        files = []
        for root, _, filenames in os.walk(source_path):
            for filename in filenames:
                files.append(os.path.join(root, filename))
    else:
        raise ValueError(f"Invalid path: {source_path}")

    print(f"Found {len(files)} files to process.")

    for file_path in files:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in [".txt", ".pdf", ".csv"]:
            continue
            
        print(f"Loading {file_path}...")
        try:
            if ext == ".txt":
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            elif ext == ".pdf":
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif ext == ".csv":
                loader = CSVLoader(file_path)
                documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return documents

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def get_embeddings_model() -> AzureOpenAIEmbeddings:
    """
    Initializes the Azure OpenAI Embeddings model.
    """
    api_key = os.getenv("AZURE_API_KEY")
    endpoint = os.getenv("AZURE_API_BASE")
    api_version = os.getenv("AZURE_EMBEDDING_API_VERSION")
    deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

    if not all([api_key, endpoint, api_version, deployment]):
        raise ValueError("Missing Azure OpenAI environment variables for embeddings. Please check your .env file.")

    return AzureOpenAIEmbeddings(
        azure_deployment=deployment,
        openai_api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )

def create_embeddings(chunks: List[Document], embeddings_model: AzureOpenAIEmbeddings) -> List[List[float]]:
    """
    Creates embeddings for the given chunks.
    """
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embeddings_model.embed_documents(texts)
    return embeddings

def index_documents(documents: List[Document], embeddings_model: AzureOpenAIEmbeddings, collection_name: str = "agentic_rag_knowledge"):
    """
    Indexes documents into Qdrant.
    """
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not all([url, api_key]):
        raise ValueError("Missing Qdrant environment variables. Please check your .env file.")

    print(f"Indexing {len(documents)} chunks into Qdrant collection '{collection_name}'...")
    
    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings_model,
        url=url,
        api_key=api_key,
        collection_name=collection_name,
    )
    
    print("Successfully indexed documents to Qdrant.")
    return vector_store

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant for Agentic RAG.")
    parser.add_argument("--path", type=str, default="knowledge", help="Path to a file or directory for ingestion.")
    parser.add_argument("--collection", type=str, default="agentic_rag_knowledge", help="Qdrant collection name.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for text splitting.")
    parser.add_argument("--overlap", type=int, default=200, help="Chunk overlap for text splitting.")
    
    args = parser.parse_args()

    print(f"--- Knowledge Ingestion Started ---")
    print(f"Source Path: {args.path}")
    print(f"Collection Name: {args.collection}")
    
    try:
        # 1. Load Documents
        print(f"\n1. Loading documents...")
        documents = load_documents(args.path)
        if not documents:
            print("No valid documents found. Exiting.")
            return
        print(f"Total documents loaded: {len(documents)}")
        
        # 2. Chunk Documents
        print(f"\n2. Chunking documents (size={args.chunk_size}, overlap={args.overlap})...")
        chunks = chunk_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.overlap)
        print(f"Total chunks created: {len(chunks)}")
        
        # 3. Get Embeddings Model
        print(f"\n3. Initializing embeddings model...")
        embeddings_model = get_embeddings_model()
        
        # 4. Index to Qdrant
        print(f"\n4. Indexing to Qdrant...")
        index_documents(chunks, embeddings_model, collection_name=args.collection)
        
        print(f"\n--- Ingestion Complete Successfully ---")
        
    except Exception as e:
        print(f"\nError during ingestion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
