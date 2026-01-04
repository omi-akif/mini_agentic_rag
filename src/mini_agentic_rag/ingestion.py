import os
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

def load_documents(source_path: str) -> List[Document]:
    """
    Loads documents from a file or directory.
    """
    if os.path.isfile(source_path):
        loader = TextLoader(source_path)
    elif os.path.isdir(source_path):
        loader = DirectoryLoader(source_path, glob="**/*.txt", loader_cls=TextLoader)
    else:
        raise ValueError(f"Invalid path: {source_path}")
    
    return loader.load()

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

if __name__ == "__main__":
    # Example usage
    source_path = "knowledge/user_preference.txt"
    
    print(f"Loading documents from {source_path}...")
    try:
        documents = load_documents(source_path)
        print(f"Loaded {len(documents)} document(s).")
        
        chunks = chunk_documents(documents)
        print(f"Split into {len(chunks)} chunks.")
        
        print("Initializing Azure OpenAI Embeddings...")
        embeddings_model = get_embeddings_model()
        
        print("Creating embeddings...")
        embeddings = create_embeddings(chunks, embeddings_model)
        
        if embeddings:
            print(f"Successfully created {len(embeddings)} embeddings.")
            print(f"Embedding dimension: {len(embeddings[0])}")
        else:
            print("No embeddings created.")
            
    except Exception as e:
        print(f"Error: {e}")
