# Mini Agentic RAG System

Clean, simple agentic RAG system using Azure OpenAI, Qdrant vector database, and modular Python architecture.

Use Python 3.10+ for this project. 

## Features
- **Build a RAG system**: Created a simple agentic RAG system using Azure OpenAI, Qdrant vector database, and modular Python architecture.
- **Agentic Pipeline**: ResearcherAgent → CriticAgent for quality answers
- **Vector Search**: Qdrant cloud for semantic document retrieval
- **Azure OpenAI**: LiteLLM integration done instead of CrewAI and LangGraph
- **Document Chunking**: Smart text splitting for better retrieval. 200 tokens overlapping. Chunking size 1000
- **Simple & Clean**: No framework was used. Done purely on Python

## Project Structure

```
mini_agentic_rag/
├── src/mini_agentic_rag/
│   ├── __init__.py
│   ├── agents.py       # Base classes for agents and tool calling
│   ├── ingestion.py    # Document loading, chunking, and vector ingestion
│   ├── llm.py          # Azure OpenAI integration via LiteLLM
│   ├── main.py         # App entry point (orchestrates agents)
│   ├── retrieval.py    # Qdrant client and search logic
│   └── config/         # YAML configurations
│       ├── agents.yaml # Agent definitions
│       └── tasks.yaml  # Task definitions
├── knowledge/          # Source documents (PDF, TXT, etc.)
├── tests/              # Test suite
│   └── test_llm.py     # Connectivity test script
├── .env                # Project secrets and configuration
├── chat.sh             # Run the chat interface
├── ingest.sh           # Run the knowledge ingestion script
├── pyproject.toml      # Project dependencies
└── README.md           # Documentation
```

## Setup

### 1. Environment Setup

It is recommended to use a Conda environment to manage dependencies:

```bash
# Create a new conda environment
conda create -n mini_agentic_rag python=3.10 -y

# Activate the environment
conda activate mini_agentic_rag

# Install the project and its dependencies
pip install -e .
```

### 2. Configuration (.env)

Create a `.env` file in the root directory and provide your Azure OpenAI and Qdrant credentials. Use the following template:

```env
# --- Azure OpenAI Configuration ---
AZURE_API_KEY=your_azure_api_key
AZURE_API_BASE=your_azure_endpoint_url

# Chat Model Settings
MODEL=azure/gpt-4o-mini  # Format: azure/<deployment_name>
AZURE_CHAT_API_VERSION=2024-02-15-preview

# Embedding Model Settings
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_EMBEDDING_API_VERSION=2023-05-15

# --- Qdrant Configuration ---
QDRANT_URL=your_qdrant_cloud_url_or_localhost
QDRANT_API_KEY=your_qdrant_api_key
```

### 3. Verify Connectivity

Before ingesting documents, verify your Azure OpenAI connectivity:

```bash
python tests/test_llm.py
```

### 4. Knowledge Ingestion

Load your documents (PDF, TXT, etc.) into the `knowledge/` folder, then run the ingestion script:

```bash
# Usage: ./ingest.sh --path <folder_path> --collection <collection_name>
./ingest.sh --path knowledge --collection agentic_rag_knowledge
```
This will chunk the documents, generate embeddings, and store them in your Qdrant collection named `agentic_rag_knowledge`.

## Usage

```bash
# To ask a question
bash chat.sh
```

## How It Works

1. **ResearcherAgent** retrieves relevant chunks from Qdrant
2. **CriticAgent** reviews and refines the answer
3. Final grounded answer based on knowledge

## Components

- **agents.py**: Simple agent classes with tool calling
- **llm.py**: Azure OpenAI via LiteLLM (verified working)
- **retrieval.py**: Qdrant vector search
- **ingestion.py**: PDF and csv processing & embedding generation
- **main.py**: CLI orchestration
