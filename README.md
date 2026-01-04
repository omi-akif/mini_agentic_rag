# Mini Agentic RAG System

Clean, simple agentic RAG system using Azure OpenAI, Qdrant vector database, and modular Python architecture.

## Features

- ✅ **Agentic Pipeline**: ResearcherAgent → CriticAgent for quality answers
- ✅ **Vector Search**: Qdrant for semantic document retrieval
- ✅ **Azure OpenAI**: Verified working LiteLLM integration
- ✅ **Document Chunking**: Smart text splitting for better retrieval
- ✅ **Simple & Clean**: No framework overhead, pure Python

## Project Structure

```
mini_agentic_rag/
├── src/mini_agentic_rag/
│   ├── agents.py       # ResearcherAgent & CriticAgent
│   ├── llm.py          # Azure OpenAI integration
│   ├── main.py         # CLI entry point
│   ├── ingestion.py    # Document loading & chunking
│   └── retrieval.py    # Qdrant vector search
├── knowledge/          # PDF documents
├── .env                # Azure & Qdrant credentials
└── test_llm.py         # LLM verification script
```

## Setup

1. **Install dependencies:**
```bash
conda activate coder_agent
pip install -e .
```

2. **Configure `.env`:**
```
AZURE_API_KEY=your_azure_key
AZURE_API_BASE=your_azure_endpoint
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
```

3. **Ingest documents:**
```bash
conda run -n coder_agent ingest
```

## Usage

```bash
# Ask questions
conda run -n coder_agent rag "Your question here"

# Examples
rag "What does Section 1 cover regarding liability?"
rag "What is covered under Section 2?"
rag "What types of insurance are available?"
```

## How It Works

1. **ResearcherAgent** retrieves relevant chunks from Qdrant
2. **CriticAgent** reviews and refines the answer
3. Final grounded answer with source citations

## Components

- **agents.py**: Simple agent classes with tool calling
- **llm.py**: Azure OpenAI via LiteLLM (verified working)
- **retrieval.py**: Qdrant vector search
- **ingestion.py**: PDF processing & embedding generation
- **main.py**: CLI orchestration
