#!/usr/bin/env python
"""
Main entry point for the simple modular agentic RAG system.
"""
import sys
from mini_agentic_rag.llm import get_azure_llm
from mini_agentic_rag.agents import run_agentic_rag

def main():
    """Run the agentic RAG system with a user question or in interactive mode."""
    # Initialize LLM
    print("Initializing Azure OpenAI...", flush=True)
    try:
        llm = get_azure_llm()
    except Exception as e:
        print(f"\n Error initializing LLM: {e}")
        sys.exit(1)

    # Interactive mode if no arguments provided
    if len(sys.argv) < 2:
        print("="*80, flush=True)
        print("AGENTIC RAG CHATBOT", flush=True)
        print("="*80, flush=True)
        print("Type 'exit', 'quit', or Press Ctrl+C to stop.", flush=True)
        
        while True:
            try:
                question = input("\n[User]: ").strip()
                if not question:
                    continue
                if question.lower() in ['exit', 'quit']:
                    print("\nGoodbye!", flush=True)
                    break
                
                print("\n" + "-"*40, flush=True)
                # Run agentic RAG pipeline
                result = run_agentic_rag(question, llm)
                
                print("\n[Assistant]:", flush=True)
                print(result["final_answer"], flush=True)
                print("-"*40, flush=True)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n Error: {e}")
        return

    # Question from command line
    question = " ".join(sys.argv[1:])
    
    print("="*80)
    print("AGENTIC RAG SYSTEM")
    print("="*80)
    print(f"\nQuestion: {question}\n")
    
    try:
        # Run agentic RAG pipeline
        result = run_agentic_rag(question, llm)
        
        # Display results
        print("\n" + "="*80)
        print("FINAL ANSWER")
        print("="*80)
        print(result["final_answer"])
        
        print("\n" + "="*80)
        print(f"SOURCES ({len(result['sources'])} chunks retrieved)")
        print("="*80)
        for i, source in enumerate(result["sources"], 1):
            print(f"\n[{i}] {source['content']}")
            if source.get('metadata'):
                print(f"    Metadata: {source['metadata']}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
