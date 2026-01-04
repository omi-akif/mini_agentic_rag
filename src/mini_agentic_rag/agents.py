import os
import yaml
from typing import List, Dict, Any
from mini_agentic_rag.retrieval import retrieve_context

# Load configurations
def load_config():
    base_path = os.path.dirname(__file__)
    config_path = os.path.join(base_path, 'config')
    
    with open(os.path.join(config_path, 'agents.yaml'), 'r') as f:
        agents_config = yaml.safe_load(f)
        
    with open(os.path.join(config_path, 'tasks.yaml'), 'r') as f:
        tasks_config = yaml.safe_load(f)
        
    return agents_config, tasks_config

AGENTS_CONFIG, TASKS_CONFIG = load_config()

class Agent:
    """Base agent class with LLM integration."""
    
    def __init__(self, name: str, role: str, backstory: str, llm_callable):
        self.name = name
        self.role = role
        self.backstory = backstory
        self.llm = llm_callable
    
    def run(self, task: str, context: str = "") -> str:
        """Execute the agent's task."""
        raise NotImplementedError("Subclasses must implement run()")
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM with conversation context."""
        return self.llm(messages)


class ResearcherAgent(Agent):
    """Agent that retrieves relevant context from the knowledge base."""
    
    def __init__(self, llm_callable):
        config = AGENTS_CONFIG['knowledge_researcher']
        super().__init__(
            name=config['role'], # In common CrewAI usage, role is often used as name/title
            role=config['role'],
            backstory=config['backstory'],
            llm_callable=llm_callable
        )
    
    def run(self, question: str) -> Dict[str, Any]:
        """
        Retrieve context and formulate initial answer.
        """
        print(f"\n[{self.name}] Searching knowledge base for: '{question}'", flush=True)
        
        # Step 1: Retrieve relevant context
        documents = retrieve_context(question, k=5)
        
        if not documents:
            return {
                "context": "",
                "sources": [],
                "initial_answer": "No relevant information found in the knowledge base."
            }
        
        # Format retrieved context
        context_text = "\n\n".join([
            f"[Source {i+1}] {doc.page_content}"
            for i, doc in enumerate(documents)
        ])
        
        sources = [
            {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]
        
        print(f"[{self.name}] Found {len(documents)} relevant chunks", flush=True)
        
        # Step 2: Formulate initial answer based on context
        task_config = TASKS_CONFIG['knowledge_extraction_task']
        task_prompt = task_config['description'].format(question=question)
        
        messages = [
            {
                "role": "system",
                "content": f"You are a {self.role}. {self.backstory}\nYour job is to answer questions based ONLY on the provided context. If the context doesn't contain enough information, say so."
            },
            {
                "role": "user",
                "content": f"Task: {task_prompt}\n\nContext from knowledge base:\n\n{context_text}\n\nQuestion: {question}\n\nProvide the expected output: {task_config['expected_output']}"
            }
        ]
        
        print(f"[{self.name}] Formulating initial answer...", flush=True)
        initial_answer = self._call_llm(messages)
        
        return {
            "context": context_text,
            "sources": sources,
            "initial_answer": initial_answer
        }


class CriticAgent(Agent):
    """Agent that reviews and refines answers to ensure accuracy."""
    
    def __init__(self, llm_callable):
        config = AGENTS_CONFIG['agentic_critic']
        super().__init__(
            name=config['role'],
            role=config['role'],
            backstory=config['backstory'],
            llm_callable=llm_callable
        )
    
    def run(self, question: str, research_result: Dict[str, Any]) -> str:
        """
        Review the researcher's answer and refine it.
        """
        print(f"\n[{self.name}] Reviewing research findings...", flush=True)
        
        context = research_result.get("context", "")
        initial_answer = research_result.get("initial_answer", "")
        
        if not context:
            return initial_answer
        
        task_config = TASKS_CONFIG['critique_task']
        task_prompt = task_config['description'].format(question=question, context=context)
        
        # Review and refine the answer
        messages = [
            {
                "role": "system",
                "content": f"You are a {self.role}. {self.backstory}\nReview the initial answer and ensure it is:\n"
                          "1. Accurate and grounded in the provided context\n"
                          "2. Complete and addresses all aspects of the question\n"
                          "3. Clear and well-structured"
            },
            {
                "role": "user",
                "content": f"Task: {task_prompt}\n\n"
                          f"Initial Answer:\n{initial_answer}\n\n"
                          f"Provide the expected output: {task_config['expected_output']}"
            }
        ]
        
        print(f"[{self.name}] Refining answer...", flush=True)
        final_answer = self._call_llm(messages)
        
        return final_answer


def run_agentic_rag(question: str, llm_callable) -> Dict[str, Any]:
    """
    Run the complete agentic RAG pipeline.
    
    Args:
        question: User's question
        llm_callable: LLM function from llm.py
    
    Returns:
        Dict with 'question', 'final_answer', 'sources', and 'research_result'
    """
    # Step 1: Research
    researcher = ResearcherAgent(llm_callable)
    research_result = researcher.run(question)
    
    # Step 2: Critique and refine
    critic = CriticAgent(llm_callable)
    final_answer = critic.run(question, research_result)
    
    return {
        "question": question,
        "final_answer": final_answer,
        "sources": research_result.get("sources", []),
        "research_result": research_result
    }
