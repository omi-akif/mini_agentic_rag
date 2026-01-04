"""
LLM initialization module with verified Azure OpenAI configuration.
"""
import os
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

def get_azure_llm():
    """
    Returns a callable that uses LiteLLM to call Azure OpenAI.
    This configuration is verified to work with the test_llm.py script.
    """
    api_key = os.getenv("AZURE_API_KEY")
    api_base = os.getenv("AZURE_API_BASE")
    api_version = "2025-01-01-preview"
    deployment = "gpt-4.1-mini"
    
    def call_llm(messages, temperature=0.7, max_tokens=1000):
        """Call Azure OpenAI via LiteLLM."""
        response = completion(
            model=f"azure/{deployment}",
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    return call_llm

if __name__ == "__main__":
    # Test the LLM
    llm = get_azure_llm()
    result = llm([{"role": "user", "content": "Say hello!"}])
    print(f"LLM Response: {result}")
