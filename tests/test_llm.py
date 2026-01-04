import os
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

def test_azure():
    api_key = os.getenv("AZURE_API_KEY")
    api_base = os.getenv("AZURE_API_BASE")
    api_version = "2025-01-01-preview" # User provided version
    deployment = "gpt-4.1-mini" # User provided deployment
    
    print(f"Testing Azure OpenAI with:")
    print(f"  Base: {api_base}")
    print(f"  Deployment: {deployment}")
    print(f"  Version: {api_version}")
    
    try:
        response = completion(
            model=f"azure/{deployment}",
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print("Success!")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Failed with error: {e}")

if __name__ == "__main__":
    test_azure()
