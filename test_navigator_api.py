import os
import toml
from openai import OpenAI

def load_key_from_secrets():
    # Attempt to load from .streamlit/secrets.toml
    try:
        if os.path.exists(".streamlit/secrets.toml"):
            secrets = toml.load(".streamlit/secrets.toml")
            return secrets.get("NAVIGATOR_TOOLKIT_API_KEY")
    except Exception as e:
        print(f"Error reading secrets.toml: {e}")
    
    # Fallback to env var
    return os.getenv("NAVIGATOR_TOOLKIT_API_KEY")

def test_api():
    api_key = load_key_from_secrets()
    if not api_key:
        print("❌ No API key found. Please ensure NAVIGATOR_TOOLKIT_API_KEY is in .streamlit/secrets.toml or your environment.")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.ai.it.ufl.edu/v1"
    )

    # 1. List Available Models
    print("\n--- 1. Checking Available Models ---")
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"✅ Connection successful! Found {len(model_ids)} models.")
        print(f"Available models: {', '.join(model_ids)}")
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
        return

    # 2. Test specific model
    target_model = "gemma-3-27b-it" 
    print(f"\n--- 2. Testing Model: {target_model} ---")
    
    try:
        response = client.chat.completions.create(
            model=target_model,
            messages=[{"role": "user", "content": "Hello! Say 'API is working' if you hear me."}],
            max_tokens=20
        )
        print(f"✅ Success! Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Error with model '{target_model}': {e}")
        if target_model not in model_ids:
            print(f"\n💡 PRO-TIP: '{target_model}' is not a valid model ID for this API.")
            print(f"   Try using 'gemma-3-27b-it' or 'llama-3.1-8b-instruct' instead.")

if __name__ == "__main__":
    test_api()
