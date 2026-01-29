"""
Quick test script to verify Bedrock Claude connection.
Run: python test_bedrock.py
"""
import os
from dotenv import load_dotenv

load_dotenv()

def test_bedrock():
    print("=" * 50)
    print("Testing Bedrock Claude Connection")
    print("=" * 50)
    
    # Check env vars
    region = os.getenv("AWS_REGION", "us-east-1")
    model_id = os.getenv("BEDROCK_MODEL_ID", "")
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    
    print(f"\nConfiguration:")
    print(f"  AWS_REGION: {region}")
    print(f"  BEDROCK_MODEL_ID: {model_id}")
    print(f"  AWS_ACCESS_KEY_ID: {'***' + access_key[-4:] if access_key else 'NOT SET'}")
    print(f"  AWS_SECRET_ACCESS_KEY: {'***' + secret_key[-4:] if secret_key else 'NOT SET'}")
    
    if not model_id:
        print("\n❌ ERROR: BEDROCK_MODEL_ID not set in .env")
        return
    
    print("\n--- Initializing Bedrock Client ---")
    try:
        from src.llm.bedrock_client import BedrockClient
        
        client = BedrockClient(
            region_name=region,
            model_id=model_id,
            max_tokens=256,
            aws_access_key_id=access_key or None,
            aws_secret_access_key=secret_key or None,
        )
        print("✅ Bedrock client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return
    
    print("\n--- Testing chat_completion ---")
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": "Say 'Hello, I am working!' in exactly 5 words."}],
            max_completion_tokens=50,
        )
        print(f"✅ Response: {response}")
    except Exception as e:
        print(f"❌ chat_completion failed: {e}")
        return
    
    print("\n" + "=" * 50)
    print("✅ SUCCESS: Bedrock Claude is working!")
    print("=" * 50)

if __name__ == "__main__":
    test_bedrock()
