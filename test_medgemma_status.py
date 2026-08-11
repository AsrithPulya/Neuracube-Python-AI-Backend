import os
import httpx
from dotenv import load_dotenv

def test_medgemma():
    # 1. Load environment variables from the .env file
    load_dotenv()

    api_url = os.getenv("MEDGEMMA_API_URL", "https://dr7.ai/api/v1/medical/chat/completions")
    api_key = os.getenv("MEDGEMMA_API_KEY", "")

    print("=" * 60)
    print("MEDGEMMA API STATUS CHECKER")
    print("=" * 60)
    print(f"Target URL: {api_url}")
    print(f"API Key configured: {'Yes (loaded successfully)' if api_key else 'No (missing)'}")
    print("-" * 60)

    if not api_key:
        print("ERROR: MEDGEMMA_API_KEY is not defined in your environment or .env file.")
        print("Please check your .env file in this directory.")
        print("=" * 60)
        return

    # Payload to send to MedGemma
    payload = {
        "model": "medgemma-4b-it",
        "messages": [
            {
                "role": "user",
                "content": "Hello, are you online? Respond with a short confirmation."
            }
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    print("Sending request to MedGemma...")
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(api_url, json=payload, headers=headers)
        
        print(f"HTTP Response Status Code: {response.status_code}")
        print("-" * 60)
        
        if response.status_code == 200:
            print("🟢 SUCCESS: MedGemma API is ONLINE!")
            print("Response:")
            try:
                data = response.json()
                print(data['choices'][0]['message']['content'])
            except Exception:
                print(response.text)
        elif response.status_code == 401:
            print("🔴 FAILED: Authentication Error (401 Unauthorized)")
            print("Your API key is invalid or has expired.")
            print(f"Response: {response.text}")
        elif response.status_code == 500:
            print("🔴 FAILED: Internal Server Error (500)")
            print("The MedGemma upstream service authenticated properly but failed internally.")
            print("This means the dr7.ai service is currently DOWN or experiencing issues.")
            print(f"Response: {response.text}")
        else:
            print(f"🟡 UNKNOWN STATUS ({response.status_code})")
            print(f"Response: {response.text}")

    except httpx.ConnectError:
        print("🔴 FAILED: Connection Error")
        print("Could not connect to dr7.ai. Check your internet connection or URL.")
    except Exception as e:
        print(f"🔴 FAILED: An unexpected error occurred")
        print(f"Details: {e}")

    print("=" * 60)

if __name__ == "__main__":
    test_medgemma()
