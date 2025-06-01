import requests
import json
import os

# WARNING: HARDCODED API KEY FOR TESTING PURPOSES ONLY. 
# DO NOT COMMIT THIS TO VERSION CONTROL WITH A REAL KEY.
# Consider using environment variables for production.
HELIUS_API_KEY = "a936080b-6fb5-4399-96d5-e1cc5694f92d" # Replace with your actual key if different, or use env var

MARKET_ID = "32D4zRxNc1EssbJieVHfPhZM3rH6CzfUPrWUuWxD9prG"  # Pool for Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB

BASE_URL = f"https://api.helius.xyz/v0/addresses/{MARKET_ID}/transactions"

def fetch_transactions(api_key: str, params=None):
    """Fetches transactions from Helius Enhanced API."""
    if params is None:
        params = {}
    
    headers = {
        'Content-Type': 'application/json',
    }
    # Add API key to params
    params['api-key'] = api_key

    try:
        print(f"Requesting URL: {BASE_URL} with params: {params}")
        response = requests.get(BASE_URL, headers=headers, params=params)
        response.raise_for_status() # Raises an exception for HTTP errors (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response content: {response.content.decode()}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except json.JSONDecodeError as json_err:
        print(f"JSON decode error: {json_err}")
        print(f"Response content: {response.content.decode()}")
    return None

if __name__ == "__main__":
    print(f"--- Testing Helius Enhanced Transactions API for Market ID: {MARKET_ID} ---")

    # --- Test 1: Fetch recent transactions (no type filter, small limit) ---
    print("\n--- Test 1: Fetching recent transactions (limit=10, no type filter) ---")
    recent_transactions = fetch_transactions(HELIUS_API_KEY, params={"limit": 10})
    if recent_transactions:
        print(json.dumps(recent_transactions, indent=2))
    else:
        print("Failed to fetch recent transactions.")

    # --- Test 2: Fetch SWAP transactions (small limit) ---
    print("\n--- Test 2: Fetching SWAP transactions (limit=10) ---")
    swap_transactions = fetch_transactions(HELIUS_API_KEY, params={"limit": 10, "type": "SWAP"})
    if swap_transactions:
        print(json.dumps(swap_transactions, indent=2))
    else:
        print("Failed to fetch SWAP transactions (or no SWAP transactions found).")
    
    # --- Test 3: Fetch TRANSFER transactions (small limit) ---
    print("\n--- Test 3: Fetching TRANSFER transactions (limit=5) ---")
    transfer_transactions = fetch_transactions(HELIUS_API_KEY, params={"limit": 5, "type": "TRANSFER"})
    if transfer_transactions:
        print(json.dumps(transfer_transactions, indent=2))
    else:
        print("Failed to fetch TRANSFER transactions (or no TRANSFER transactions found).")

    # --- Test 4: Example of fetching transactions before a certain signature (if you have one) ---
    # Replace 'SOME_SIGNATURE_FROM_PREVIOUS_OUTPUT' with an actual signature string
    # print("\n--- Test 4: Fetching transactions before a specific signature (limit=2) ---")
    # signature_to_go_before = "SOME_SIGNATURE_FROM_PREVIOUS_OUTPUT" 
    # # You'd get this signature from the output of Test 1, 2, or 3 (e.g., recent_transactions[0]['signature'])
    # if recent_transactions and len(recent_transactions) > 0 and 'signature' in recent_transactions[0]:
    #     signature_to_go_before = recent_transactions[0]['signature']
    #     print(f"Using signature {signature_to_go_before} for 'before' parameter.")
    #     paginated_transactions = fetch_transactions(HELIUS_API_KEY, params={"limit": 2, "before": signature_to_go_before})
    #     if paginated_transactions:
    #         print(json.dumps(paginated_transactions, indent=2))
    #     else:
    #         print("Failed to fetch paginated transactions.")
    # else:
    #     print("Skipping Test 4 because no signature was available from previous tests.")
        
    print("\n--- Testing finished ---") 