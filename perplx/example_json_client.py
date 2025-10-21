"""
Simple example client for JSON chat endpoint
Demonstrates clean JSON response format
"""

import requests
import json

class NutrimoodJSONClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client"""
        self.base_url = base_url
        self.session_id = None
    
    def chat(self, message: str) -> dict:
        """
        Send a chat message and get clean JSON response
        
        Args:
            message: The user's message
        
        Returns:
            Dictionary with message, session_id, and food_recommendation_id
        """
        payload = {
            "message": message
        }
        
        if self.session_id:
            payload["session_id"] = self.session_id
        
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Store session ID for next request
            self.session_id = result.get("session_id")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return {"error": str(e)}
    
    def print_response(self, response: dict):
        """Pretty print the response"""
        if "error" in response:
            print(f"❌ Error: {response['error']}")
        else:
            print(f"\n📱 Response:")
            print(f"Session ID: {response.get('session_id')}")
            print(f"Food IDs: {response.get('food_recommendation_id')}")
            print(f"\n💬 Message:\n{response.get('message')}")
            print("-" * 60)


def main():
    """Example usage"""
    print("=" * 60)
    print("🍽️  NutriMood Chatbot - JSON Client Example")
    print("=" * 60)
    print()
    
    client = NutrimoodJSONClient()
    
    # Example 1: First message
    print("Example 1: Initial greeting")
    print("User: Hi")
    response = client.chat("Hi")
    client.print_response(response)
    
    # Example 2: Food request
    print("\nExample 2: Specific food request")
    print("User: I want something spicy and healthy")
    response = client.chat("I want something spicy and healthy")
    client.print_response(response)
    
    # Example 3: Follow-up question (with session context)
    print("\nExample 3: Follow-up question")
    print("User: How many calories are in those?")
    response = client.chat("How many calories are in those?")
    client.print_response(response)
    
    # Print session info
    print("\n" + "=" * 60)
    print(f"Session ID: {client.session_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()



