"""
Example Client for Nutrimood Chatbot
Demonstrates how to interact with the chatbot API
"""

import requests
import json
import time
from typing import Optional

class NutrimoodClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client"""
        self.base_url = base_url
        self.session_id = None
    
    def chat(self, message: str, stream: bool = True) -> dict:
        """
        Send a chat message and get streaming response
        
        Args:
            message: The user's message
            stream: Whether to display streaming text (always streams from server)
        
        Returns:
            Response dictionary with message and metadata
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
                stream=True  # Always stream from server
            )
            response.raise_for_status()
            
            # Handle streaming response
            full_response = ""
            final_json = None
            
            if stream:
                print("NutriMood: ", end="", flush=True)
            
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
                    
                    # Check for final JSON response
                    if "__RESPONSE__:" in chunk_str:
                        # Extract the response part before __RESPONSE__
                        parts = chunk_str.split("__RESPONSE__:")
                        if parts[0] and stream:
                            print(parts[0], end="", flush=True)
                            full_response += parts[0]
                        
                        # Parse the JSON
                        json_part = parts[1]
                        final_json = json.loads(json_part)
                        break
                    else:
                        if stream:
                            print(chunk_str, end="", flush=True)
                        full_response += chunk_str
            
            if stream:
                print("\n")  # New line after response
            
            if final_json:
                self.session_id = final_json.get("session_id")
                return final_json
            else:
                return {
                    "message": full_response,
                    "session_id": self.session_id,
                    "food_recommendation_id": ""
                }
        
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return {"error": str(e)}
    
    def get_recommendations(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[dict] = None
    ) -> dict:
        """Get food recommendations without conversation"""
        payload = {
            "query": query,
            "top_k": top_k
        }
        
        if filters:
            payload["filters"] = filters
        
        try:
            response = requests.post(
                f"{self.base_url}/recommend",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return {"error": str(e)}
    
    def list_foods(
        self, 
        category: Optional[str] = None,
        limit: int = 50
    ) -> dict:
        """List available food items"""
        params = {"limit": limit}
        if category:
            params["category"] = category
        
        try:
            response = requests.get(
                f"{self.base_url}/foods",
                params=params
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return {"error": str(e)}
    
    def get_session_history(self) -> dict:
        """Get current session history"""
        if not self.session_id:
            return {"error": "No active session"}
        
        try:
            response = requests.get(
                f"{self.base_url}/session/{self.session_id}"
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return {"error": str(e)}
    
    def clear_session(self):
        """Clear current session"""
        if self.session_id:
            try:
                requests.delete(f"{self.base_url}/session/{self.session_id}")
            except:
                pass
        self.session_id = None


def demo_conversation():
    """Demonstrate a conversation flow"""
    print("=" * 60)
    print("🍽️  Nutrimood Chatbot Demo")
    print("=" * 60)
    print()
    
    client = NutrimoodClient()
    
    # Example 1: Junk food request
    print("Example 1: Junk Food Request")
    print("-" * 60)
    print("You: chefs specials for this kind of weather for very junk food lovers")
    client.chat("chefs specials for this kind of weather for very junk food lovers")
    time.sleep(1)
    print()
    
    # Example 2: Follow-up about calories
    print("Example 2: Calorie Inquiry (Context-Aware)")
    print("-" * 60)
    print("You: how many calories are these?")
    client.chat("how many calories are these?")
    time.sleep(1)
    print()
    
    # Example 3: Conflicting preferences
    print("Example 3: Handling Conflicting Preferences")
    print("-" * 60)
    print("You: my partner is on a healthy diet and i love junk food, what dish can u suggest that we both can share")
    client.chat("my partner is on a healthy diet and i love junk food, what dish can u suggest that we both can share")
    time.sleep(1)
    print()
    
    # Example 4: Dietary restriction
    print("Example 4: Dietary Restriction")
    print("-" * 60)
    print("You: ohh partner doesn't eat rice")
    client.chat("ohh partner doesn't eat rice")
    time.sleep(1)
    print()
    
    # Example 5: Appetite question
    print("Example 5: Appetite Consideration")
    print("-" * 60)
    print("You: would that fill the junk guys appetite")
    client.chat("would that fill the junk guys appetite")
    time.sleep(1)
    print()
    
    # Show session history
    print("Session Summary")
    print("-" * 60)
    history = client.get_session_history()
    if "messages" in history:
        print(f"Total messages: {history['message_count']}")
        print(f"Session ID: {history['session_id']}")
    
    print()
    print("=" * 60)


def interactive_mode():
    """Interactive chat mode"""
    print("=" * 60)
    print("🍽️  Nutrimood Chatbot - Interactive Mode")
    print("=" * 60)
    print("Type 'quit' to exit, 'history' to view session, 'clear' to reset")
    print()
    
    client = NutrimoodClient()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Goodbye! 👋")
                break
            
            if user_input.lower() == 'history':
                history = client.get_session_history()
                if "error" not in history:
                    print(f"\nSession ID: {history.get('session_id')}")
                    print(f"Messages: {history.get('message_count')}")
                    print(f"Recommendations: {len(history.get('recommendations', []))}")
                else:
                    print("No active session")
                print()
                continue
            
            if user_input.lower() == 'clear':
                client.clear_session()
                print("Session cleared!\n")
                continue
            
            # Send message
            client.chat(user_input)
            print()
        
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def demo_recommendations():
    """Demonstrate direct recommendation API"""
    print("=" * 60)
    print("🔍  Direct Recommendation API Demo")
    print("=" * 60)
    print()
    
    client = NutrimoodClient()
    
    # Example 1: Simple query
    print("Example 1: Healthy options")
    result = client.get_recommendations("healthy food")
    print(f"Found {result.get('count')} recommendations:")
    for rec in result.get('recommendations', [])[:3]:
        print(f"  - {rec['name']} ({rec['calories']} cal) - {rec['category']}")
    print()
    
    # Example 2: With filters
    print("Example 2: Low calorie snacks")
    result = client.get_recommendations(
        "snacks",
        filters={"category": "Snacks", "max_calories": 300}
    )
    print(f"Found {result.get('count')} recommendations:")
    for rec in result.get('recommendations', []):
        print(f"  - {rec['name']} ({rec['calories']} cal)")
    print()
    
    # Example 3: List foods by category
    print("Example 3: All beverages")
    result = client.list_foods(category="Beverages")
    print(f"Found {result.get('count')} beverages:")
    for food in result.get('foods', []):
        print(f"  - {food['name']} ({food.get('calories', 'N/A')} cal)")
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "demo":
            demo_conversation()
        elif mode == "interactive":
            interactive_mode()
        elif mode == "recommend":
            demo_recommendations()
        else:
            print("Usage: python example_client.py [demo|interactive|recommend]")
    else:
        # Default to demo
        demo_conversation()
        print("\nTo try interactive mode, run: python example_client.py interactive")
