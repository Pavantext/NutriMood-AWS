"""
Response Formatter - Utilities for formatting API responses
"""

from typing import Dict, List, Optional
import re

class ResponseFormatter:
    def __init__(self):
        """Initialize response formatter"""
        pass
    
    def format_chat_response(
        self,
        message: str,
        session_id: str,
        food_ids: List[str]
    ) -> Dict:
        """Format standard chat response"""
        return {
            "message": message,
            "session_id": session_id,
            "food_recommendation_id": ",".join(food_ids) if food_ids else ""
        }
    
    def format_error_response(self, error_message: str, session_id: Optional[str] = None) -> Dict:
        """Format error response"""
        response = {
            "message": f"Sorry, something went wrong: {error_message}",
            "food_recommendation_id": ""
        }
        
        if session_id:
            response["session_id"] = session_id
        
        return response
    
    def format_no_match_response(self, session_id: str) -> Dict:
        """Format response when no food items match"""
        return {
            "message": "I couldn't find a food item matching your request. Would you like to try something else? Maybe describe what you're craving in a different way! 😊",
            "session_id": session_id,
            "food_recommendation_id": ""
        }
    
    def clean_response_text(self, text: str) -> str:
        """Clean and format response text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure proper spacing around emojis
        text = re.sub(r'([!?.])([\U0001F300-\U0001F9FF])', r'\1 \2', text)
        
        return text
    
    def extract_food_mentions(self, text: str, food_items: List[Dict]) -> List[str]:
        """Extract mentioned food item IDs from text"""
        text_lower = text.lower()
        mentioned_ids = []
        
        for food in food_items:
            food_name = food.get('name', '').lower()
            food_id = food.get('id')
            
            if food_name in text_lower:
                mentioned_ids.append(food_id)
        
        return mentioned_ids
    
    def format_food_details(self, food: Dict) -> str:
        """Format food item details for display"""
        parts = [
            f"**{food.get('name')}**",
            f"Category: {food.get('category', 'N/A')}",
        ]
        
        if food.get('description'):
            parts.append(f"Description: {food.get('description')}")
        
        if food.get('calories'):
            parts.append(f"Calories: {food.get('calories')}")
        
        return " | ".join(parts)
    
    def create_greeting_response(self, session_id: str) -> Dict:
        """Create greeting response for new users"""
        return {
            "message": "Hey there! I'm NutriMood, your friendly food buddy! 🍽️ Tell me what you're craving, your mood, or what kind of food you're looking for, and I'll hook you up with perfect recommendations! What sounds good to you?",
            "session_id": session_id,
            "food_recommendation_id": ""
        }
