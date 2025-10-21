"""
Bedrock Service - Handles AWS Bedrock LLM interactions with streaming support
"""

import boto3
import json
from typing import List, Dict, AsyncGenerator
import asyncio
from botocore.exceptions import ClientError

class BedrockService:
    def __init__(self):
        """Initialize AWS Bedrock client"""
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'  # Change to your region
        )
        
        # Model configuration
        self.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"  # or use Claude 3.5 Sonnet
        self.model_config = {
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop_sequences": []
        }
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for NutriMood chatbot personality"""
        return """You are NutriMood, the fun chef at Niloufer! 🍽️

TONE & STYLE (Match this exactly!):
- Casual, friendly, and playful - like texting a friend
- Use fun phrases: "yum!", "No worries!", "Win-win!", "Perfect!", "crunchy goodness"
- Creative food descriptions: "yum-fest", "crunch party", "gooey goodness", "bold flavors"
- Keep it light and conversational
- 2-3 emojis per response (use them well!)
- NEVER use asterisks for actions (*waves*, *smiles*) - just talk naturally!
- If no name given, skip it or say "friend" - NEVER use [Name]

NILOUFER:
- 100% vegetarian restaurant (only mention if they ask for meat/chicken/fish)
- Our signature items: Niloufer Special Tea, Niloufer Special Coffee, Maska
- When asked about "specials" or "popular" items, prioritize recommending these!

CRITICAL RULES:
- Max 50-60 words for simple recommendations
- Max 40 words for calorie/info questions  
- NO stage directions in asterisks
- NO food IDs in your text
- NO placeholders like [Name]
- Be direct and fun
- READ CONVERSATION HISTORY! If user asks follow-ups like "what nutrients does it have", "how many calories", "tell me more" etc. - they're asking about items YOU JUST recommended in your last response. Stay on topic!

RESPONSE EXAMPLES (Copy this style!):

Greeting:
"Hey! 👋 Welcome to Niloufer! What are you craving today - spicy, healthy, or pure indulgence? 😊"

Junk food request:
"For junk food lovers, try Peri Peri Fries (spicy, yum!), Corn Cheese Balls (gooey goodness), and Niloufer Spl Lassi to cool it down! 😋🍟🥤"

Calorie question:
"Here's the calorie count:
Peri Peri Fries: 360 cal 🍟
Corn Cheese Balls: 320 cal 🧀
Niloufer Spl Lassi: 120 cal 🥤
Perfect for tasty indulgence without the math headache! 😄"

Dietary conflict:
"How about Paneer Hyderabadi? Tasty, wholesome, perfect for healthy vibes AND bold flavors! Pair with Curd for smooth finish. Win-win! 🍚💚😋"

No rice:
"No rice? No worries! Try Aloo Fry for crispy junk vibes and Mixed Veg Salad for healthy crunch. Perfect crunchy combo! 🥔🥗😄"

Follow-up:
"Aloo Fry is satisfying, but adding Peri Peri Fries would crank up the yum-fest! Your partner gets healthy, you get the crunch party! 🎉🍟"

Nutrients question (CONTEXT-AWARE):
"Paneer Hyderabadi: Protein 12g, Carbs 35g, Fat 8g 💪
Veg Grill: Protein 10g, Carbs 40g, Fat 6g
Both pack good nutrients! 😊"

Non-veg:
"We're 100% veggie! But our Paneer Tikka is so good, you won't even miss it. Trust me! 🌱😋"

Specials/Popular request:
"Our chef specials? You GOTTA try Niloufer Special Tea, Niloufer Special Coffee, and Maska - they're legendary! 😋☕🧈"

Remember: Short, fun, natural conversation. STAY ON TOPIC for follow-ups - talk about items you JUST recommended! No asterisks, no placeholders, just vibes!"""

    def _build_prompt(
        self,
        user_query: str,
        conversation_history: List[Dict],
        food_context: str,
        session_preferences: Dict
    ) -> str:
        """Build the complete prompt with context"""
        
        prompt_parts = []
        
        # Extract user name if available
        user_name = session_preferences.get("name", "") if session_preferences else ""
        
        # Add conversation history
        if conversation_history:
            prompt_parts.append("Previous conversation:")
            for msg in conversation_history[-6:]:  # Last 6 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role.capitalize()}: {content}")
            prompt_parts.append("")
        
        # Add user info if available
        if user_name:
            prompt_parts.append(f"Customer's name: {user_name} (use this in your response)")
            prompt_parts.append("")
        
        # Add user preferences if any
        if session_preferences:
            prefs_without_name = {k: v for k, v in session_preferences.items() if k != "name"}
            if prefs_without_name:
                prompt_parts.append(f"User preferences: {json.dumps(prefs_without_name)}")
            prompt_parts.append("")
        
        # Detect query type
        query_lower = user_query.lower().strip()
        
        # Check if it's a greeting (must be ONLY greeting, not with food request)
        is_greeting = query_lower in ['hi', 'hello', 'hey', 'hii', 'helo', 'hiii', 'hi!', 'hello!', 'hey!']
        
        # If starts with greeting but has more words, check if it's a food request
        if not is_greeting and any(query_lower.startswith(g + ' ') for g in ['hi', 'hello', 'hey']):
            # Check if there are food-related words after greeting
            words = query_lower.split()
            if len(words) > 1:
                # Has content after greeting - treat as food request, not greeting
                is_greeting = False
            else:
                is_greeting = True
        
        # Check if it's a non-veg request
        is_nonveg_query = any(word in query_lower for word in [
            'chicken', 'fish', 'meat', 'mutton', 'beef', 'pork', 'egg', 
            'non-veg', 'non veg', 'nonveg', 'seafood', 'prawn', 'shrimp'
        ])
        
        # Check if it's an ordering query
        is_order_query = any(word in query_lower for word in [
            'order', 'checkout', 'cart', 'buy', 'purchase', 'payment', 'pay'
        ])
        
        # Check if needs detailed response
        needs_detail = any(word in query_lower for word in [
            'detail', 'explain', 'compare', 'tell me more', 'all options', 
            'list everything', 'comprehensive', 'breakdown'
        ])
        
        # Detect if this is a follow-up question about properties
        is_followup_about_items = any(word in query_lower for word in [
            'calorie', 'nutrient', 'health', 'protein', 'benefit', 'ingredient',
            'price', 'cost', 'spicy', 'how much', 'what about', 'tell me',
            'more about', 'which one', 'compare'
        ]) and len(query_lower.split()) <= 8
        
        # Add food context (only if not a greeting or order query)
        if not is_greeting and not is_order_query:
            if is_followup_about_items and conversation_history:
                prompt_parts.append("IMPORTANT: User is asking a FOLLOW-UP question about items you JUST recommended in your last message!")
                prompt_parts.append("Look at your previous response and answer about THOSE items, not the search results below.")
                prompt_parts.append("")
            
        prompt_parts.append("Available food items you can recommend from:")
        prompt_parts.append(food_context)
        prompt_parts.append("")
        
        # Add current query
        prompt_parts.append(f"User's current request: {user_query}")
        prompt_parts.append("")
        
        # Add specific instructions based on query type
        if is_greeting:
            if user_name:
                prompt_parts.append(f"GREETING: Say 'Hey {user_name}!' or 'Hey there {user_name}!', then ask what they're craving. 40-50 words. Be warm and friendly!")
            else:
                prompt_parts.append("GREETING: Say 'Hey!' or 'Welcome!', ask what they're craving. Skip the name. 40-50 words. Be warm!")
        
        elif is_nonveg_query:
            prompt_parts.append("NON-VEG: Say 'We're 100% veggie!' then suggest alternatives. Be fun! 50-60 words.")
        
        elif is_order_query:
            prompt_parts.append("ORDER: Say 'Click the cart icon to add items!' Keep it simple. 30-40 words.")
        
        elif needs_detail:
            if user_name:
                prompt_parts.append(f"Use {user_name}'s name if natural. Recommend 1-3 items. Can go up to 100 words. Be fun!")
            else:
                prompt_parts.append("Recommend 1-3 items. Can go up to 100 words. Be fun!")
        
        else:
            if user_name:
                prompt_parts.append(f"Use {user_name}'s name naturally. Recommend 1-3 items. 50-70 words. Be fun! No IDs!")
            else:
                prompt_parts.append("Recommend 1-3 items. 50-70 words. Be fun! Skip name or say 'friend'. No IDs!")
        
        return "\n".join(prompt_parts)
    
    async def generate_streaming_response(
        self,
        user_query: str,
        conversation_history: List[Dict],
        food_context: str,
        session_preferences: Dict
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from AWS Bedrock
        Yields text chunks as they arrive
        """
        try:
            # Build the complete prompt
            prompt = self._build_prompt(
                user_query,
                conversation_history,
                food_context,
                session_preferences
            )
            
            system_prompt = self._build_system_prompt()
            
            # Prepare request body for Claude
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.model_config["max_tokens"],
                "temperature": self.model_config["temperature"],
                "top_p": self.model_config["top_p"],
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            # Invoke model with streaming
            response = self.client.invoke_model_with_response_stream(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            # Process streaming response
            stream = response.get('body')
            if stream:
                for event in stream:
                    chunk = event.get('chunk')
                    if chunk:
                        chunk_data = json.loads(chunk.get('bytes').decode())
                        
                        # Handle different chunk types
                        if chunk_data.get('type') == 'content_block_delta':
                            delta = chunk_data.get('delta', {})
                            if delta.get('type') == 'text_delta':
                                text = delta.get('text', '')
                                yield text
                        
                        elif chunk_data.get('type') == 'message_stop':
                            break
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            yield f"Sorry, I encountered an error: {error_code} - {error_message}"
            
        except Exception as e:
            yield f"Oops! Something went wrong: {str(e)}"
    
    async def generate_response(
        self,
        user_query: str,
        conversation_history: List[Dict],
        food_context: str,
        session_preferences: Dict
    ) -> str:
        """
        Generate non-streaming response (for internal use)
        """
        full_response = ""
        async for chunk in self.generate_streaming_response(
            user_query,
            conversation_history,
            food_context,
            session_preferences
        ):
            full_response += chunk
        
        return full_response
    
    def extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON data from LLM response if present"""
        try:
            # Try to find JSON blocks in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx + 1]
                return json.loads(json_str)
            
            return {}
        except:
            return {}
