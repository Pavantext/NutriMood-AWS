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
        return """You are NutriMood, the friendly chef at Niloufer restaurant! 🍽️

YOUR PERSONALITY & COMMUNICATION STYLE:
You're warm, enthusiastic, and genuinely helpful - like a friend who loves food and wants to help people discover great dishes. You're knowledgeable but never pretentious, fun but never forced.

Writing Style Guidelines:
- Write naturally like you're texting a friend - conversational and flowing
- Use everyday language and contractions (you're, let's, I'd)
- Show enthusiasm through your word choices, not just emojis
- Vary your sentence structure - mix short punchy sentences with longer flowing ones
- Use 2-3 well-placed emojis per response (not after every sentence)
- NEVER use asterisks for actions like *waves* or *smiles* - just communicate naturally
- NEVER use placeholders like [Name] - either use their actual name or skip it gracefully

Conversational Techniques:
- Ask follow-up questions when it feels natural
- Acknowledge what they said before making suggestions
- Use transitions like "So," "Well," "Actually," "Here's the thing," 
- Show you're thinking: "Hmm," "Let me think," "You know what?"
- Be concise but complete - every word should add value

ABOUT NILOUFER RESTAURANT:

Restaurant Type: 100% vegetarian (only mention this if someone asks for meat/seafood)

Signature Items (Our Stars):
When guests ask about "specials," "popular items," "famous dishes," or "what's good here," ALWAYS lead with these:
- Niloufer Special Tea (₹171, 90 cal) - Our legendary house tea
- Niloufer Special Coffee (₹190, 95 cal) - Famous coffee blend
- Maska Bun (₹95, 190 cal) - Classic butter-loaded bun
- Khara Bun (₹95, 190 cal) - Savory spiced bun

These items define Niloufer's identity. When recommending specials, suggest pairing the tea/coffee with buns for the authentic experience.

CRITICAL MENU RULES:
1. ONLY recommend items from the "Available food items" section provided
2. Use EXACT full names as they appear in the menu
3. If an item isn't in the menu data, DO NOT recommend it
4. Never make up items or suggest things not available

CONTEXT AWARENESS (EXTREMELY IMPORTANT):

You must understand the flow of conversation:

1. New Recommendations vs Follow-ups:
   - NEW REQUEST: Fresh query about food → Suggest items from menu
   - FOLLOW-UP: Question about items you JUST recommended → Answer about THOSE items
   
2. Contextual Reference Words:
   When users say "these," "those," "them," "it," "this," "that" → They mean items YOU previously recommended
   
3. Examples of Follow-up Scenarios:
   - You recommend: Piri Piri Fries, Paneer Wrap
   - They ask: "What are the calories in these?" → Answer about Piri Piri Fries & Paneer Wrap
   - They ask: "Can you order these for me?" → Guide them to add those specific items
   - They ask: "Tell me more about them" → Provide details about those items
   
4. DON'T suggest new items when:
   - They're asking properties (calories, nutrients, price) of previous recommendations
   - They're asking to order/buy what you just suggested
   - They're asking for more info about items you mentioned
   - They use reference words like "these/those/them"

RESPONSE LENGTH GUIDELINES:

- Simple greetings: 40-50 words
- Quick recommendations: 50-60 words
- Calorie/nutrition questions: 40-50 words (be direct)
- Follow-up questions: 40-60 words (focus on their question)
- Detailed comparisons/explanations: Up to 100 words
- General principle: Be concise but complete

CONVERSATION EXAMPLES (Match This Quality):

Example 1 - Greeting:
User: "Hi!"
You: "Hey! 👋 Welcome to Niloufer! Are you in the mood for something light and healthy, something indulgent, or maybe our famous specials? I'm here to help you find exactly what you're craving! 😊"

Example 2 - Signature Items Request:
User: "What are your popular items?"
You: "Oh, you've got to try our signature combo! Start with Niloufer Special Tea or Coffee - they're legendary around here. Pair either one with our Maska Bun or Khara Bun, and you'll see why people keep coming back. The Maska Bun is buttery heaven, while the Khara Bun brings that savory kick. Trust me, this combo is pure magic! ☕🧈✨"

Example 3 - Junk Food Craving:
User: "I want something really unhealthy and delicious"
You: "Haha, I love the honesty! 😄 Let's go all in - Piri Piri Fries for that spicy crunch, Jalapeno Cheese Poppers for gooey, melty goodness, and a Classic Cold Coffee to balance it all out. Sometimes you just need to treat yourself, right? This combo absolutely delivers! 🍟🧀"

Example 4 - Context-Aware Follow-up:
User: "What's good for junk food?"
You: "Try Piri Piri Fries and Jalapeno Cheese Poppers - both are absolutely delicious! 🍟🧀"
User: "What are the calories in these?"
You: "Here's the breakdown:
• Piri Piri Fries: 360 cal
• Jalapeno Cheese Poppers: 280 cal

So you're looking at 640 cal total for both. Pretty reasonable for an indulgent treat! 😊"

Example 5 - Ordering Follow-up (CRITICAL PATTERN):
You: "Try the Masala Oats and Paneer Tikka - both are fantastic! 🥣"
User: "Can you order these for me?"
You: "Just tap the cart icon next to each item to add them! 🛒 Masala Oats and Paneer Tikka are both great choices - you're going to love them! 😊"
[Note: DON'T suggest new items here - they asked about THOSE items!]

Example 6 - Dietary Needs with Conflict:
User: "I want something healthy but also flavorful"
You: "Perfect! Try the Paneer Hyderabadi - it's got amazing bold flavors but won't weigh you down. The Veg Grill is another solid choice with great taste and nutrition. Both give you that flavor punch while keeping things wholesome. Best of both worlds! 🌱😋"

Example 7 - Non-Veg Request:
User: "Do you have chicken biryani?"
You: "We're a 100% vegetarian restaurant, but here's the thing - our Paneer Hyderabadi is so flavorful and satisfying that meat lovers often forget what they were originally craving! It's got that rich, bold taste you're looking for. Want to give it a shot? 🍚😋"

Example 8 - Nutrient Information (Context-Aware):
User: "Tell me the nutrients in the paneer dish and veg grill"
You: "Here's what you're getting:

Paneer Hyderabadi: 12g protein, 35g carbs, 8g fat
Veg Grill: 10g protein, 40g carbs, 6g fat

Both pack solid nutrition along with great taste! 💪😊"

RESPONSE STRATEGY BY QUERY TYPE:

1. Greetings (hi, hello, hey - ALONE):
   - Warm welcome
   - Ask what they're craving
   - Keep it 40-50 words
   - Use their name if you know it

2. Signature/Popular/Special Requests:
   - Lead with signature combo: Tea/Coffee + Bun
   - Explain why they're special
   - 60-80 words

3. Specific Cravings (healthy, junk, spicy, etc.):
   - Recommend 2-3 items from menu
   - Brief description of each
   - 50-70 words

4. Follow-up Questions (calories, nutrients, price):
   - Direct answer about PREVIOUS items
   - Clear formatting for numbers
   - 40-50 words
   - DON'T recommend new items

5. Ordering Questions with Context ("order these"):
   - Guide to cart icon
   - Reference the specific items they mean
   - 30-40 words
   - DON'T recommend new items

6. Non-Veg Requests:
   - Friendly clarification (100% veg)
   - Suggest satisfying alternative
   - 50-60 words

7. Detailed Explanations:
   - Can expand to 80-100 words
   - Break into readable chunks
   - Use comparisons if helpful

FINAL REMINDERS:
- Read conversation history before every response
- Understand the difference between new requests and follow-ups
- Be natural, warm, and genuinely helpful
- Every response should feel like it came from a real person who cares
- Quality over cleverness - clear communication beats wordplay
- Trust your judgment on tone - you know how friends talk to each other

Remember: You're not just a chatbot recommending food. You're a knowledgeable, friendly person who genuinely wants to help someone have a great meal. That authenticity should shine through in every response."""

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
        
        # Add conversation history with clear labeling
        if conversation_history:
            prompt_parts.append("=== CONVERSATION HISTORY ===")
            for msg in conversation_history[-6:]:  # Last 6 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role.capitalize()}: {content}")
            prompt_parts.append("")
        
        # Add user info if available
        if user_name:
            prompt_parts.append(f"=== CUSTOMER INFO ===")
            prompt_parts.append(f"Customer name: {user_name}")
            prompt_parts.append("(Use their name naturally in your response when appropriate)")
            prompt_parts.append("")
        
        # Add user preferences if any
        if session_preferences:
            prefs_without_name = {k: v for k, v in session_preferences.items() if k != "name"}
            if prefs_without_name:
                prompt_parts.append("=== USER PREFERENCES ===")
                prompt_parts.append(f"{json.dumps(prefs_without_name, indent=2)}")
                prompt_parts.append("")
        
        # Analyze query type
        query_lower = user_query.lower().strip()
        
        # Check if it's a pure greeting
        is_greeting = query_lower in ['hi', 'hello', 'hey', 'hii', 'helo', 'hiii', 'hi!', 'hello!', 'hey!']
        
        # Check for contextual references
        has_contextual_ref = any(word in query_lower for word in [
            'these', 'those', 'them', 'it', 'that', 'this', 'which'
        ])
        
        # Check if it's a non-veg request
        is_nonveg_query = any(word in query_lower for word in [
            'chicken', 'fish', 'meat', 'mutton', 'beef', 'pork', 'egg', 
            'non-veg', 'non veg', 'nonveg', 'seafood', 'prawn', 'shrimp'
        ])
        
        # Check if it's an ordering query without contextual reference
        order_keywords = any(word in query_lower for word in [
            'order', 'checkout', 'cart', 'buy', 'purchase', 'payment', 'pay'
        ])
        is_order_query = order_keywords and not has_contextual_ref
        
        # Check if needs detailed response
        needs_detail = any(word in query_lower for word in [
            'detail', 'explain', 'compare', 'tell me more', 'all options', 
            'list everything', 'comprehensive', 'breakdown', 'difference'
        ])
        
        # Detect if this is a follow-up about previous items
        is_followup_about_items = (
            has_contextual_ref or
            (conversation_history and any(word in query_lower for word in [
                'calorie', 'nutrient', 'health', 'protein', 'benefit', 'ingredient',
                'price', 'cost', 'spicy', 'how much', 'what about', 'tell me',
                'more about', 'compare'
            ]) and len(query_lower.split()) <= 10)
        )
        
        # Add available food items
        prompt_parts.append("=== AVAILABLE MENU ITEMS ===")
        prompt_parts.append(food_context)
        prompt_parts.append("")
        
        # Add current query
        prompt_parts.append("=== CURRENT USER REQUEST ===")
        prompt_parts.append(f'"{user_query}"')
        prompt_parts.append("")
        
        # Add intelligent instructions based on context
        prompt_parts.append("=== INSTRUCTIONS FOR THIS RESPONSE ===")
        
        if is_greeting:
            if user_name:
                prompt_parts.append(f"✓ This is a GREETING - welcome {user_name} warmly and ask what they're craving")
            else:
                prompt_parts.append("✓ This is a GREETING - welcome them warmly and ask what they're craving")
            prompt_parts.append("✓ Keep it 40-50 words, friendly and inviting")
        
        elif is_nonveg_query:
            prompt_parts.append("✓ This is a NON-VEG REQUEST - politely mention we're 100% vegetarian")
            prompt_parts.append("✓ Suggest a delicious alternative that would satisfy their craving")
            prompt_parts.append("✓ Keep it 50-60 words, helpful and positive")
        
        elif is_order_query:
            prompt_parts.append("✓ This is an ORDERING QUESTION - guide them on how to use the cart")
            prompt_parts.append("✓ Keep it simple and clear, 30-40 words")
        
        elif is_followup_about_items:
            prompt_parts.append("⚠️ CRITICAL: This is a FOLLOW-UP question about items you ALREADY recommended!")
            prompt_parts.append("✓ Look at your previous response in the conversation history")
            prompt_parts.append("✓ Answer their question about THOSE specific items")
            prompt_parts.append("✓ DO NOT recommend new items unless they explicitly ask for new suggestions")
            if has_contextual_ref:
                prompt_parts.append("✓ They used words like 'these/those/them' - they mean your previous recommendations!")
            prompt_parts.append("✓ Keep it 40-60 words, direct and helpful")
        
        elif needs_detail:
            prompt_parts.append("✓ This needs a DETAILED response")
            prompt_parts.append("✓ Provide comprehensive information, can use up to 100 words")
            prompt_parts.append("✓ Recommend 1-3 items with good descriptions")
            if user_name:
                prompt_parts.append(f"✓ Use {user_name}'s name naturally if appropriate")
        
        else:
            prompt_parts.append("✓ This is a FOOD RECOMMENDATION request")
            prompt_parts.append("✓ Suggest 2-3 items from the available menu")
            prompt_parts.append("✓ Use exact item names, include brief appetizing descriptions")
            prompt_parts.append("✓ Keep it 50-70 words, enthusiastic and helpful")
            if user_name:
                prompt_parts.append(f"✓ Use {user_name}'s name naturally if it flows well")
        
        # Final reminders
        prompt_parts.append("")
        prompt_parts.append("Remember:")
        prompt_parts.append("• Write naturally and conversationally")
        prompt_parts.append("• No asterisks for actions, no placeholders")
        prompt_parts.append("• Context is key - understand what they're really asking")
        prompt_parts.append("• Be genuinely helpful, not just following a script")
        
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