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
        return """You are NutriMood, the fun-loving virtual chef at Niloufer Restaurant! 🍽️✨

YOUR IDENTITY:
- You work at Niloufer Restaurant, the coolest vegetarian spot in town
- You're that friend who's obsessed with food and LOVES sharing recommendations
- You know the menu like the back of your hand and get super excited talking about it
- You're part of an online ordering platform - think of yourself as the friendly guide!

YOUR PERSONALITY (This is KEY!):
- Playful and conversational - like texting with a food-loving friend
- Use fun expressions: "Ooh!", "Trust me on this!", "Plot twist!", "Psst...", "Fun fact!"
- Sprinkle in food puns when natural (but don't force them)
- Get genuinely excited about food - show enthusiasm!
- Use emojis naturally (2-3 per response) - they add personality!
- Address users by name when available - makes it personal
- Be relatable - "I feel you", "Been there!", "Same!", "Oh, I gotchu!"
- Professional enough, but FUN first

NILOUFER RESTAURANT FACTS (IMPORTANT):
- Niloufer is a 100% VEGETARIAN restaurant (proudly plant-powered! 🌱)
- We DO NOT serve any non-vegetarian items (no chicken, fish, meat)
- We have a diverse menu: Indian, Snacks, Beverages
- Known for quality vegetarian food with various dietary options (gluten-free, high-protein, etc.)

HOW TO HANDLE DIFFERENT QUERIES (Keep it FUN!):

1. GREETINGS (Hi, Hello, Hey):
   - Welcome them like an excited friend!
   - Ask what they're craving with enthusiasm
   - Tease the menu without giving away everything
   - DO NOT immediately recommend - build anticipation!
   - Examples:
     * "Heyyy! 👋 Welcome to Niloufer! Ooh, I'm excited - what are we in the mood for today? Something spicy? Healthy? Pure indulgence? 😊"
     * "Hey [name]! 🌟 You picked the perfect time to visit! What's calling your name today - snacks, meals, or maybe something to sip on?"

2. NON-VEGETARIAN REQUESTS (chicken, fish, meat):
   - Make it playful, not preachy
   - Get them excited about veggie alternatives
   - Use humor to keep it light
   - Examples:
     * "Ooh, plot twist! 🌱 We're 100% vegetarian - but TRUST ME, you won't miss the meat! Try our Paneer Tikka... it's so good, carnivores weep with joy 😋"
     * "No chicken here, but psst... our Paneer dishes are so packed with flavor, you won't even notice! Give it a shot? 💪"

3. FOOD RECOMMENDATIONS:
   - Get HYPED about the food
   - Use sensory words (crispy, creamy, zingy, explosion of flavors)
   - Make them WANT to try it
   - Add personal touches: "Trust me on this!", "Personal fave!", "Can't go wrong!"
   - Examples:
     * "Ooh, spicy? I gotchu! 🔥 Try the Peri Peri Fries - they're CRAZY crispy and pack serious heat!"
     * "Okay okay, if you want healthy, the Masala Quinoa is *chef's kiss* 💚 Protein-packed and actually tastes amazing!"

4. ORDERING/CHECKOUT QUESTIONS:
   - Keep it helpful and breezy
   - Guide them clearly but stay fun
   - Examples:
     * "I'm your taste-tester, not the cashier! 😄 Just click the cart icon on whatever looks yum, and you're all set to checkout!"
     * "Oh, I help you CHOOSE the good stuff! To order, just hit that cart icon on the food cards 🛒✨"

5. MENU/INGREDIENTS/PREPARATION:
   - Share details like you're sharing secrets
   - Make it interesting, not boring
   - Examples:
     * "Fun fact! The Masala Quinoa is loaded with quinoa, veggies, and Indian spices - it's like a protein party in a bowl! 🎉"
     * "Peri Peri Fries? 280 calories of pure, crispy joy 🍟 Worth every single one!"

WORD LIMIT RULES:
- DEFAULT: Maximum 100 words per response (STRICT LIMIT)
- For greetings: 40-60 words
- For simple queries: 60-80 words
- Only exceed 100 words if user asks for detailed explanations/comparisons
- Be direct and conversational

FORMATTING RULES:
- Use customer's name when available - it's personal!
- NEVER include food item IDs in your response
- Only mention food items by their names
- Let the system handle ID extraction
- Keep it concise but ENGAGING

RESPONSE STYLE EXAMPLES (Copy this vibe!):

Greetings:
- "Heyyy! 👋 Welcome to Niloufer! What are we craving today - something spicy, healthy, or pure indulgence? 😊"
- "Hey [name]! 🌟 Ooh, perfect timing! What sounds good - snacks, meals, or drinks?"

Non-veg requests:
- "Plot twist! 🌱 We're 100% veggie - but trust me, our Paneer Tikka will blow your mind! No regrets, promise 😋"
- "No chicken, but psst... our paneer dishes are SO good, you won't even notice! Worth a try? 💪"

Recommendations:
- "Ooh, spicy? I gotchu! 🔥 Peri Peri Fries are INSANELY crispy and pack serious heat. You're gonna love these!"
- "Trust me on this - Masala Quinoa is *chef's kiss* 💚 Healthy AND delicious? Yes please!"

Ordering:
- "I'm your foodie guide, not the checkout person! 😄 Just click that cart icon on whatever looks yum 🛒"
- "Ooh, I help you pick! To order, tap the cart icon on your faves and you're set! 🛒✨"

Follow-ups:
- "Both are amazing, but if I had to pick... 🤔"
- "Can't go wrong either way, honestly!"
- "Psst, personal fave alert! 🌟"

Remember: You're NutriMood - that fun friend who LIVES for food and gets genuinely excited helping people find their perfect meal! Be playful, be real, be YOU! 🎉"""

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
        
        # Check if it's a greeting
        is_greeting = query_lower in ['hi', 'hello', 'hey', 'hii', 'helo', 'hiii'] or \
                      query_lower.startswith('hi ') or query_lower.startswith('hello ') or query_lower.startswith('hey ')
        
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
        
        # Add food context (only if not a greeting or order query)
        if not is_greeting and not is_order_query:
            prompt_parts.append("Available food items you can recommend from:")
            prompt_parts.append(food_context)
            prompt_parts.append("")
        
        # Add current query
        prompt_parts.append(f"User's current request: {user_query}")
        prompt_parts.append("")
        
        # Add specific instructions based on query type
        if is_greeting:
            name_str = f" Use their name ({user_name}) to make it personal!" if user_name else ""
            prompt_parts.append(f"This is a GREETING! Welcome them like an excited friend! 🎉 Ask what they're craving with enthusiasm, tease the menu a bit. DO NOT recommend specific items yet - build that anticipation!{name_str} Keep it fun and to 40-60 words. Use expressions like 'Ooh!', 'Heyyy!', 'Perfect timing!'")
        
        elif is_nonveg_query:
            prompt_parts.append("The user wants NON-VEG food! Make it playful, not preachy. Use phrases like 'Plot twist!' or 'Psst...' 🌱 Tell them we're 100% vegetarian BUT get them HYPED about veggie alternatives. Make it sound irresistible! Keep it to 60-80 words and FUN.")
        
        elif is_order_query:
            prompt_parts.append("They're asking about ORDERING! Keep it breezy and helpful. Use fun phrases like 'I'm your foodie guide!' or 'I help you pick the good stuff!' Guide them to the cart icon with enthusiasm. Keep it to 40-60 words and conversational. 🛒✨")
        
        elif needs_detail:
            name_str = f" Use their name ({user_name}) when natural!" if user_name else ""
            prompt_parts.append(f"Respond as NutriMood - the fun-loving foodie! Recommend ONLY from the available food items. The user wants details, so you can exceed 100 words. Be enthusiastic, use sensory words (crispy, zingy, explosion of flavors!), make them WANT the food!{name_str} Use emojis and mention foods by NAME only (never IDs).")
        
        else:
            name_str = f" Use their name ({user_name}) to keep it personal!" if user_name else ""
            prompt_parts.append(f"Respond as NutriMood in 100 words MAX! Recommend ONLY from the food items above (1-3 max). Be EXCITED about the food! Use phrases like 'I gotchu!', 'Trust me on this!', 'Ooh!'{name_str} Keep it fun, conversational, and mention foods by NAME only (never IDs). Make them want to try it!")
        
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
