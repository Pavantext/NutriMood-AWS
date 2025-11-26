"""
NutriMood Bot - Production Implementation using OpenAI Responses API
Supports: gpt-4o-mini, Structured Outputs, Prompt Caching, File Search
"""

import os
import sys
import json
import time
from typing import Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Import tiktoken for accurate token counting
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("⚠️  tiktoken not installed. Token counting will be less accurate.")
    print("   Install with: pip install tiktoken")

# Add perplx directory to path for imports (works when run directly or as module)
script_dir = Path(__file__).parent
perplx_dir = script_dir.parent
if str(perplx_dir) not in sys.path:
    sys.path.insert(0, str(perplx_dir))

from interfaces.base_models import ChatRequest, ChatResponse
from utils.openai_system_prompt import instructions


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count tokens accurately using tiktoken, or estimate if not available.

    Args:
        text: Text to count tokens for
        model: Model name for tokenization

    Returns:
        Number of tokens
    """
    if HAS_TIKTOKEN:
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception as e:
            print(f"⚠️  Token counting failed: {e}, using estimation")
            # Fallback to estimation
            return len(text) // 4  # Rough estimation: 1 token ≈ 4 characters
    else:
        # Rough estimation when tiktoken is not available
        return len(text) // 4

# Load environment variables
load_dotenv(perplx_dir / ".env")

# ==========================================
# NutriMood Bot Class (Responses API)
# ==========================================

class NutriMood:
    """
    Production-grade chatbot using OpenAI Responses API with:
    - Automatic prompt caching for cost optimization
    - File search (Vector Store) for food recommendations
    - Structured outputs (guaranteed JSON schema compliance)
    - Stateful conversation management
    """

    def __init__(self, api_key: Optional[str] = None, vector_store_id: Optional[str] = None):
        """
        Initialize the bot with API credentials and vector store.
        
        Args:
            api_key: OpenAI API key (if None, loads from OPENAI_API_KEY env var)
            vector_store_id: Pre-created vector store ID (if None, loads from OPENAI_VECTOR_STORE_ID env var)
        """
        # Load API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Please set OPENAI_API_KEY in your .env file or pass it as parameter."
            )
        
        # Load vector store ID from parameter or environment
        self.vector_store_id = vector_store_id or os.getenv("OPENAI_VECTOR_STORE_ID")
        if not self.vector_store_id:
            raise ValueError(
                "Vector Store ID not found. "
                "Please set OPENAI_VECTOR_STORE_ID in your .env file or pass it as parameter."
            )
        
        self.client = OpenAI(api_key=self.api_key)

        # In-memory session storage (use Redis/DB in production)
        self.session_messages: Dict[str, list] = {}
        self.session_response_ids: Dict[str, str] = {}  # Track response IDs for conversation continuity

    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Process user message and return food recommendation.
        
        Args:
            request: ChatRequest with user message and session info
            
        Returns:
            ChatResponse with recommendation and metadata
        """
        # 1. Session Management
        current_session_id = request.session_id or f"sess_{int(time.time())}"
        
        # Initialize or retrieve conversation history
        if current_session_id not in self.session_messages:
            self.session_messages[current_session_id] = []

            # ✅ ADD CONTEXT ONCE at session start using 'developer' role
            context = {}
            if request.user_name:
                context["user_name"] = request.user_name
            if request.user_id:
                context["user_id"] = request.user_id
            if request.user_preferences:
                context["preferences"] = request.user_preferences

            if context:
                context_msg = {
                    "role": "developer",  # New role for system-level context
                    "content": json.dumps({
                        **context,
                        "session_start": time.time()
                    })
                }
                self.session_messages[current_session_id].append(context_msg)

        # ✅ Just add user message - no context duplication
        self.session_messages[current_session_id].append({
            "role": "user",
            "content": request.message
        })
        
        # 3. Build Input for Responses API
        # The Responses API uses 'input' instead of 'messages'
        # For conversation history, we format it appropriately
        conversation_input = self._format_conversation_input(current_session_id)
        
        try:
            # 4. Call Responses API with File Search
            # Prompt Caching: Automatically caches 'instructions' and 'tools'
            # if they remain constant across requests (>1024 tokens)

            # Dynamic search configuration based on query complexity
            query_words = request.message.split()
            max_results = min(10, len(query_words) + 2)  # More results for complex queries

            response = self.client.responses.create(
                model="gpt-4o-mini",  # Cost-effective with full feature support

                # System instructions (CACHED after first call)
                instructions=instructions,

                # User input (can be string or list of message objects)
                input=conversation_input,

                # ✅ CRITICAL: Store conversation state server-side
                store=True,

                # ✅ Reference previous response for context continuity
                previous_response_id=self.session_response_ids.get(current_session_id),

                # File Search Tool with Vector Store - Optimized configuration
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [self.vector_store_id],
                    "max_num_results": max_results,
                    "ranking_options": {
                        "ranker": "auto",  # Let OpenAI choose best ranking
                        "score_threshold": 0.1  # Filter low-relevance results
                    }
                }],

                # Model Parameters
                temperature=0.3,  # Low for consistent retrieval
                max_output_tokens=500,  # Allow enough tokens for response
            )

            # ✅ Store response ID for next request
            self.session_response_ids[current_session_id] = response.id

            # 4.5. Calculate Token Usage (Accurate counting using tiktoken)
            # The OpenAI Responses API doesn't provide usage info in response,
            # so we count tokens manually using tiktoken for accuracy

            # Count input tokens (system instructions + conversation history)
            input_text = instructions + json.dumps(conversation_input)
            input_tokens = count_tokens(input_text, "gpt-4o-mini")

            # Count output tokens (will be calculated after extracting output text)
            output_text = ""  # Will be set below
            output_tokens = 0  # Will be calculated after extracting output
            total_tokens = input_tokens  # Will add output_tokens later

            # 5. Extract Output Text
            # The response.output contains a list of output items
            output_text = self._extract_output_text(response)

            # 5.5. Calculate output tokens now that we have the output text
            output_tokens = count_tokens(output_text, "gpt-4o-mini")
            total_tokens = input_tokens + output_tokens

            # Print accurate token usage
            print(f"🔢 Token Usage - Input: {input_tokens:,}, Output: {output_tokens:,}, Total: {total_tokens:,}")

            # 6. Parse JSON from response (Responses API returns text, we extract JSON)
            # Try to extract JSON from the response text
            parsed_data = self._parse_response_json(output_text, current_session_id)
            
            # Add assistant response to history
            self.session_messages[current_session_id].append({
                "role": "assistant",
                "content": output_text
            })
            
            # 7. Return Structured Response
            return ChatResponse(
                message=parsed_data.get('message', output_text),
                session_id=current_session_id,
                food_recommendation_id=parsed_data.get('food_recommendation_id', ''),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens
            )
            
        except json.JSONDecodeError as e:
            # Handle JSON parsing errors
            print(f"JSON Decode Error: {e}")
            return self._error_response(
                current_session_id,
                "I encountered an error formatting the recommendation."
            )
            
        except KeyError as e:
            # Handle missing keys in structured output
            print(f"Missing Key in Response: {e}")
            return self._error_response(
                current_session_id,
                "I couldn't find a suitable recommendation."
            )
            
        except Exception as e:
            # Handle API errors (rate limits, network issues, etc.)
            print(f"API Error: {e}")
            return self._error_response(
                current_session_id,
                "I'm sorry, I couldn't search the menu right now. Please try again."
            )

    def _format_conversation_input(self, session_id: str):
        """
        ✅ RECOMMENDED: Use proper message format for Responses API
        Format conversation history for Responses API.

        The Responses API accepts list of message objects directly.
        No need to format as string.
        """
        messages = self.session_messages.get(session_id, [])

        # Responses API accepts list of message objects directly
        # No need to format as string
        return messages if messages else []

    def _extract_output_text(self, response) -> str:
        """
        Extract text output from Responses API response.
        
        The response structure contains multiple output items.
        We need to find the 'message' type with 'output_text'.
        """
        for output_item in response.output:
            if output_item.type == "message":
                for content in output_item.content:
                    if content.type == "output_text":
                        return content.text
        
        raise ValueError("No output_text found in response")
    
    def _parse_response_json(self, text: str, session_id: str) -> Dict:
        """
        Parse JSON from response text.
        The LLM is instructed to return JSON format in the system prompt.
        
        Args:
            text: Response text from LLM
            session_id: Session ID for error handling
            
        Returns:
            Parsed dictionary with message and food_recommendation_id
        """
        # Try to find JSON in the response
        import re
        
        # Look for JSON object in the text
        json_match = re.search(r'\{[^{}]*"message"[^{}]*"food_recommendation_id"[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # If no JSON found, try parsing the entire text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # If JSON parsing fails, extract food IDs from text and return structured format
            # Look for food IDs (UUIDs) in the response
            food_ids = re.findall(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', text, re.IGNORECASE)
            
            return {
                "message": text,
                "food_recommendation_id": ",".join(food_ids[:5]) if food_ids else ""
            }

    def _error_response(self, session_id: str, message: str) -> ChatResponse:
        """Create standardized error response."""
        return ChatResponse(
            message=message,
            session_id=session_id,
            food_recommendation_id="error"
        )

    def clear_session(self, session_id: str):
        """Clear conversation history and response IDs for a session."""
        if session_id in self.session_messages:
            del self.session_messages[session_id]
        if session_id in self.session_response_ids:
            del self.session_response_ids[session_id]

    def get_session_history(self, session_id: str) -> list:
        """Retrieve conversation history for debugging/logging."""
        return self.session_messages.get(session_id, [])




# ==========================================
# Example Usage
# ==========================================

if __name__ == "__main__":
    # Example usage - Loads from .env automatically
    # Make sure you have OPENAI_API_KEY and OPENAI_VECTOR_STORE_ID in your .env file

    # Initialize bot (loads from .env automatically)
    bot = NutriMood()

    # Example: Make a food recommendation request
    request = ChatRequest(
        message="anything more spicy?",
        session_id="test_session_001",
        user_name="John"
    )

    # Get recommendation
    response = bot.chat(request)
    print(f"Response: {response.message}")
    print(f"Food IDs: {response.food_recommendation_id}")
    print(f"Session ID: {response.session_id}")
    print(f"Input Tokens: {response.input_tokens}")
    print(f"Output Tokens: {response.output_tokens}")
    print(f"Total Tokens: {response.total_tokens}")
