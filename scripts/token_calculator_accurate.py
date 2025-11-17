"""
Accurate Token Usage Calculator for NutriMood
Uses actual code to build prompts and real tokenizer for accurate counts
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict

# Add parent directory to Python path to allow imports from perplx
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try to import tiktoken for accurate Claude tokenization
try:
    import tiktoken
    HAS_TIKTOKEN = True
    # Claude uses cl100k_base encoding
    encoding = tiktoken.get_encoding("cl100k_base")
except ImportError:
    HAS_TIKTOKEN = False
    print("⚠️  tiktoken not installed. Install with: pip install tiktoken")
    print("   Falling back to estimation method.\n")

def count_tokens(text: str) -> int:
    """Count tokens accurately using tiktoken, or estimate if not available"""
    if not text:
        return 0
    
    if HAS_TIKTOKEN:
        # Accurate token count using Claude's actual tokenizer
        return len(encoding.encode(text))
    else:
        # Fallback estimation
        word_count = len(text.split())
        char_count = len(text)
        tokens_by_words = int(word_count * 1.3)
        tokens_by_chars = int(char_count / 4)
        return int((tokens_by_words + tokens_by_chars) / 2)

def calculate_real_token_usage(num_responses: int = 10):
    """Calculate token usage using actual code and real data"""
    
    # Import here to avoid errors if dependencies missing
    from perplx.services.bedrock_service import BedrockService
    from perplx.services.food_service import FoodService
    
    bedrock_service = BedrockService()
    food_service = FoodService()
    
    # Load actual food data
    food_data_path = os.getenv("FOOD_DATA_PATH", "../data/raw/Niloufer_data.json")
    possible_paths = [
        food_data_path,
        "data/Niloufer_data.json",
        "../data/raw/Niloufer_data.json",
        "data/food_items.json"
    ]
    
    food_loaded = False
    for path in possible_paths:
        try:
            if os.path.exists(path):
                food_service.load_food_data(path)
                if food_service.food_items:
                    food_loaded = True
                    print(f"✅ Loaded {len(food_service.food_items)} food items from {path}")
                    break
        except Exception as e:
            continue
    
    if not food_loaded:
        print("⚠️  Could not load food data. Using estimation for food context.")
    
    # Get actual system prompt
    system_prompt = bedrock_service._build_system_prompt()
    system_tokens = count_tokens(system_prompt)
    
    print("=" * 80)
    print("NUTRIMOOD ACCURATE TOKEN CALCULATION")
    print("=" * 80)
    print(f"\nSystem Prompt:")
    print(f"  Characters: {len(system_prompt):,}")
    print(f"  Words: {len(system_prompt.split()):,}")
    print(f"  {'Actual Tokens' if HAS_TIKTOKEN else 'Estimated Tokens'}: {system_tokens:,}")
    
    # Sample queries for testing
    sample_queries = [
        "I want something spicy",
        "What are the calories in these?",
        "Tell me more about them",
        "What's good for breakfast?",
        "I want healthy food",
        "What are your popular items?",
        "Can you order these for me?",
        "Tell me about yourself",
        "What's the price?",
        "Recommend something"
    ]
    
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Average output tokens (40 words target)
    avg_output_text = "Hey! 👋 What sounds good - healthy bites, total indulgence, or our legendary specials? Let's find you something amazing! 😊"
    avg_output_tokens = count_tokens(avg_output_text)
    
    print(f"\nPer-Response Calculation:")
    print(f"  Average Output Example: {len(avg_output_text.split())} words")
    print(f"  {'Actual Tokens' if HAS_TIKTOKEN else 'Estimated Tokens'}: {avg_output_tokens:,}")
    
    # Simulate conversation
    conversation_history = []
    
    for i in range(num_responses):
        user_query = sample_queries[i % len(sample_queries)]
        
        # Find matching foods using actual service
        if food_loaded:
            food_matches = food_service.find_matching_foods(
                query=user_query,
                conversation_history=conversation_history,
                top_k=5
            )
            # Build actual food context
            food_context = food_service.build_food_context(food_matches)
        else:
            # Fallback if food data not loaded
            food_context = "No food data available"
        
        # Build actual user prompt using the real method
        user_prompt = bedrock_service._build_prompt(
            user_query=user_query,
            conversation_history=conversation_history[-6:] if conversation_history else [],
            food_context=food_context,
            session_preferences={}
        )
        
        # Count actual tokens
        user_prompt_tokens = count_tokens(user_prompt)
        total_input_tokens += system_tokens + user_prompt_tokens
        total_output_tokens += avg_output_tokens
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": user_query})
        conversation_history.append({"role": "assistant", "content": avg_output_text})
        
        # Keep only last 6 messages (as per code)
        if len(conversation_history) > 6:
            conversation_history = conversation_history[-6:]
        
        if i == 0 or i == num_responses - 1:
            print(f"\n  Response {i+1}:")
            print(f"    User Query: '{user_query}'")
            print(f"    Conversation History: {len(conversation_history) - 2 if i > 0 else 0} messages")
            print(f"    Food Context: {len(food_matches) if food_loaded else 0} items")
            print(f"    User Prompt Tokens: {user_prompt_tokens:,}")
            print(f"    Total Input Tokens: {system_tokens + user_prompt_tokens:,}")
            print(f"    Output Tokens: {avg_output_tokens:,}")
    
    print("\n" + "=" * 80)
    print(f"TOTAL FOR {num_responses} RESPONSES:")
    print("=" * 80)
    print(f"\nInput Tokens:")
    print(f"  Total: {total_input_tokens:,}")
    print(f"  Average per response: {total_input_tokens // num_responses:,}")
    
    print(f"\nOutput Tokens:")
    print(f"  Total: {total_output_tokens:,}")
    print(f"  Average per response: {avg_output_tokens:,}")
    
    print(f"\nTotal Tokens:")
    print(f"  Total: {total_input_tokens + total_output_tokens:,}")
    print(f"  Average per response: {(total_input_tokens + total_output_tokens) // num_responses:,}")
    
    print("\n" + "=" * 80)
    print("BREAKDOWN:")
    print("=" * 80)
    print(f"\nSystem Prompt (sent every request):")
    print(f"  {system_tokens:,} tokens × {num_responses} = {system_tokens * num_responses:,} tokens")
    
    avg_user_prompt = (total_input_tokens - (system_tokens * num_responses)) // num_responses
    print(f"\nUser Prompt (average per request):")
    print(f"  ~{avg_user_prompt:,} tokens")
    
    # Calculate breakdown from actual last prompt
    if food_loaded:
        sample_food_context = food_service.build_food_context(food_matches[:5])
        food_context_tokens = count_tokens(sample_food_context)
        print(f"  - Food context (5 items): ~{food_context_tokens:,} tokens")
    
    print(f"  - Conversation history: ~{avg_user_prompt - food_context_tokens - count_tokens(user_query) - 150 if food_loaded else 'N/A'} tokens")
    print(f"  - User query: ~{count_tokens(user_query):,} tokens")
    print(f"  - Instructions & formatting: ~150 tokens")
    
    print(f"\nOutput (average per response):")
    print(f"  ~{avg_output_tokens:,} tokens")
    
    if not HAS_TIKTOKEN:
        print("\n⚠️  NOTE: Using token estimation. For accurate counts, install tiktoken:")
        print("   pip install tiktoken")
    
    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "avg_input_per_response": total_input_tokens // num_responses,
        "avg_output_per_response": avg_output_tokens,
        "avg_total_per_response": (total_input_tokens + total_output_tokens) // num_responses,
        "uses_real_tokenizer": HAS_TIKTOKEN,
        "uses_real_food_data": food_loaded
    }

if __name__ == "__main__":
    results = calculate_real_token_usage(10)
    
    print("\n" + "=" * 80)
    print("SUMMARY FOR 10 RESPONSES:")
    print("=" * 80)
    print(f"Total Input Tokens:  {results['total_input_tokens']:,}")
    print(f"Total Output Tokens: {results['total_output_tokens']:,}")
    print(f"Total Tokens:        {results['total_tokens']:,}")
    print(f"\nAverage per Response:")
    print(f"  Input:  {results['avg_input_per_response']:,} tokens")
    print(f"  Output: {results['avg_output_per_response']:,} tokens")
    print(f"  Total:  {results['avg_total_per_response']:,} tokens")
    
    print(f"\nAccuracy:")
    print(f"  Using Real Tokenizer: {'✅ Yes' if results['uses_real_tokenizer'] else '❌ No (estimated)'}")
    print(f"  Using Real Food Data: {'✅ Yes' if results['uses_real_food_data'] else '❌ No (estimated)'}")

