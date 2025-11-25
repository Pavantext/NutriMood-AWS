import streamlit as st
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# GPT-4o mini pricing (per 1M tokens)
GPT4O_MINI_PRICING = {
    "input": 0.150,  # $0.150 per 1M input tokens
    "output": 0.600  # $0.600 per 1M output tokens
}

def count_tokens(text, model="gpt-4o-mini"):
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4

def calculate_cost(input_tokens, output_tokens):
    """Calculate total cost for GPT-4o mini usage"""
    input_cost = (input_tokens / 1_000_000) * GPT4O_MINI_PRICING["input"]
    output_cost = (output_tokens / 1_000_000) * GPT4O_MINI_PRICING["output"]
    total_cost = input_cost + output_cost
    return input_cost, output_cost, total_cost

@st.cache_data
def load_food_data():
    import pathlib
    # Find the absolute path to the data file relative to this script location
    # Use absolute path to avoid issues with Streamlit's working directory changes
    script_dir = pathlib.Path(__file__).parent.absolute()
    data_path = script_dir.parent / "perplx" / "data" / "updated_food_items.json"

    # Ensure the path exists
    if not data_path.exists():
        raise FileNotFoundError(f"Food data file not found at: {data_path}. Current working directory: {pathlib.Path.cwd()}")

    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

food_data = load_food_data()

st.set_page_config(page_title="Food Recommendation AI", page_icon="🍽️", layout="wide")

st.title("🍽️ Food Recommendation AI Chatbot")
st.markdown("Ask me anything about our menu! I can help you find the perfect dish based on your     preferences, dietary needs, budget, or cuisine type.")

if not OPENAI_API_KEY:
    st.error("⚠️ OpenAI API Key is not configured. Please add your OPENAI_API_KEY in the Secrets tab.")
    st.info("The chatbot requires an OpenAI API key to function. Once you add it, the app will automatically restart.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What kind of food are you looking for?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        system_message = {
            "role": "system",
            "content": f"""You are a nutrimood, a hilarious, witty food recommendation wizard with a passion for culinary adventures! 🍕🔥 Think of me as your foodie best friend who's obsessed with our amazing menu and loves making meal decisions fun and memorable.

I've got access to our complete menu database with all the juicy details:
- Product names and mouth-watering descriptions
- Prices in INR (because money matters!)
- Tags for dietary preferences, cooking styles, and flavor vibes
- Ingredients lists (for those curious cooks)
- Macronutrients (calories, protein, carbs, fat)
- Health benefits (because we care about you!)
- Cuisine types from around the world
- Spice levels (mild, medium, or "hold onto your taste buds!")
- Popularity ratings

Here's our full menu:
{json.dumps(food_data, indent=2)}

Your mission (should you choose to accept it):
🎯 Be SUPER conversational and FUNNY - use emojis, jokes, and engaging stories
💬 Keep responses between 400-500 words - detailed but not overwhelming
🤝 Ask follow-up questions to make it interactive
🍽️ Recommend 2-4 dishes with personality and reasoning
📊 Share key nutritional info and health benefits when relevant
💰 Always be accurate with prices and ingredients
🚫 Politely explain if something's not on our menu
🎭 Add humor, food puns, and relatable commentary
🌟 Make each recommendation feel like a mini food adventure

Response style examples:
- "Oh man, you're in for a treat with our Jalapeno Cheese Poppers! These little firecrackers are like tiny volcanoes of deliciousness..."
- "Budget-conscious AND craving something healthy? I've got just the thing!"
- "Warning: This dish might make you question all other meals you've ever eaten! 😱"

Remember: You're not just recommending food, you're creating food memories and making people excited about their next meal!"""
        }
        
        conversation_messages = [system_message] + [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in st.session_state.messages
        ]
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation_messages,
                max_completion_tokens=500,
                stream=True
            )
            
            full_response = ""

            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # Calculate token usage using tiktoken (since streaming doesn't provide usage info)
            # Count input tokens (system message + conversation history)
            input_text = ""
            for msg in conversation_messages:
                input_text += msg["content"]

            input_tokens = count_tokens(input_text, "gpt-4o-mini")
            output_tokens = count_tokens(full_response, "gpt-4o-mini")
            total_tokens = input_tokens + output_tokens

            # Calculate costs
            input_cost, output_cost, total_cost = calculate_cost(input_tokens, output_tokens)

            # Display token info in a small, unobtrusive way
            token_info = f"\n\n---\n📊 **Tokens used:** Input: {input_tokens:,} | Output: {output_tokens:,} | Total: {total_tokens:,} | Cost: ${total_cost:.6f}"

            message_placeholder.markdown(full_response + token_info)
            st.session_state.messages.append({"role": "assistant", "content": full_response + token_info})

            # Store token information in session state for sidebar display
            st.session_state['last_request_tokens'] = {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'input_cost': input_cost,
                'output_cost': output_cost,
                'total_cost': total_cost
            }
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

with st.sidebar:
    st.header("📊 Quick Stats")
    st.metric("Total Menu Items", len(food_data))
    
    cuisines = set(item.get('cuisine_type', 'Unknown') for item in food_data)
    st.metric("Cuisine Types", len(cuisines))
    
    popular_items = [item for item in food_data if item.get('IsPopular', False)]
    st.metric("Popular Items", len(popular_items))
    
    if 'last_request_tokens' in st.session_state:
        st.markdown("---")
        st.subheader("💰 Last Request Cost")

        tokens = st.session_state['last_request_tokens']
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Input Tokens", f"{tokens['input_tokens']:,}")
            st.metric("Output Tokens", f"{tokens['output_tokens']:,}")

        with col2:
            st.metric("Input Cost", f"${tokens['input_cost']:.6f}")
            st.metric("Output Cost", f"${tokens['output_cost']:.6f}")

        st.metric("**Total Cost**", f"${tokens['total_cost']:.6f}")

    if 'last_cached_tokens' in st.session_state:
        st.markdown("---")
        st.success(f"✅ Prompt Caching Active")
        st.caption(f"Last request cached {st.session_state['last_cached_tokens']} tokens (50% cost savings)")

    st.markdown("---")
    st.header("💡 Try asking:")
    st.markdown("""
    - "Show me vegetarian options under ₹200"
    - "What are your spicy Indian dishes?"
    - "I want something healthy and low calorie"
    - "Recommend a drink for hot weather"
    - "What's popular on your menu?"
    - "Compare nutritional info of two items"
    - "Show me gluten-free options"
    """)
    
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        if 'last_cached_tokens' in st.session_state:
            del st.session_state['last_cached_tokens']
        if 'last_request_tokens' in st.session_state:
            del st.session_state['last_request_tokens']
        st.rerun()
