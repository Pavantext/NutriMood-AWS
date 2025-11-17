"""
Standalone MCP Server Integration with Claude 3 Sonnet
This server uses AWS Bedrock to call Claude 3 Sonnet and connects to the MCP server
to search Pinecone for food recommendations.

Usage:
    # Interactive CLI mode
    python server.py
    
    # Programmatic usage
    from server import MCPClaudeServer
    
    server = MCPClaudeServer()
    response = server.process_query("I want something spicy and healthy")
    print(response)

Requirements:
    - AWS credentials configured (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    - PINECONE_API_KEY in environment variables
    - PINECONE_INDEX_NAME in environment variables
    - mcp_server.py in services/ directory
"""

import os
import json
import logging
import boto3
import time
import asyncio
import threading
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# Import cost calculator
from utils.cost_calculator import BedrockCostCalculator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Global Rate Limiter (Thread-safe, shared across all requests)
# ============================================================================

class GlobalRateLimiter:
    """Thread-safe global rate limiter for Bedrock API calls"""
    
    def __init__(self, rpm: int = None):
        """
        Initialize rate limiter
        
        Args:
            rpm: Requests per minute (defaults to BEDROCK_RPM_LIMIT env var or 50)
        """
        self.rpm = rpm or int(os.getenv("BEDROCK_RPM_LIMIT"))
        self.min_interval = 60.0 / self.rpm  # Minimum seconds between requests
        self.last_request_time = 0.0
        self.lock = threading.Lock()
        logger.info(f"🔒 Global Rate Limiter initialized: {self.rpm} RPM ({self.min_interval:.2f}s between requests)")
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limit"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                logger.info(f"⏸️  Global rate limit: Waiting {wait_time:.2f}s (RPM: {self.rpm})")
                time.sleep(wait_time)
                self.last_request_time = time.time()
            else:
                self.last_request_time = current_time


# Global rate limiter instance (shared across all server instances)
_global_rate_limiter = GlobalRateLimiter()  # RPM from BEDROCK_RPM_LIMIT env var or defaults to 50


class MCPClaudeServer:
    """Server that integrates Claude 3 Sonnet with MCP tools for Pinecone search"""
    
    def __init__(self):
        """Initialize the server with AWS Bedrock and MCP server connection"""
        # AWS Bedrock configuration
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.getenv("AWS_DEFAULT_REGION")
        )
        
        # Prefer inference profile if provided, otherwise fall back to direct model ID
        self.model_id = os.getenv("BEDROCK_INFERENCE_PROFILE_ID")
        self.model_config = {
            "max_tokens": int(os.getenv("BEDROCK_MAX_TOKENS")),
            "temperature": float(os.getenv("BEDROCK_TEMPERATURE")),
            "top_p": float(os.getenv("BEDROCK_TOP_P"))
        }
        
        # MCP Server configuration
        self.mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8001")
        self.mcp_tools_cache = None
        
        # Use global rate limiter (shared across all instances)
        self.rate_limiter = _global_rate_limiter
        
        # Initialize cost calculator
        self.cost_calculator = BedrockCostCalculator(use_batch_pricing=False)
        
        logger.info(f"✅ MCP Claude Server initialized")
        logger.info(f"🤖 Model: {self.model_id}")
        logger.info(f"🔗 MCP Server: {self.mcp_server_url}")
    
    def _get_mcp_tools(self) -> List[Dict]:
        """Fetch available tools from MCP server"""
        if self.mcp_tools_cache:
            return self.mcp_tools_cache
        
        try:
            # Try to get tools from MCP server via HTTP
            # Note: This assumes MCP server exposes tools via HTTP endpoint
            # If not available, we'll define tools directly from mcp_server.py
            
            # For now, define tools based on mcp_server.py
            tools = [
                {
                    "name": "search_food_by_description",
                    "description": "Semantic search for food items by description or preference. Use this when user asks for food recommendations, specific types of food, or food preferences.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language description (e.g., 'healthy breakfast with protein', 'spicy vegetarian food')"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5, max: 10)",
                                "default": 5
                            },
                            "namespace": {
                                "type": "string",
                                "description": "Pinecone namespace to search in",
                                "default": "default"
                            },
                            "include_metadata": {
                                "type": "boolean",
                                "description": "Include food metadata in results",
                                "default": True
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "search_food_by_category",
                    "description": "Search food items filtered by category (breakfast, lunch, dinner, snacks, beverages, desserts, healthy, comfort-food, vegetarian, vegan).",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Food category (breakfast, lunch, dinner, snacks, beverages, desserts, healthy, comfort-food, vegetarian, vegan)"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 10
                            },
                            "namespace": {
                                "type": "string",
                                "description": "Pinecone namespace",
                                "default": "default"
                            }
                        },
                        "required": ["category"]
                    }
                },
                {
                    "name": "search_by_mood",
                    "description": "Find food recommendations based on mood (happy, energetic, calm, focused, sad, stressed, etc.). Use this when user mentions their mood or emotional state.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "mood": {
                                "type": "string",
                                "description": "Target mood (happy, energetic, calm, focused, sad, stressed, etc.)"
                            },
                            "query": {
                                "type": "string",
                                "description": "Optional natural language query to refine search"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results",
                                "default": 5
                            },
                            "namespace": {
                                "type": "string",
                                "description": "Pinecone namespace",
                                "default": "default"
                            }
                        },
                        "required": ["mood"]
                    }
                },
                {
                    "name": "get_food_details",
                    "description": "Retrieve detailed information about a specific food item by its ID.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "item_id": {
                                "type": "string",
                                "description": "Unique identifier of the food item"
                            },
                            "namespace": {
                                "type": "string",
                                "description": "Pinecone namespace",
                                "default": "default"
                            }
                        },
                        "required": ["item_id"]
                    }
                },
                {
                    "name": "list_all_food_items",
                    "description": "List all food items in the index (with pagination). Use this when user asks to see all available items or browse the menu.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "namespace": {
                                "type": "string",
                                "description": "Pinecone namespace to list from",
                                "default": "default"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum items to return (max 1000)",
                                "default": 100
                            }
                        }
                    }
                }
            ]
            
            self.mcp_tools_cache = tools
            logger.info(f"📋 Loaded {len(tools)} MCP tools")
            return tools
            
        except Exception as e:
            logger.error(f"Error fetching MCP tools: {str(e)}")
            return []
    
    def _call_mcp_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """Call an MCP tool and return the result"""
        try:
            # Import helper functions from mcp_server.py
            import importlib.util
            
            # Get the path to mcp_server.py
            current_dir = os.path.dirname(os.path.abspath(__file__))
            mcp_server_path = os.path.join(current_dir, 'services', 'mcp_server.py')
            
            if not os.path.exists(mcp_server_path):
                raise FileNotFoundError(f"MCP server file not found at {mcp_server_path}")
            
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location("mcp_server", mcp_server_path)
            mcp_server_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mcp_server_module)
            
            # Get helper functions
            get_index = mcp_server_module.get_index
            
            # Custom format_search_results that handles Usage object serialization
            def format_search_results(results):
                """Format search results for LLM consumption."""
                formatted_matches = []
                
                # Handle both dict and object responses
                matches = results.get("matches", []) if isinstance(results, dict) else getattr(results, "matches", [])
                
                for match in matches:
                    match_id = match.get("id") if isinstance(match, dict) else getattr(match, "id", None)
                    match_score = match.get("score", 0) if isinstance(match, dict) else getattr(match, "score", 0)
                    match_metadata = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
                    
                    formatted_match = {
                        "id": match_id,
                        "score": round(match_score, 4),
                        "metadata": match_metadata
                    }
                    formatted_matches.append(formatted_match)
                
                # Handle usage - convert Usage object to dict if needed
                usage = results.get("usage", {}) if isinstance(results, dict) else getattr(results, "usage", None)
                if usage:
                    # Convert Usage object to dict
                    if hasattr(usage, "__dict__"):
                        usage_dict = {
                            "read_units": getattr(usage, "read_units", 0),
                            "write_units": getattr(usage, "write_units", 0)
                        }
                    elif isinstance(usage, dict):
                        usage_dict = usage
                    else:
                        usage_dict = {}
                else:
                    usage_dict = {}
                
                namespace = results.get("namespace", "default") if isinstance(results, dict) else getattr(results, "namespace", "default")
                
                return {
                    "matches": formatted_matches,
                    "namespace": namespace,
                    "usage": usage_dict
                }
            
            # Execute tool logic directly (avoiding FastMCP wrapper)
            if tool_name == "search_food_by_description":
                query = arguments.get("query", "")
                top_k = min(arguments.get("top_k", 5), 10)  # Reduced max from 20 to 10
                namespace = arguments.get("namespace", "default")
                include_metadata = arguments.get("include_metadata", True)
                
                if not query or len(query.strip()) < 2:
                    return {"error": "Query must be at least 2 characters"}
                
                index = get_index()
                results = index.query(
                    vector=[0] * 1536,
                    text=query,
                    namespace=namespace,
                    top_k=top_k,
                    include_metadata=include_metadata,
                    include_values=False
                )
                formatted_results = format_search_results(results)
                logger.info(f"Search query: '{query}' returned {len(formatted_results['matches'])} results")
                return formatted_results
                
            elif tool_name == "search_food_by_category":
                category = arguments.get("category", "")
                top_k = arguments.get("top_k", 10)
                namespace = arguments.get("namespace", "default")
                
                index = get_index()
                filter_condition = {"category": {"$eq": category}}
                
                results = index.query(
                    vector=[0] * 1536,
                    top_k=top_k,
                    namespace=namespace,
                    filter=filter_condition,
                    include_metadata=True,
                    include_values=False
                )
                formatted_results = format_search_results(results)
                logger.info(f"Category search '{category}' returned {len(formatted_results['matches'])} items")
                return formatted_results
                
            elif tool_name == "search_by_mood":
                mood = arguments.get("mood", "")
                query = arguments.get("query")
                top_k = arguments.get("top_k", 5)
                namespace = arguments.get("namespace", "default")
                
                search_query = query or f"Food for {mood} mood"
                index = get_index()
                filter_condition = {"mood_tags": {"$in": [mood]}}
                
                results = index.query(
                    vector=[0] * 1536,
                    text=search_query,
                    namespace=namespace,
                    top_k=top_k,
                    filter=filter_condition,
                    include_metadata=True,
                    include_values=False
                )
                formatted_results = format_search_results(results)
                logger.info(f"Mood search for '{mood}' returned {len(formatted_results['matches'])} items")
                return formatted_results
                
            elif tool_name == "get_food_details":
                item_id = arguments.get("item_id", "")
                namespace = arguments.get("namespace", "default")
                
                index = get_index()
                result = index.fetch(
                    ids=[item_id],
                    namespace=namespace
                )
                
                if not result.vectors or item_id not in result.vectors:
                    return {
                        "error": f"Food item '{item_id}' not found",
                        "item_id": item_id
                    }
                
                food_item = result.vectors[item_id]
                return {
                    "id": food_item.id,
                    "metadata": food_item.metadata,
                    "found": True
                }
                
            elif tool_name == "list_all_food_items":
                namespace = arguments.get("namespace", "default")
                limit = min(arguments.get("limit", 100), 1000)
                
                index = get_index()
                results = index.list(
                    namespace=namespace,
                    limit=limit
                )
                
                item_ids = [item for item in results]
                
                if item_ids:
                    fetch_result = index.fetch(ids=item_ids, namespace=namespace)
                    items = [
                        {
                            "id": vec.id,
                            "metadata": vec.metadata
                        }
                        for vec in fetch_result.vectors
                    ]
                else:
                    items = []
                
                logger.info(f"Listed {len(items)} food items from namespace '{namespace}'")
                return {
                    "total_items": len(items),
                    "namespace": namespace,
                    "items": items
                }
            else:
                return {"error": f"Unknown tool: {tool_name}"}
            
        except Exception as e:
            logger.error(f"Error calling MCP tool '{tool_name}': {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": f"Tool execution failed: {str(e)}"}
    
    def _format_tools_for_claude(self) -> List[Dict]:
        """Format MCP tools for Claude's tool use format"""
        tools = self._get_mcp_tools()
        
        claude_tools = []
        for tool in tools:
            claude_tool = {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["input_schema"]
            }
            claude_tools.append(claude_tool)
        
        return claude_tools
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for Claude"""
        return """You are NutriMood, a helpful AI assistant that recommends food items from Niloufer restaurant.

Your role:
- Understand user queries about food preferences, mood, dietary requirements
- Use the available MCP tools to search Pinecone vector database for relevant food items
- Provide friendly, conversational recommendations based on search results
- Include relevant details like calories, nutrition info, and prices when available
- Be warm, helpful, and enthusiastic about food

Guidelines:
- Always use the MCP tools to search for food items before making recommendations
- If user asks for food by description (e.g., "spicy vegetarian"), use search_food_by_description
- If user mentions a category (breakfast, lunch, etc.), use search_food_by_category
- If user mentions mood (happy, stressed, etc.), use search_by_mood
- Present results in a friendly, conversational manner
- Include 2-3 food recommendations when appropriate
- Mention key details like name, calories, and price for each recommendation"""
    
    def _normalize_conversation_history(self, history: List[Dict]) -> List[Dict]:
        """Normalize conversation history to ensure proper role alternation"""
        if not history:
            return []
        
        normalized = []
        last_role = None
        
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "")
            
            # Skip empty messages
            if not content:
                continue
            
            # If same role as last, merge content
            if role == last_role and role == "user":
                # Merge user messages
                if isinstance(normalized[-1]["content"], str):
                    normalized[-1]["content"] += f"\n\n{content}"
                else:
                    normalized[-1]["content"] = str(normalized[-1]["content"]) + f"\n\n{content}"
            elif role == last_role and role == "assistant":
                # Merge assistant messages
                if isinstance(normalized[-1]["content"], str):
                    normalized[-1]["content"] += f"\n\n{content}"
                else:
                    normalized[-1]["content"] = str(normalized[-1]["content"]) + f"\n\n{content}"
            else:
                # Add new message
                normalized.append({
                    "role": role,
                    "content": content
                })
                last_role = role
        
        return normalized
    
    def process_query(self, user_query: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Process a user query using Claude 3 Sonnet with MCP tools
        
        Args:
            user_query: The user's question or request
            conversation_history: Optional conversation history
            
        Returns:
            Claude's response with food recommendations
        """
        try:
            # Get tools formatted for Claude
            tools = self._format_tools_for_claude()
            
            # Build messages
            messages = []
            
            # Add conversation history if provided (normalized to ensure proper alternation)
            if conversation_history:
                normalized_history = self._normalize_conversation_history(conversation_history[-10:])
                messages.extend(normalized_history)
            
            # Ensure we don't have consecutive user messages
            # If last message is user, merge with current query
            if messages and messages[-1].get("role") == "user":
                if isinstance(messages[-1]["content"], str):
                    messages[-1]["content"] += f"\n\n{user_query}"
                else:
                    messages[-1]["content"] = str(messages[-1]["content"]) + f"\n\n{user_query}"
            else:
                # Add current user query
                messages.append({
                    "role": "user",
                    "content": user_query
                })
            
            # Prepare request body
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.model_config["max_tokens"],
                "temperature": self.model_config["temperature"],
                "top_p": self.model_config["top_p"],
                "system": self._build_system_prompt(),
                "tools": tools,
                "messages": messages
            }
            
            # Call Claude with tool use
            max_iterations = 3  # Reduced from 5 to limit API calls
            iteration = 0
            full_response = ""
            
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"🔄 Claude iteration {iteration}")
                
                # Use global rate limiter BEFORE making API call (ensures 50 RPM limit)
                self.rate_limiter.wait_if_needed()
                
                # Add delay between iterations (increased from 500ms to 1.5s)
                if iteration > 1:
                    time.sleep(1.5)  # 1.5 second delay between Claude iterations
                
                # Invoke model with retry logic for throttling
                max_retries = 3
                retry_delay = 5  # Base delay increased to 5 seconds
                
                for retry_attempt in range(max_retries):
                    try:
                        response = self.bedrock_client.invoke_model(
                            modelId=self.model_id,
                            contentType="application/json",
                            accept="application/json",
                            body=json.dumps(request_body)
                        )
                        break  # Success, exit retry loop
                    except ClientError as e:
                        error_code = e.response['Error']['Code']
                        if error_code == 'ThrottlingException' and retry_attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** retry_attempt)  # Exponential backoff: 5s, 10s, 20s
                            logger.warning(f"⏳ Rate limited. Retrying in {wait_time} seconds... (Attempt {retry_attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            # Use global rate limiter after waiting
                            self.rate_limiter.wait_if_needed()
                            continue
                        else:
                            raise  # Re-raise if not throttling or max retries reached
                
                # Parse response
                response_body = json.loads(response.get('body').read())
                
                # Calculate cost if usage info is available
                try:
                    usage = response_body.get('usage', {})
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    if input_tokens > 0 or output_tokens > 0:
                        cost_data = self.cost_calculator.calculate_cost(int(input_tokens), int(output_tokens))
                        logger.info(f"💰 Cost: ${cost_data['total_cost']:.6f} "
                                   f"(Input: {input_tokens}, Output: {output_tokens})")
                except Exception as e:
                    logger.debug(f"Could not calculate cost: {str(e)}")
                
                # Collect all content blocks
                text_blocks = []
                tool_use_blocks = []
                
                for content_block in response_body.get("content", []):
                    if content_block.get("type") == "text":
                        text_blocks.append(content_block.get("text", ""))
                    elif content_block.get("type") == "tool_use":
                        tool_use_blocks.append(content_block)
                
                # Add text response if any
                if text_blocks:
                    full_response += " ".join(text_blocks)
                
                # Handle tool use - collect all tools first, then execute and add results
                if tool_use_blocks:
                    # Add assistant message with all tool_use blocks
                    messages.append({
                        "role": "assistant",
                        "content": tool_use_blocks
                    })
                    
                    # Execute all tools and collect results
                    tool_results = []
                    for tool_use_block in tool_use_blocks:
                        tool_name = tool_use_block.get("name")
                        tool_input = tool_use_block.get("input", {})
                        tool_use_id = tool_use_block.get("id")
                        
                        logger.info(f"🔧 Claude requesting tool: {tool_name}")
                        logger.info(f"📥 Tool input: {json.dumps(tool_input, indent=2)}")
                        
                        # Call the MCP tool
                        tool_result = self._call_mcp_tool(tool_name, tool_input)
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": json.dumps(tool_result, indent=2)
                        })
                    
                    # Add user message with all tool results
                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })
                    
                    # Update request body with new messages
                    request_body["messages"] = messages
                    continue  # Loop to get Claude's response to tool results
                
                # Check if Claude is done (no more tool use)
                if response_body.get("stop_reason") != "tool_use":
                    break
            
            logger.info(f"✅ Query processed successfully")
            return full_response
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"❌ AWS Bedrock error: {error_code} - {error_message}")
            return f"Sorry, I encountered an error: {error_code} - {error_message}"
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {str(e)}")
            return f"Oops! Something went wrong: {str(e)}"
    
    def chat(self, user_query: str) -> str:
        """Simple chat interface"""
        return self.process_query(user_query)


# ============================================================================
# FastAPI Server for Web Frontend
# ============================================================================

# Initialize FastAPI app
app = FastAPI(title="NutriMood MCP Server", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize server instance
mcp_server = None

def get_server():
    """Get or create server instance"""
    global mcp_server
    if mcp_server is None:
        mcp_server = MCPClaudeServer()
    return mcp_server

# Request models
class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict]] = []

# Serve static files if they exist
static_dir = os.path.join(os.path.dirname(__file__), 'frontend')
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates
template_dir = os.path.join(os.path.dirname(__file__), 'frontend', 'templates')
if os.path.exists(template_dir):
    templates = Jinja2Templates(directory=template_dir)
else:
    templates = None

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the chat interface"""
    if templates:
        return templates.TemplateResponse("mcp_chat.html", {"request": request})
    else:
        # Fallback HTML if template doesn't exist
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NutriMood MCP Chat</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body>
            <h1>NutriMood MCP Chat</h1>
            <p>Please create frontend/templates/mcp_chat.html</p>
        </body>
        </html>
        """)

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint for frontend"""
    try:
        server = get_server()
        response = server.process_query(request.message, request.conversation_history)
        return JSONResponse({
            "response": response,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Streaming chat endpoint"""
    async def generate():
        try:
            server = get_server()
            # Get the full response
            response = server.process_query(request.message, request.conversation_history)
            
            # Stream response in chunks (simulate streaming by sending word by word)
            words = response.split(' ')
            chunk_size = 3  # Send 3 words at a time for smoother streaming
            
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                if i > 0:
                    chunk = ' ' + chunk  # Add space before chunk (except first)
                
                yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"
                # Small delay to simulate streaming
                await asyncio.sleep(0.05)
            
            # Send final done message
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "NutriMood MCP Server"}


def main():
    """Main function - can run CLI or server"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Run as web server
        port = int(os.getenv("APP_PORT", 8000))
        host = os.getenv("APP_HOST", "0.0.0.0")
        logger.info(f"🚀 Starting web server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
    else:
        # Interactive CLI mode
        print("=" * 60)
        print("🍽️  NutriMood MCP Claude Server")
        print("=" * 60)
        print("\nThis server uses Claude 3 Sonnet to search Pinecone for food items.")
        print("Type 'quit' or 'exit' to stop.\n")
        print("To run as web server: python server.py server\n")
        
        server = MCPClaudeServer()
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                print("\n🤖 NutriMood: ", end="", flush=True)
                response = server.process_query(user_input, conversation_history)
                print(response)
                
                # Update conversation history
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": response})
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()

