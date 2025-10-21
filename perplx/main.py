"""
Nutrimood Chatbot - Main Application
A food recommendation chatbot using AWS Bedrock and MCP
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
import uvicorn
import json
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import custom modules
from services.bedrock_service import BedrockService
from services.food_service import FoodService
from services.session_service import SessionService
from services.mcp_server import MCPServer
from services.database_service import DatabaseService
from utils.response_formatter import ResponseFormatter

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("🚀 Starting Nutrimood Chatbot...")
    
    # Get food data path from environment or use default
    food_data_path = os.getenv("FOOD_DATA_PATH", "../data/raw/Niloufer_data.json")
    
    # Try multiple possible paths
    possible_paths = [
        food_data_path,
        "data/Niloufer_data.json",
        "../data/raw/Niloufer_data.json",
        "data/food_items.json"
    ]
    
    loaded = False
    for path in possible_paths:
        try:
            food_service.load_food_data(path)
            if food_service.food_items:
                print(f"✅ Loaded {len(food_service.food_items)} food items from {path}")
                loaded = True
                break
        except FileNotFoundError:
            continue
    
    if not loaded:
        print("⚠️  Warning: Could not load food data. Please check the data path.")
    
    # Initialize MCP server after food data is loaded
    global mcp_server
    mcp_server = MCPServer(food_service)
    print("✅ MCP Server initialized")
    
    yield
    
    # Shutdown (if needed)
    print("👋 Shutting down Nutrimood Chatbot...")

app = FastAPI(title="Nutrimood Chatbot API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins like ["http://localhost:3000", "https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"],
)

# Initialize services
bedrock_service = BedrockService()
food_service = FoodService()
session_service = SessionService()
database_service = DatabaseService()  # AWS RDS PostgreSQL
response_formatter = ResponseFormatter()
mcp_server = None  # Will be initialized after food data is loaded

# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_preferences: Optional[Dict] = None
    user_name: Optional[str] = None  # User's name for personalized responses
    user_id: Optional[str] = None  # User ID for database tracking

class ChatResponse(BaseModel):
    message: str
    session_id: str
    food_recommendation_id: str

class RecommendRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[Dict] = None

def _is_followup_question(query: str, conversation_history: List[Dict]) -> bool:
    """Check if query is a follow-up about previous recommendations"""
    if not conversation_history:
        return False
    
    query_lower = query.lower().strip()
    
    # Follow-up keywords
    followup_keywords = [
        'calorie', 'nutrient', 'health', 'protein', 'benefit', 'ingredient',
        'price', 'cost', 'how much', 'what about', 'tell me',
        'more about', 'which one', 'compare', 'it', 'these', 'those', 'them',
        'that', 'this'
    ]
    
    # Check if query has follow-up keywords AND is short
    has_followup_word = any(keyword in query_lower for keyword in followup_keywords)
    is_short = len(query_lower.split()) <= 8
    
    return has_followup_word and is_short

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Nutrimood Chatbot",
        "version": "3.25.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint - streams response with final JSON structure
    """
    try:
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())
        session = session_service.get_or_create_session(session_id)
        
        # Update user name, ID, and preferences if provided
        if request.user_name or request.user_id or request.user_preferences:
            current_prefs = session.get("preferences", {})
            
            if request.user_name:
                current_prefs["name"] = request.user_name
            
            if request.user_id:
                current_prefs["user_id"] = request.user_id
            
            if request.user_preferences:
                current_prefs.update(request.user_preferences)
            
            session_service.update_preferences(session_id, current_prefs)
        
        # Add user message to history
        session_service.add_message(session_id, "user", request.message)
        
        # Get conversation context
        conversation_history = session_service.get_conversation_history(session_id)
        
        # Generate streaming response
        async def generate_stream():
            full_response = ""
            recommended_ids = []
            
            # Check if this is a follow-up question about previous recommendations
            is_followup = _is_followup_question(request.message, conversation_history)
            
            if is_followup:
                # For follow-ups, get the last recommended food IDs from session
                last_recommendations = session.get("recommendations", [])
                if last_recommendations:
                    # Get the most recent recommendation
                    last_rec = last_recommendations[-1]
                    previous_food_ids = last_rec.get("food_ids", [])
                    
                    # Fetch these specific foods
                    food_matches = []
                    for food_id in previous_food_ids[:5]:
                        food_item = food_service.get_food_by_id(food_id)
                        if food_item:
                            food_matches.append((food_item, 1.0))
                else:
                    # No previous recommendations, do normal search
                    food_matches = food_service.find_matching_foods(
                        request.message,
                        conversation_history
                    )
            else:
                # Normal query - search for new items
                food_matches = food_service.find_matching_foods(
                    request.message,
                    conversation_history
                )
            
            # Build context for LLM
            food_context = food_service.build_food_context(food_matches)
            
            # Stream response from Bedrock
            async for chunk in bedrock_service.generate_streaming_response(
                user_query=request.message,
                conversation_history=conversation_history,
                food_context=food_context,
                session_preferences=session.get("preferences", {})
            ):
                full_response += chunk
                yield chunk  # Stream the text as it comes
            
            # Extract recommended food IDs from the response
            recommended_ids = food_service.extract_food_ids_from_response(
                full_response,
                food_matches
            )
            
            # Save assistant response to session
            session_service.add_message(session_id, "assistant", full_response)
            session_service.add_recommendations(session_id, recommended_ids)
            
            # Save to database if enabled
            if database_service.enabled:
                user_id = session.get("preferences", {}).get("user_id")
                database_service.save_conversation(
                    session_id=session_id,
                    user_id=user_id,
                    user_message=request.message,
                    bot_response=full_response,
                    recommendations=recommended_ids,
                    query_intent=None,  # Can add intent detection later
                    response_time_ms=None  # Can add timing later
                )
                
                # Update session analytics
                messages = session.get("messages", [])
                recommendations = session.get("recommendations", [])
                database_service.update_session_analytics(
                    session_id=session_id,
                    user_id=user_id,
                    total_messages=len(messages),
                    total_recommendations=len(recommendations),
                    first_message_at=datetime.fromisoformat(session.get("created_at")) if session.get("created_at") else None,
                    last_message_at=datetime.now()
                )
            
            # Send final JSON response
            final_response = {
                "message": full_response,
                "session_id": session_id,
                "food_recommendation_id": ",".join(recommended_ids) if recommended_ids else ""
            }
            
            yield f"\n\n__RESPONSE__:{json.dumps(final_response)}"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    Retrieve session information and chat history
    """
    try:
        session = session_service.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "created_at": session.get("created_at"),
            "last_activity": session.get("last_activity"),
            "message_count": len(session.get("messages", [])),
            "messages": session.get("messages", []),
            "recommendations": session.get("recommendations", []),
            "preferences": session.get("preferences", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session: {str(e)}")

@app.post("/recommend")
async def recommend(request: RecommendRequest):
    """
    Internal endpoint for generating food recommendations
    Returns ranked list of food items based on query
    """
    try:
        # Find matching foods
        matches = food_service.find_matching_foods(
            query=request.query,
            conversation_history=[],
            top_k=request.top_k,
            filters=request.filters
        )
        
        # Format recommendations
        recommendations = []
        for food, score in matches:
            recommendations.append({
                "id": food.get("id"),
                "name": food.get("name"),
                "category": food.get("category"),
                "description": food.get("description"),
                "calories": food.get("calories"),
                "relevance_score": round(score, 3)
            })
        
        return {
            "query": request.query,
            "count": len(recommendations),
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and its history
    """
    try:
        success = session_service.delete_session(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": "Session deleted successfully", "session_id": session_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

@app.get("/foods")
async def list_foods(
    category: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    List available food items with optional filtering
    """
    try:
        foods = food_service.get_all_foods(
            category=category,
            limit=limit,
            offset=offset
        )
        
        return {
            "count": len(foods),
            "limit": limit,
            "offset": offset,
            "foods": foods
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing foods: {str(e)}")

# MCP Endpoints
@app.get("/mcp/info")
async def mcp_info():
    """
    Get MCP server information
    """
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP server not initialized")
    
    return mcp_server.get_server_info()

@app.get("/mcp/tools")
async def mcp_list_tools():
    """
    List available MCP tools
    """
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP server not initialized")
    
    return {
        "tools": mcp_server.list_tools()
    }

@app.post("/mcp/tools/{tool_name}")
async def mcp_call_tool(tool_name: str, arguments: Dict):
    """
    Execute an MCP tool
    """
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP server not initialized")
    
    try:
        result = mcp_server.call_tool(tool_name, arguments)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing tool: {str(e)}")

@app.get("/mcp/resources")
async def mcp_list_resources():
    """
    List available MCP resources
    """
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP server not initialized")
    
    return {
        "resources": mcp_server.list_resources()
    }

@app.get("/mcp/resources/{uri:path}")
async def mcp_read_resource(uri: str):
    """
    Read an MCP resource
    """
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP server not initialized")
    
    try:
        result = mcp_server.read_resource(f"nutrimood://{uri}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading resource: {str(e)}")

@app.get("/mcp/prompts")
async def mcp_list_prompts():
    """
    List available MCP prompts
    """
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP server not initialized")
    
    return {
        "prompts": mcp_server.list_prompts()
    }

# Database Analytics Endpoints
@app.get("/analytics/session/{session_id}")
async def get_session_analytics_endpoint(session_id: str):
    """
    Get analytics for a specific session
    """
    if not database_service.enabled:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    analytics = database_service.get_session_analytics(session_id)
    
    if not analytics:
        raise HTTPException(status_code=404, detail="Session analytics not found")
    
    return analytics

@app.get("/analytics/conversations/{session_id}")
async def get_session_conversations_db(session_id: str, limit: int = 50):
    """
    Get conversation history from database for a session
    """
    if not database_service.enabled:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    conversations = database_service.get_conversation_history(session_id, limit)
    
    return {
        "session_id": session_id,
        "count": len(conversations),
        "conversations": conversations
    }

@app.get("/analytics/user/{user_id}/conversations")
async def get_user_conversations_endpoint(user_id: str, limit: int = 100):
    """
    Get all conversations for a specific user
    """
    if not database_service.enabled:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    conversations = database_service.get_user_conversations(user_id, limit)
    
    return {
        "user_id": user_id,
        "count": len(conversations),
        "conversations": conversations
    }

@app.get("/analytics/feedback/stats")
async def get_feedback_statistics(user_id: Optional[str] = None):
    """
    Get feedback statistics (overall or for specific user)
    """
    if not database_service.enabled:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    stats = database_service.get_feedback_stats(user_id)
    
    return stats

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
