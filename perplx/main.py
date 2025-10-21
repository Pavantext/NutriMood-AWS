"""
Nutrimood Chatbot - Main Application
A food recommendation chatbot using AWS Bedrock and MCP
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
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
from utils.response_formatter import ResponseFormatter

app = FastAPI(title="Nutrimood Chatbot API", version="1.0.0")

# Initialize services
bedrock_service = BedrockService()
food_service = FoodService()
session_service = SessionService()
response_formatter = ResponseFormatter()
mcp_server = None  # Will be initialized after food data is loaded

# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_preferences: Optional[Dict] = None
    user_name: Optional[str] = None  # User's name for personalized responses

class ChatResponse(BaseModel):
    message: str
    session_id: str
    food_recommendation_id: str

class RecommendRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[Dict] = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
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

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Nutrimood Chatbot",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - processes user queries and returns recommendations
    """
    try:
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())
        session = session_service.get_or_create_session(session_id)
        
        # Update user name and preferences if provided
        if request.user_name:
            current_prefs = session.get("preferences", {})
            current_prefs["name"] = request.user_name
            session_service.update_preferences(session_id, current_prefs)
        
        if request.user_preferences:
            session_service.update_preferences(session_id, request.user_preferences)
        
        # Add user message to history
        session_service.add_message(session_id, "user", request.message)
        
        # Get conversation context
        conversation_history = session_service.get_conversation_history(session_id)
        
        # Get food recommendations based on query
        food_matches = food_service.find_matching_foods(
            request.message,
            conversation_history
        )
        
        # Build context for LLM
        food_context = food_service.build_food_context(food_matches)
        
        # Generate response from Bedrock (non-streaming)
        full_response = ""
        async for chunk in bedrock_service.generate_streaming_response(
            user_query=request.message,
            conversation_history=conversation_history,
            food_context=food_context,
            session_preferences=session.get("preferences", {})
        ):
            full_response += chunk
        
        # Extract recommended food IDs from the response
        recommended_ids = food_service.extract_food_ids_from_response(
            full_response,
            food_matches
        )
        
        # Save assistant response to session
        session_service.add_message(session_id, "assistant", full_response)
        session_service.add_recommendations(session_id, recommended_ids)
        
        # Return clean JSON response
        return {
            "message": full_response,
            "session_id": session_id,
            "food_recommendation_id": ",".join(recommended_ids) if recommended_ids else ""
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint - returns response as text stream for real-time display
    """
    try:
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())
        session = session_service.get_or_create_session(session_id)
        
        # Update user name and preferences if provided
        if request.user_name:
            current_prefs = session.get("preferences", {})
            current_prefs["name"] = request.user_name
            session_service.update_preferences(session_id, current_prefs)
        
        if request.user_preferences:
            session_service.update_preferences(session_id, request.user_preferences)
        
        # Add user message to history
        session_service.add_message(session_id, "user", request.message)
        
        # Get conversation context
        conversation_history = session_service.get_conversation_history(session_id)
        
        # Generate streaming response
        async def generate_stream():
            full_response = ""
            recommended_ids = []
            
            # Get food recommendations based on query
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
                yield chunk
            
            # Extract recommended food IDs from the response
            recommended_ids = food_service.extract_food_ids_from_response(
                full_response,
                food_matches
            )
            
            # Save assistant response to session
            session_service.add_message(session_id, "assistant", full_response)
            session_service.add_recommendations(session_id, recommended_ids)
            
            # Send final JSON metadata
            final_metadata = {
                "session_id": session_id,
                "food_recommendation_id": ",".join(recommended_ids) if recommended_ids else ""
            }
            
            yield f"\n\n__METADATA__:{json.dumps(final_metadata)}"
        
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

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
