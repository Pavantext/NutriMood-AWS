from fastapi import FastAPI, HTTPException, Request, Form, status
from fastapi.responses import StreamingResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from interfaces.base_models import ChatRequest, RecommendRequest, TrackSessionRequest, TrackFoodOrderRequest, ChatbotRatingRequest, MenuItemRequest, MenuItemIngestResponse
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
import uvicorn
import json
import asyncio
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv
import secrets

# Load environment variables
load_dotenv()

# Custom middleware to force streaming headers
class StreamingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Force streaming headers for chat endpoint
        if request.url.path == "/chat" and isinstance(response, StreamingResponse):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            response.headers["X-Accel-Buffering"] = "no"
            response.headers["X-Render-Buffering"] = "no"
            response.headers["X-Proxy-Buffering"] = "no"
            response.headers["Connection"] = "keep-alive"
            response.headers["Transfer-Encoding"] = "chunked"
            
        return response

# Import custom modules
from services.bedrock_service import BedrockService
from services.food_service import FoodService
from services.session_service import SessionService
# from services.mcp_server import MCPServer
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
    # global mcp_server
    # mcp_server = MCPServer(food_service)
    # print("✅ MCP Server initialized")
    
    yield
    
    # Shutdown (if needed)
    print("👋 Shutting down Nutrimood Chatbot...")

app = FastAPI(title="Nutrimood Chatbot API", version="1.0.0", lifespan=lifespan)

# Add session middleware for admin login
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", secrets.token_urlsafe(32)))

# Configure Jinja2 templates
template_dir = os.path.join(os.path.dirname(__file__), "frontend", "templates")
templates = Jinja2Templates(directory=template_dir)

# Mount static files (if they exist)
static_path = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(static_path):
    try:
        app.mount("/static", StaticFiles(directory=static_path), name="static")
    except Exception as e:
        print(f"⚠️  Could not mount static files: {e}")

# Add streaming middleware
app.add_middleware(StreamingMiddleware)

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


def _is_followup_question(query: str, conversation_history: List[Dict]) -> bool:
    """
    Simplified follow-up detection - let LLM handle most context understanding
    Only detect obvious cases that need special handling
    """
    if not conversation_history:
        return False
    
    query_lower = query.lower().strip()
    
    # Only detect obvious contextual references
    contextual_refs = ['these', 'those', 'them', 'it', 'that', 'this', 'which']
    has_contextual_ref = any(ref in query_lower for ref in contextual_refs)
    
    # Only detect obvious food property questions
    food_property_words = ['calorie', 'nutrient', 'price', 'cost', 'how much']
    has_food_property = any(word in query_lower for word in food_property_words)
    
    # Simple detection: contextual reference OR food property question
    return has_contextual_ref or has_food_property

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
                    
                    # Filter out invalid IDs
                    valid_previous_ids = [fid for fid in previous_food_ids if fid and str(fid).strip()]
                    
                    # Fetch these specific foods
                    food_matches = []
                    for food_id in valid_previous_ids[:5]:
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
            
            # Stream response from Bedrock with explicit flushing
            async for chunk in bedrock_service.generate_streaming_response(
                user_query=request.message,
                conversation_history=conversation_history,
                food_context=food_context,
                session_preferences=session.get("preferences", {})
            ):
                full_response += chunk
                # Yield each character individually to force streaming
                for char in chunk:
                    yield char
                    await asyncio.sleep(0.01)  # Small delay to force transmission
            
            # Extract recommended food IDs from the response
            recommended_ids = food_service.extract_food_ids_from_response(
                full_response,
                food_matches
            )

            # If no food IDs extracted, don't show any food items to frontend
            if not recommended_ids:
                print("⚠️ No valid food matches found in LLM response. Not showing any food items.")
            
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
            # Filter out any None/empty values before joining
            valid_ids = [str(fid) for fid in recommended_ids if fid and str(fid).strip()]
            final_response = {
                "message": full_response,
                "session_id": session_id,
                "food_recommendation_id": ",".join(valid_ids) if valid_ids else ""
            }
            
            yield f"\n\n__RESPONSE__:{json.dumps(final_response)}"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
                "X-Accel-Buffering": "no",  # Critical for nginx/proxies
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "Pragma": "no-cache",
                "Expires": "0",
                # Additional headers to force streaming on Render
                "X-Render-Buffering": "no",
                "X-Proxy-Buffering": "no",
            }
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

@app.post("/menu/ingest", response_model=MenuItemIngestResponse)
async def ingest_menu_item(request: MenuItemRequest):
    """
    Ingest a new menu item (Bill of Materials) and store it in Pinecone BOM index
    
    This endpoint allows users to upload their menu items which will be:
    1. Converted to embeddings using AWS Titan
    2. Stored in a separate Pinecone BOM index (for user-uploaded items)
    3. Made available for semantic search
    
    Note: User-uploaded BOM items are stored in a separate Pinecone index to keep them
    isolated from the original menu items.
    
    **Request Body:**
    ```json
    {
        "Id": "e754af8d-bb53-421a-ace5-c28ab216b4d2",
        "GST": 5.0,
        "IsPopular": false,
        "ProductName": "Jalapeno Cheese Poppers (6.Pcs)",
        "Description": "Our jalapeños are carefully selected for their perfect balance of heat and flavor, .....",
        "Image": "https://niloufer.blob.core.windows.net/menu-images/jalapino%20poppers%2001-min-min.jpg",
        "Price": 380.0,
        "Calories": 280,
        "Macronutrients": {
            "protein": "10g",
            "carbohydrates": "25g",
            "fat": "20g",
            "fiber": "2g"
        },
        "Ingredients": ["JALAPINO CHEESE POPPERS SEMI FINISHED 1 NO", "GARLIC MAYO SEMI FINISHED 60 GRAM"],
        "Dietary": ["High-protein"],
        "HealthBenefits": "The JALAPINO CHEESE POPPERS contain capsaicin from jalapeños, offering mild antioxidant...",
        "CuisineType": "Fusion",
        "MealType": "Snack",
        "Occasion": "Indulgence",
        "SpiceLevel": "Hot"
    }
    ```
    
    **Response:**
    ```json
    {
      "status": "success",
      "message": "Menu item successfully ingested and stored in Pinecone",
      "item_id": "menu-item-001"
    }
    ```
    """
    try:
        # Check if Pinecone and embedding services are available
        if not food_service.use_vector_search:
            raise HTTPException(
                status_code=503,
                detail="Vector search not available. Please configure Pinecone and AWS Titan embeddings."
            )
        
        # Build embedding text for the menu item
        embedding_parts = [
            f"Food: {request.ProductName}",
            f"Description: {request.Description}",
            f"Calories: {request.Calories}",
            f"Protein: {request.Macronutrients.get('protein', '0g') if request.Macronutrients else '0g'}",
            f"Macronutrients: {request.Macronutrients}",
            f"Ingredients: {', '.join(request.Ingredients)}" if request.Ingredients else "",
            f"Dietary: {', '.join(request.Dietary)}" if request.Dietary else "",
            f"Health Benefits: {request.HealthBenefits}" if request.HealthBenefits else "",
            f"Cuisine Type: {request.CuisineType}" if request.CuisineType else "",
            f"Meal Type: {request.MealType}" if request.MealType else "",
            f"Occasion: {request.Occasion}" if request.Occasion else "",
            f"Spice Level: {request.SpiceLevel}" if request.SpiceLevel else ""
        ]
        embedding_text = '. '.join(filter(None, embedding_parts))
        
        # Generate embedding using AWS Titan
        embedding = food_service.embedding_service.generate_embedding(embedding_text)
        
        if not embedding:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate embedding for menu item"
            )
        
        # Prepare food data dictionary
        food_data = {
            'id': request.Id,
            'product_name': request.ProductName,
            'description': request.Description,
            'calories': request.Calories,
            'price': request.Price,
            'image_url': request.Image,
            'gst': request.GST,
            'is_popular': request.IsPopular,
            'ingredients': request.Ingredients,
            'dietary': request.Dietary,
            'macronutrients': request.Macronutrients,
            'health_benefits': request.HealthBenefits,
            'cuisine_type': request.CuisineType,
            'meal_type': request.MealType,
            'occasion': request.Occasion,
            'spice_level': request.SpiceLevel
        }
        
        # Upsert to Pinecone BOM index (separate index for user-uploaded items)
        success = food_service.pinecone_service.upsert_food_item(
            food_id=request.Id,
            embedding=embedding,
            food_data=food_data,
            use_bom_index=True  # Use BOM index for user-uploaded items
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to store menu item in Pinecone BOM index"
            )
        
        return MenuItemIngestResponse(
            status="success",
            message="Menu item successfully ingested and stored in Pinecone BOM index",
            item_id=request.Id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting menu item: {str(e)}")

@app.post("/menu/ingest/batch", response_model=MenuItemIngestResponse)
async def ingest_menu_items_batch(requests: List[MenuItemRequest]):
    """
    Ingest multiple menu items (Bill of Materials) in batch and store them in Pinecone BOM index
    
    This endpoint allows users to upload multiple menu items at once.
    All items will be converted to embeddings and stored in the separate Pinecone BOM index.
    
    **Request Body:**
    ```json
    [
      {
        "Id": "menu-item-001",
        "ProductName": "Chicken Biryani",
        ...
      },
      {
        "Id": "menu-item-002",
        "ProductName": "Vegetable Curry",
        ...
      }
    ]
    ```
    
    **Response:**
    ```json
    {
      "status": "success",
      "message": "Successfully ingested 2 menu items",
      "total_items": 2
    }
    ```
    """
    try:
        # Check if Pinecone and embedding services are available
        if not food_service.use_vector_search:
            raise HTTPException(
                status_code=503,
                detail="Vector search not available. Please configure Pinecone and AWS Titan embeddings."
            )
        
        success_count = 0
        failed_items = []
        
        for request in requests:
            try:
                # Build embedding text for the menu item
                embedding_parts = [
                    f"Food: {request.ProductName}",
                    f"Description: {request.Description}",
                    f"Calories: {request.Calories}",
                    f"Protein: {request.Macronutrients.get('protein', '0g') if request.Macronutrients else '0g'}",
                    f"Macronutrients: {request.Macronutrients}",
                    f"Ingredients: {', '.join(request.Ingredients)}" if request.Ingredients else "",
                    f"Dietary: {', '.join(request.Dietary)}" if request.Dietary else "",
                    f"Health Benefits: {request.HealthBenefits}" if request.HealthBenefits else "",
                    f"Cuisine Type: {request.CuisineType}" if request.CuisineType else "",
                    f"Meal Type: {request.MealType}" if request.MealType else "",
                    f"Occasion: {request.Occasion}" if request.Occasion else "",
                    f"Spice Level: {request.SpiceLevel}" if request.SpiceLevel else ""
                ]
                embedding_text = '. '.join(filter(None, embedding_parts))
                
                # Generate embedding using AWS Titan
                embedding = food_service.embedding_service.generate_embedding(embedding_text)
                
                if not embedding:
                    failed_items.append(request.Id)
                    continue
                
                # Prepare food data dictionary
                food_data = {
                    'id': request.Id,
                    'product_name': request.ProductName,
                    'description': request.Description,
                    'calories': request.Calories,
                    'price': request.Price,
                    'image_url': request.Image,
                    'gst': request.GST,
                    'is_popular': request.IsPopular,
                    'ingredients': request.Ingredients,
                    'dietary': request.Dietary,
                    'macronutrients': request.Macronutrients,
                    'health_benefits': request.HealthBenefits,
                    'cuisine_type': request.CuisineType,
                    'meal_type': request.MealType,
                    'occasion': request.Occasion,
                    'spice_level': request.SpiceLevel
                }
                
                # Upsert to Pinecone BOM index (separate index for user-uploaded items)
                success = food_service.pinecone_service.upsert_food_item(
                    food_id=request.Id,
                    embedding=embedding,
                    food_data=food_data,
                    use_bom_index=True  # Use BOM index for user-uploaded items
                )
                
                if success:
                    success_count += 1
                else:
                    failed_items.append(request.Id)
                    
            except Exception as e:
                print(f"❌ Error processing menu item {request.Id}: {e}")
                failed_items.append(request.Id)
        
        if success_count == 0:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to ingest all menu items. Failed items: {', '.join(failed_items)}"
            )
        
        message = f"Successfully ingested {success_count} menu item(s) to BOM index"
        if failed_items:
            message += f". Failed items: {', '.join(failed_items)}"
        
        return MenuItemIngestResponse(
            status="success" if len(failed_items) == 0 else "partial",
            message=message,
            total_items=success_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting menu items: {str(e)}")

@app.get("/menu/bom/stats")
async def get_bom_index_stats():
    """
    Get statistics for the BOM (Bill of Materials) Pinecone index
    
    Returns information about the separate index used for user-uploaded menu items.
    
    **Response:**
    ```json
    {
      "index_name": "niloufer-bom",
      "total_vectors": 150,
      "dimension": 1024,
      "namespaces": {}
    }
    ```
    """
    try:
        stats = food_service.pinecone_service.get_bom_index_stats()
        
        if "error" in stats:
            raise HTTPException(
                status_code=503,
                detail=f"BOM index not available: {stats['error']}"
            )
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting BOM index stats: {str(e)}")

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


@app.post("/chat/track-session")
async def track_session(request: TrackSessionRequest):
    """
    Track chatbot session time
    Called when chat window opens (start_time) and closes (end_time)
    """
    try:
        # Parse timestamps - handle ISO 8601 format
        start_time_str = request.start_time.replace('Z', '+00:00') if request.start_time.endswith('Z') else request.start_time
        start_time = datetime.fromisoformat(start_time_str)
        
        end_time = None
        if request.end_time:
            end_time_str = request.end_time.replace('Z', '+00:00') if request.end_time.endswith('Z') else request.end_time
            end_time = datetime.fromisoformat(end_time_str)
        
        # Convert to UTC if timezone-aware, otherwise assume UTC
        if start_time.tzinfo is not None:
            start_time = start_time.astimezone().replace(tzinfo=None)
        
        if end_time and end_time.tzinfo is not None:
            end_time = end_time.astimezone().replace(tzinfo=None)
        
        success = database_service.track_chatbot_session(
            session_id=request.session_id,
            start_time=start_time,
            end_time=end_time,
            total_time_seconds=request.total_time_seconds,
            user_name=request.user_name
        )
        
        if success:
            return {"status": "success", "message": "Session tracked successfully"}
        else:
            return {"status": "error", "message": "Failed to track session"}
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tracking session: {str(e)}")

@app.post("/chat/track-food-order")
async def track_food_order(request: TrackFoodOrderRequest):
    """
    Track food orders from chatbot
    Called when items are added to cart or orders are placed
    """
    try:
        # Validate event_type
        if request.event_type not in ['added_to_cart', 'order_placed']:
            raise HTTPException(
                status_code=400, 
                detail="event_type must be either 'added_to_cart' or 'order_placed'"
            )
        
        # Parse timestamp - handle ISO 8601 format
        timestamp_str = request.timestamp.replace('Z', '+00:00') if request.timestamp.endswith('Z') else request.timestamp
        timestamp = datetime.fromisoformat(timestamp_str)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone().replace(tzinfo=None)
        
        success = database_service.track_food_order(
            session_id=request.session_id,
            product_id=request.product_id,
            product_name=request.product_name,
            timestamp=timestamp,
            event_type=request.event_type,
            user_name=request.user_name,
            order_id=request.order_id,
            quantity=request.quantity
        )
        
        if success:
            return {"status": "success", "message": "Food order tracked successfully"}
        else:
            return {"status": "error", "message": "Failed to track food order"}
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tracking food order: {str(e)}")

@app.post("/chat/rating")
async def submit_chatbot_rating(request: ChatbotRatingRequest):
    """
    Submit chatbot rating/feedback
    Called when user clicks on a star rating (1-5 stars)
    Tracks which specific bot message was rated using message_id
    """
    try:
        # Validate rating
        if request.rating < 1 or request.rating > 5:
            raise HTTPException(
                status_code=400,
                detail="Rating must be between 1 and 5"
            )
        
        # Parse timestamp - handle ISO 8601 format
        timestamp_str = request.timestamp.replace('Z', '+00:00') if request.timestamp.endswith('Z') else request.timestamp
        timestamp = datetime.fromisoformat(timestamp_str)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone().replace(tzinfo=None)
        
        success = database_service.track_chatbot_rating(
            session_id=request.session_id,
            rating=request.rating,
            timestamp=timestamp,
            message_id=request.message_id,
            user_name=request.user_name
        )
        
        if success:
            return {"status": "success", "message": "Rating submitted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to submit rating")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting rating: {str(e)}")

@app.get("/analytics/chatbot/stats")
async def get_chatbot_analytics():
    """
    Get comprehensive chatbot analytics for admin dashboard
    """
    if not database_service.enabled:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    analytics = database_service.get_chatbot_analytics()
    return analytics

@app.get("/analytics/orders")
async def get_orders_analytics():
    """Get orders analytics (added to cart vs placed)"""
    if not database_service.enabled:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    analytics = database_service.get_orders_analytics()
    return analytics

@app.get("/analytics/users")
async def get_users_analytics(start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Get users analytics with optional date range filter"""
    if not database_service.enabled:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    start_dt = None
    end_dt = None
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if start_dt.tzinfo:
                start_dt = start_dt.astimezone().replace(tzinfo=None)
        except:
            pass
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            if end_dt.tzinfo:
                end_dt = end_dt.astimezone().replace(tzinfo=None)
        except:
            pass
    
    analytics = database_service.get_users_analytics(start_dt, end_dt)
    return analytics

@app.get("/analytics/feedback")
async def get_feedback_analytics(
    limit: int = 50,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    rating: Optional[str] = None
):
    """Get feedback with user queries and responses, with optional date and rating filters"""
    if not database_service.enabled:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    print(f"🔍 Received feedback request - start_date: {start_date}, end_date: {end_date}, rating: {rating}")
    
    start_dt = None
    end_dt = None
    if start_date:
        try:
            # Handle YYYY-MM-DD format from HTML date inputs
            if len(start_date) == 10:  # YYYY-MM-DD format
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            else:
                # Handle ISO format with time
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if start_dt.tzinfo:
                    start_dt = start_dt.astimezone().replace(tzinfo=None)
        except Exception as e:
            print(f"Error parsing start_date: {e}")
            pass
    if end_date:
        try:
            # Handle YYYY-MM-DD format from HTML date inputs
            if len(end_date) == 10:  # YYYY-MM-DD format
                # Add time to end of day for inclusive end date filtering
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                end_dt = end_dt.replace(hour=23, minute=59, second=59)
            else:
                # Handle ISO format with time
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                if end_dt.tzinfo:
                    end_dt = end_dt.astimezone().replace(tzinfo=None)
        except Exception as e:
            print(f"Error parsing end_date: {e}")
            pass
    
    # Validate and convert rating
    rating_int = None
    if rating and rating.strip():
        try:
            rating_int = int(rating.strip())
            if rating_int < 1 or rating_int > 5:
                rating_int = None
        except (ValueError, AttributeError):
            rating_int = None
    
    print(f"🔍 Parsed filters - start_dt: {start_dt}, end_dt: {end_dt}, rating_int: {rating_int}")
    
    feedback = database_service.get_feedback_with_conversations(limit, start_dt, end_dt, rating_int)
    return feedback

@app.get("/analytics/session-times")
async def get_session_times_analytics():
    """Get session time analytics per user"""
    if not database_service.enabled:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    analytics = database_service.get_session_times_analytics()
    return analytics

@app.get("/analytics/sessions")
async def get_sessions_analytics(start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Get user sessions with optional date range filter"""
    if not database_service.enabled:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    start_dt = None
    end_dt = None
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if start_dt.tzinfo:
                start_dt = start_dt.astimezone().replace(tzinfo=None)
        except:
            pass
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            if end_dt.tzinfo:
                end_dt = end_dt.astimezone().replace(tzinfo=None)
        except:
            pass
    
    sessions = database_service.get_all_users_filtered(start_dt, end_dt)
    return {"sessions": sessions, "count": len(sessions)}

# Admin Endpoints
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

def check_admin_session(request: Request) -> bool:
    """Check if user is logged in as admin"""
    return request.session.get("admin_logged_in", False)

@app.get("/admin", response_class=HTMLResponse)
async def admin_root(request: Request):
    """Redirect to login or dashboard"""
    if check_admin_session(request):
        return RedirectResponse(url="/admin/dashboard", status_code=status.HTTP_302_FOUND)
    return RedirectResponse(url="/admin/login", status_code=status.HTTP_302_FOUND)

@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    """Show admin login page"""
    if check_admin_session(request):
        return RedirectResponse(url="/admin/dashboard", status_code=status.HTTP_302_FOUND)
    
    return templates.TemplateResponse("admin_login.html", {
        "request": request,
        "error": None
    })

@app.post("/admin/login", response_class=HTMLResponse)
async def admin_login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle admin login"""
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        request.session["admin_logged_in"] = True
        request.session["admin_username"] = username
        return RedirectResponse(url="/admin/dashboard", status_code=status.HTTP_302_FOUND)
    else:
        return templates.TemplateResponse("admin_login.html", {
            "request": request,
            "error": "Invalid username or password"
        })

@app.get("/admin/logout")
async def admin_logout(request: Request):
    """Handle admin logout"""
    request.session.clear()
    return RedirectResponse(url="/admin/login", status_code=status.HTTP_302_FOUND)

@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Show admin dashboard with all users"""
    if not check_admin_session(request):
        return RedirectResponse(url="/admin/login", status_code=status.HTTP_302_FOUND)
    
    if not database_service.enabled:
        return templates.TemplateResponse("admin_dashboard.html", {
            "request": request,
            "users": {},
            "chatbot_analytics": {},
            "error": "Database not configured"
        })
    
    users = database_service.get_all_users()
    chatbot_analytics = database_service.get_chatbot_analytics()
    
    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "users": users,
        "chatbot_analytics": chatbot_analytics
    })

@app.get("/admin/user/{session_id}", response_class=HTMLResponse)
async def admin_user_details(request: Request, session_id: str):
    """Show detailed session information"""
    if not check_admin_session(request):
        return RedirectResponse(url="/admin/login", status_code=status.HTTP_302_FOUND)
    
    if not database_service.enabled:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    user_data = database_service.get_user_details(session_id)
    
    if not user_data:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    
    # Populate food details for recommendations
    for conversation in user_data['conversations']:
        food_details = []
        for food_id in conversation.get('recommended_food_ids', []):
            # Handle both string IDs and dict objects
            if isinstance(food_id, dict):
                # Already a food object, use it directly
                food_item = food_id
            else:
                # It's an ID string, fetch the food item
                food_item = food_service.get_food_by_id(food_id)
            
            if food_item:
                food_details.append({
                    'id': food_item.get('Id') or food_item.get('id'),
                    'name': food_item.get('Name') or food_item.get('ProductName', 'Unknown'),
                    'description': food_item.get('Description', ''),
                    'image_url': food_item.get('Image') or food_item.get('ImageUrl') or food_item.get('image_url') or '/static/default-food.jpg'
                })
        conversation['recommended_foods'] = food_details
    
    return templates.TemplateResponse("admin_user_details.html", {
        "request": request,
        "username": user_data.get('display_name', session_id),
        "session_id": session_id,
        "user_data": user_data
    })

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
