from pydantic import BaseModel
from typing import Optional, Dict, List

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

# Chatbot Tracking Endpoints
class TrackSessionRequest(BaseModel):
    session_id: str
    start_time: str  # ISO 8601 timestamp
    end_time: Optional[str] = None  # ISO 8601 timestamp (optional)
    total_time_seconds: Optional[int] = None
    user_name: Optional[str] = None

class TrackFoodOrderRequest(BaseModel):
    session_id: str
    product_id: str
    product_name: str
    timestamp: str  # ISO 8601 timestamp
    event_type: str  # 'added_to_cart' or 'order_placed'
    user_name: Optional[str] = None
    order_id: Optional[str] = None
    quantity: Optional[int] = None

class ChatbotRatingRequest(BaseModel):
    session_id: str
    message_id: str  # ID of the specific bot message being rated
    rating: int  # 1-5
    timestamp: str  # ISO 8601 timestamp
    user_name: Optional[str] = None

# Menu Item Ingestion (Bill of Materials)
class MenuItemRequest(BaseModel):
    """Request model for ingesting a menu item (Bill of Materials)"""
    Id: str  # Unique identifier for the menu item  
    GST: Optional[float] = 5.0 
    IsPopular: Optional[bool] = False  # Whether item is popular
    ProductName: str  # Name of the product
    Description: Optional[str] = ""  # Product description
    Image: Optional[str] = ""  # Image URL
    Price: Optional[float] = 0.0  # Price
    Calories: Optional[int] = 0  # Calories
    Macronutrients: Optional[Dict[str, str]] = {}  # Macronutrients dict with keys like "protein", "carbohydrates", "fat", "fiber"
    Ingredients: Optional[List[str]] = []  # List of ingredients
    Dietary: Optional[List[str]] = []  # Dietary information (e.g., ["vegetarian", "vegan"])
    HealthBenefits: Optional[str] = ""  # Health benefits
    CuisineType: Optional[str] = ""  # Cuisine type
    MealType: Optional[str] = ""  # Meal type
    Occasion: Optional[str] = ""  # Occasion
    SpiceLevel: Optional[str] = ""  # Spice level
    
class MenuItemIngestResponse(BaseModel):
    """Response model for menu item ingestion"""
    status: str  # "success" or "error"
    message: str
    item_id: Optional[str] = None
    total_items: Optional[int] = None  # For batch ingestion

