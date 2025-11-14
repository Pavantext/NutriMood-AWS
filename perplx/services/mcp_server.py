import os
import json
import logging
from typing import Any, Optional
from dotenv import load_dotenv
from fastmcp import FastMCP
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "food-recommendations")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

# Validate configuration
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize MCP server
mcp = FastMCP(name="NutriMood Vector DB Server")

logger.info(f"✅ NutriMood MCP Server initialized")
logger.info(f"📍 Index: {PINECONE_INDEX_NAME}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_index():
    """Get Pinecone index instance with connection pooling."""
    try:
        if PINECONE_INDEX_HOST:
            # Connect to existing index
            index = pc.Index(
                name=PINECONE_INDEX_NAME,
                host=PINECONE_INDEX_HOST,
                pool_threads=50,
                connection_pool_maxsize=50
            )
        else:
            # Describe index to get host
            index_desc = pc.describe_index(PINECONE_INDEX_NAME)
            index = pc.Index(
                name=PINECONE_INDEX_NAME,
                host=index_desc.host,
                pool_threads=50,
                connection_pool_maxsize=50
            )
        return index
    except Exception as e:
        logger.error(f"Failed to connect to index: {str(e)}")
        raise


def format_search_results(results: dict) -> dict:
    """Format search results for LLM consumption."""
    formatted_matches = []
    
    for match in results.get("matches", []):
        formatted_match = {
            "id": match.get("id"),
            "score": round(match.get("score", 0), 4),
            "metadata": match.get("metadata", {})
        }
        formatted_matches.append(formatted_match)
    
    return {
        "matches": formatted_matches,
        "namespace": results.get("namespace", "default"),
        "usage": results.get("usage", {})
    }


# ============================================================================
# RESOURCES - Read-only data access for LLM context
# ============================================================================

@mcp.resource("index://stats")
def get_index_stats() -> dict:
    """
    Get comprehensive statistics about the Pinecone index.
    Provides: total vectors, dimensions, namespaces, memory usage
    """
    try:
        index = get_index()
        stats = index.describe_index_stats()
        
        return {
            "total_vectors": stats.total_vector_count,
            "index_name": PINECONE_INDEX_NAME,
            "index_fullness": stats.index_fullness,
            "namespaces": list(stats.namespaces.keys()) if stats.namespaces else ["default"],
            "dimension": getattr(stats, "dimension", "Unknown"),
            "status": "Ready"
        }
    except Exception as e:
        logger.error(f"Error fetching index stats: {str(e)}")
        return {"error": str(e), "status": "Error"}


@mcp.resource("schema://food-categories")
def get_food_categories() -> dict:
    """
    Get available food categories from index metadata.
    Helps LLM understand what types of foods are available.
    """
    try:
        return {
            "categories": [
                "breakfast",
                "lunch",
                "dinner",
                "snacks",
                "beverages",
                "desserts",
                "healthy",
                "comfort-food",
                "vegetarian",
                "vegan"
            ],
            "description": "Food categories used in the NutriMood database"
        }
    except Exception as e:
        logger.error(f"Error fetching categories: {str(e)}")
        return {"error": str(e)}


# ============================================================================
# TOOLS - Executable functions for LLM actions
# ============================================================================

@mcp.tool()
def search_food_by_description(
    query: str,
    top_k: int = 5,
    namespace: str = "default",
    include_metadata: bool = True
) -> str:
    """
    Semantic search for food items by description or preference.
    
    Args:
        query: Natural language description (e.g., "healthy breakfast with protein")
        top_k: Number of results to return (default: 5, max: 20)
        namespace: Pinecone namespace to search in
        include_metadata: Include food metadata in results
    
    Returns: JSON string with matching food items and similarity scores
    
    Example query: "I want something spicy and vegetarian"
    """
    try:
        if not query or len(query.strip()) < 2:
            return json.dumps({"error": "Query must be at least 2 characters"})
        
        # Limit top_k for safety
        top_k = min(top_k, 20)
        
        index = get_index()
        
        # Perform semantic search
        results = index.query(
            vector=[0] * 1536,  # Placeholder - Pinecone will embed for integrated models
            text=query,  # Use text query with integrated embedding
            namespace=namespace,
            top_k=top_k,
            include_metadata=include_metadata,
            include_values=False
        )
        
        formatted_results = format_search_results(results)
        logger.info(f"Search query: '{query}' returned {len(formatted_results['matches'])} results")
        
        return json.dumps(formatted_results, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return json.dumps({"error": f"Search failed: {str(e)}"})


@mcp.tool()
def search_food_by_category(
    category: str,
    top_k: int = 10,
    namespace: str = "default"
) -> str:
    """
    Search food items filtered by category using metadata.
    
    Args:
        category: Food category (breakfast, lunch, dinner, snacks, etc.)
        top_k: Number of results to return
        namespace: Pinecone namespace
    
    Returns: JSON string with filtered food items
    
    Example: category="vegetarian" returns all vegetarian options
    """
    try:
        index = get_index()
        
        # Use metadata filter to find items in category
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
        
        return json.dumps(formatted_results, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Category search error: {str(e)}")
        return json.dumps({"error": f"Category search failed: {str(e)}"})


@mcp.tool()
def upsert_food_item(
    item_id: str,
    name: str,
    description: str,
    category: str,
    nutrition_info: dict,
    mood_tags: Optional[list] = None,
    namespace: str = "default",
    vector_values: Optional[list] = None
) -> str:
    """
    Add or update a food item in the vector database.
    
    Args:
        item_id: Unique identifier for the food item
        name: Name of the food item
        description: Description of the food (used for semantic search)
        category: Food category
        nutrition_info: Dict with calories, protein, carbs, fat, etc.
        mood_tags: List of mood tags (happy, energetic, calm, etc.)
        namespace: Pinecone namespace
        vector_values: Pre-computed embedding vector (optional)
    
    Returns: JSON with upsert confirmation
    
    Example: Add "Grilled Chicken Salad" with health metrics
    """
    try:
        if not item_id or not name or not description:
            return json.dumps({"error": "item_id, name, and description are required"})
        
        index = get_index()
        
        # Prepare metadata
        metadata = {
            "name": name,
            "category": category,
            "description": description,
            "nutrition_info": nutrition_info,
            "mood_tags": mood_tags or []
        }
        
        # Use random vector if not provided (Pinecone will embed for integrated models)
        if vector_values is None:
            vector_values = [0.1] * 1536  # Placeholder vector
        
        # Upsert to Pinecone
        index.upsert(
            vectors=[(item_id, vector_values, metadata)],
            namespace=namespace
        )
        
        logger.info(f"Upserted food item: {name} (ID: {item_id})")
        
        return json.dumps({
            "status": "success",
            "message": f"Food item '{name}' added/updated successfully",
            "item_id": item_id,
            "namespace": namespace
        })
        
    except Exception as e:
        logger.error(f"Upsert error: {str(e)}")
        return json.dumps({"error": f"Upsert failed: {str(e)}"})


@mcp.tool()
def get_food_details(
    item_id: str,
    namespace: str = "default"
) -> str:
    """
    Retrieve detailed information about a specific food item.
    
    Args:
        item_id: Unique identifier of the food item
        namespace: Pinecone namespace
    
    Returns: JSON with complete food item details
    """
    try:
        index = get_index()
        
        # Fetch specific item
        result = index.fetch(
            ids=[item_id],
            namespace=namespace
        )
        
        if not result.vectors:
            return json.dumps({
                "error": f"Food item '{item_id}' not found",
                "item_id": item_id
            })
        
        food_item = result.vectors[0]
        return json.dumps({
            "id": food_item.id,
            "metadata": food_item.metadata,
            "found": True
        }, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Fetch error: {str(e)}")
        return json.dumps({"error": f"Failed to fetch item: {str(e)}"})


@mcp.tool()
def delete_food_item(
    item_id: str,
    namespace: str = "default"
) -> str:
    """
    Delete a food item from the vector database.
    
    Args:
        item_id: Unique identifier of the food item to delete
        namespace: Pinecone namespace
    
    Returns: JSON with deletion confirmation
    """
    try:
        index = get_index()
        
        index.delete(
            ids=[item_id],
            namespace=namespace
        )
        
        logger.info(f"Deleted food item: {item_id}")
        
        return json.dumps({
            "status": "success",
            "message": f"Food item deleted successfully",
            "item_id": item_id
        })
        
    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
        return json.dumps({"error": f"Delete failed: {str(e)}"})


@mcp.tool()
def search_by_mood(
    mood: str,
    query: Optional[str] = None,
    top_k: int = 5,
    namespace: str = "default"
) -> str:
    """
    Find food recommendations based on mood using semantic search and metadata filtering.
    
    Args:
        mood: Target mood (happy, energetic, calm, focused, sad, stressed, etc.)
        query: Optional natural language query to refine search
        top_k: Number of results
        namespace: Pinecone namespace
    
    Returns: JSON with mood-matched food items
    
    Example: mood="energetic" finds foods tagged for energy boost
    """
    try:
        index = get_index()
        
        # Build search query
        search_query = query or f"Food for {mood} mood"
        
        # Create metadata filter for mood
        filter_condition = {"mood_tags": {"$in": [mood]}}
        
        # Search with mood filter
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
        
        return json.dumps(formatted_results, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Mood search error: {str(e)}")
        return json.dumps({"error": f"Mood search failed: {str(e)}"})


@mcp.tool()
def list_all_food_items(
    namespace: str = "default",
    limit: int = 100
) -> str:
    """
    List all food items in the index (with pagination).
    
    Args:
        namespace: Pinecone namespace to list from
        limit: Maximum items to return (max 1000)
    
    Returns: JSON with list of all food items and their metadata
    """
    try:
        index = get_index()
        limit = min(limit, 1000)  # Cap limit for safety
        
        # List all vectors from namespace
        results = index.list(
            namespace=namespace,
            limit=limit
        )
        
        # Convert IDs to full items with metadata
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
        
        return json.dumps({
            "total_items": len(items),
            "namespace": namespace,
            "items": items
        }, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"List error: {str(e)}")
        return json.dumps({"error": f"Failed to list items: {str(e)}"})


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Determine transport from command line or env variable
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    
    if len(sys.argv) > 1:
        transport = sys.argv[1]
    
    logger.info(f"🚀 Starting MCP server with {transport} transport")
    
    if transport == "sse":
        # For remote access (e.g., from AWS Bedrock or remote clients)
        port = int(os.getenv("MCP_PORT", 8000))
        host = os.getenv("MCP_HOST", "127.0.0.1")
        logger.info(f"📡 Server listening on {host}:{port}")
        mcp.run(transport="sse", host=host, port=port)
    else:
        # For local development with Claude Desktop (default: stdio)
        logger.info("📟 Using stdio transport (local mode)")
        mcp.run(transport="stdio")