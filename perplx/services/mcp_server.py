"""
MCP Server - Model Context Protocol integration for food data
Provides structured access to food database via MCP protocol
"""

import json
from typing import Dict, List, Optional
from datetime import datetime


class MCPServer:
    """
    MCP (Model Context Protocol) Server for NutriMood
    Provides structured interface for LLMs to query food database
    """
    
    def __init__(self, food_service):
        """
        Initialize MCP Server
        
        Args:
            food_service: FoodService instance for accessing food data
        """
        self.food_service = food_service
        self.server_info = {
            "name": "nutrimood-food-mcp",
            "version": "1.0.0",
            "protocol_version": "2024-01-01",
            "capabilities": {
                "tools": True,
                "resources": True,
                "prompts": True
            }
        }
        
        # Define available MCP tools
        self.tools = self._define_tools()
        
        # Define available resources
        self.resources = self._define_resources()
        
        # Define available prompts
        self.prompts = self._define_prompts()
    
    def _define_tools(self) -> List[Dict]:
        """Define MCP tools for food operations"""
        return [
            {
                "name": "search_foods",
                "description": "Search for food items based on query and filters",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for food items"
                        },
                        "category": {
                            "type": "string",
                            "description": "Filter by category (optional)"
                        },
                        "max_calories": {
                            "type": "number",
                            "description": "Maximum calories per item (optional)"
                        },
                        "min_calories": {
                            "type": "number",
                            "description": "Minimum calories per item (optional)"
                        },
                        "dietary": {
                            "type": "string",
                            "description": "Dietary preference filter (optional)"
                        },
                        "top_k": {
                            "type": "number",
                            "description": "Number of results to return",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_food_by_id",
                "description": "Get detailed information about a specific food item",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "food_id": {
                            "type": "string",
                            "description": "Unique identifier for the food item"
                        }
                    },
                    "required": ["food_id"]
                }
            },
            {
                "name": "list_categories",
                "description": "List all available food categories",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_food_statistics",
                "description": "Get statistics about the food database",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "recommend_foods",
                "description": "Get personalized food recommendations based on preferences",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "preferences": {
                            "type": "object",
                            "description": "User preferences (dietary, calorie_goal, mood, etc.)"
                        },
                        "exclude_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Food IDs to exclude from recommendations"
                        },
                        "top_k": {
                            "type": "number",
                            "description": "Number of recommendations",
                            "default": 5
                        }
                    },
                    "required": ["preferences"]
                }
            }
        ]
    
    def _define_resources(self) -> List[Dict]:
        """Define MCP resources for food data access"""
        return [
            {
                "uri": "nutrimood://foods",
                "name": "All Foods",
                "description": "Access to complete food database",
                "mimeType": "application/json"
            },
            {
                "uri": "nutrimood://categories",
                "name": "Food Categories",
                "description": "List of all food categories",
                "mimeType": "application/json"
            },
            {
                "uri": "nutrimood://statistics",
                "name": "Database Statistics",
                "description": "Statistics about the food database",
                "mimeType": "application/json"
            }
        ]
    
    def _define_prompts(self) -> List[Dict]:
        """Define MCP prompts for common use cases"""
        return [
            {
                "name": "food_recommendation",
                "description": "Generate food recommendation based on user preferences",
                "arguments": [
                    {
                        "name": "mood",
                        "description": "User's current mood",
                        "required": False
                    },
                    {
                        "name": "dietary_preference",
                        "description": "Dietary preference (vegetarian, vegan, etc.)",
                        "required": False
                    },
                    {
                        "name": "calorie_goal",
                        "description": "Target calorie range",
                        "required": False
                    }
                ]
            },
            {
                "name": "meal_planning",
                "description": "Help plan meals for specific requirements",
                "arguments": [
                    {
                        "name": "meal_type",
                        "description": "Type of meal (breakfast, lunch, dinner, snack)",
                        "required": True
                    },
                    {
                        "name": "dietary_restrictions",
                        "description": "Any dietary restrictions",
                        "required": False
                    }
                ]
            }
        ]
    
    def get_server_info(self) -> Dict:
        """Get MCP server information"""
        return self.server_info
    
    def list_tools(self) -> List[Dict]:
        """List available MCP tools"""
        return self.tools
    
    def list_resources(self) -> List[Dict]:
        """List available MCP resources"""
        return self.resources
    
    def list_prompts(self) -> List[Dict]:
        """List available MCP prompts"""
        return self.prompts
    
    def call_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """
        Execute an MCP tool
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
        
        Returns:
            Tool execution result
        """
        try:
            if tool_name == "search_foods":
                return self._tool_search_foods(arguments)
            
            elif tool_name == "get_food_by_id":
                return self._tool_get_food_by_id(arguments)
            
            elif tool_name == "list_categories":
                return self._tool_list_categories(arguments)
            
            elif tool_name == "get_food_statistics":
                return self._tool_get_food_statistics(arguments)
            
            elif tool_name == "recommend_foods":
                return self._tool_recommend_foods(arguments)
            
            else:
                return {
                    "error": f"Unknown tool: {tool_name}",
                    "available_tools": [tool["name"] for tool in self.tools]
                }
        
        except Exception as e:
            return {
                "error": f"Tool execution failed: {str(e)}",
                "tool": tool_name,
                "arguments": arguments
            }
    
    def _tool_search_foods(self, args: Dict) -> Dict:
        """Execute search_foods tool"""
        query = args.get("query", "")
        top_k = args.get("top_k", 5)
        
        # Build filters
        filters = {}
        if "category" in args:
            filters["category"] = args["category"]
        if "max_calories" in args:
            filters["max_calories"] = args["max_calories"]
        if "min_calories" in args:
            filters["min_calories"] = args["min_calories"]
        if "dietary" in args:
            filters["dietary"] = args["dietary"]
        
        # Search using food service
        matches = self.food_service.find_matching_foods(
            query=query,
            conversation_history=[],
            top_k=top_k,
            filters=filters if filters else None
        )
        
        # Format results
        results = []
        for food, score in matches:
            results.append({
                "id": food.get("Id"),
                "name": food.get("ProductName"),
                "description": food.get("Description"),
                "category": food.get("KioskCategoryName"),
                "calories": food.get("calories"),
                "price": food.get("Price"),
                "relevance_score": round(score, 3)
            })
        
        return {
            "query": query,
            "count": len(results),
            "results": results
        }
    
    def _tool_get_food_by_id(self, args: Dict) -> Dict:
        """Execute get_food_by_id tool"""
        food_id = args.get("food_id")
        
        if not food_id:
            return {"error": "food_id is required"}
        
        food = self.food_service.get_food_by_id(food_id)
        
        if not food:
            return {"error": f"Food not found: {food_id}"}
        
        return {
            "food": food
        }
    
    def _tool_list_categories(self, args: Dict) -> Dict:
        """Execute list_categories tool"""
        categories = self.food_service.get_categories()
        
        return {
            "categories": categories,
            "count": len(categories)
        }
    
    def _tool_get_food_statistics(self, args: Dict) -> Dict:
        """Execute get_food_statistics tool"""
        stats = self.food_service.get_food_statistics()
        
        return stats
    
    def _tool_recommend_foods(self, args: Dict) -> Dict:
        """Execute recommend_foods tool"""
        preferences = args.get("preferences", {})
        exclude_ids = args.get("exclude_ids", [])
        top_k = args.get("top_k", 5)
        
        # Build query from preferences
        query_parts = []
        
        if "mood" in preferences:
            query_parts.append(preferences["mood"])
        
        if "dietary" in preferences:
            query_parts.append(preferences["dietary"])
        
        if "cuisine" in preferences:
            query_parts.append(preferences["cuisine"])
        
        query = " ".join(query_parts) if query_parts else "food"
        
        # Build filters from preferences
        filters = {}
        if "calorie_max" in preferences:
            filters["max_calories"] = preferences["calorie_max"]
        if "calorie_min" in preferences:
            filters["min_calories"] = preferences["calorie_min"]
        if "category" in preferences:
            filters["category"] = preferences["category"]
        
        # Get recommendations
        matches = self.food_service.find_matching_foods(
            query=query,
            conversation_history=[],
            top_k=top_k * 2,  # Get more to filter
            filters=filters if filters else None
        )
        
        # Filter out excluded IDs
        filtered_matches = [
            (food, score) for food, score in matches
            if food.get("Id") not in exclude_ids
        ][:top_k]
        
        # Format results
        recommendations = []
        for food, score in filtered_matches:
            recommendations.append({
                "id": food.get("Id"),
                "name": food.get("ProductName"),
                "description": food.get("Description"),
                "category": food.get("KioskCategoryName"),
                "calories": food.get("calories"),
                "price": food.get("Price"),
                "score": round(score, 3)
            })
        
        return {
            "preferences": preferences,
            "count": len(recommendations),
            "recommendations": recommendations
        }
    
    def read_resource(self, uri: str) -> Dict:
        """
        Read an MCP resource
        
        Args:
            uri: Resource URI
        
        Returns:
            Resource content
        """
        if uri == "nutrimood://foods":
            return {
                "uri": uri,
                "mimeType": "application/json",
                "content": self.food_service.get_all_foods(limit=100)
            }
        
        elif uri == "nutrimood://categories":
            return {
                "uri": uri,
                "mimeType": "application/json",
                "content": self.food_service.get_categories()
            }
        
        elif uri == "nutrimood://statistics":
            return {
                "uri": uri,
                "mimeType": "application/json",
                "content": self.food_service.get_food_statistics()
            }
        
        else:
            return {
                "error": f"Unknown resource: {uri}",
                "available_resources": [r["uri"] for r in self.resources]
            }
    
    def get_prompt(self, prompt_name: str, arguments: Dict) -> str:
        """
        Generate a prompt template
        
        Args:
            prompt_name: Name of the prompt
            arguments: Prompt arguments
        
        Returns:
            Generated prompt text
        """
        if prompt_name == "food_recommendation":
            return self._prompt_food_recommendation(arguments)
        
        elif prompt_name == "meal_planning":
            return self._prompt_meal_planning(arguments)
        
        else:
            return f"Unknown prompt: {prompt_name}"
    
    def _prompt_food_recommendation(self, args: Dict) -> str:
        """Generate food recommendation prompt"""
        prompt_parts = [
            "Generate a food recommendation based on the following:"
        ]
        
        if "mood" in args:
            prompt_parts.append(f"- User's mood: {args['mood']}")
        
        if "dietary_preference" in args:
            prompt_parts.append(f"- Dietary preference: {args['dietary_preference']}")
        
        if "calorie_goal" in args:
            prompt_parts.append(f"- Calorie goal: {args['calorie_goal']}")
        
        prompt_parts.append("\nProvide 3-5 food recommendations with explanations.")
        
        return "\n".join(prompt_parts)
    
    def _prompt_meal_planning(self, args: Dict) -> str:
        """Generate meal planning prompt"""
        meal_type = args.get("meal_type", "meal")
        
        prompt = f"Help plan a {meal_type} with the following requirements:\n"
        
        if "dietary_restrictions" in args:
            prompt += f"- Dietary restrictions: {args['dietary_restrictions']}\n"
        
        prompt += "\nProvide a complete meal plan with food items from the database."
        
        return prompt
    
    def to_mcp_format(self) -> Dict:
        """
        Export server configuration in MCP format
        
        Returns:
            MCP server configuration
        """
        return {
            "server": self.server_info,
            "tools": self.tools,
            "resources": self.resources,
            "prompts": self.prompts
        }

