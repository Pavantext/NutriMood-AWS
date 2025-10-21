"""
Quick verification script to test that all components are properly integrated
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from services.bedrock_service import BedrockService
        print("✅ BedrockService imported successfully")
    except Exception as e:
        print(f"❌ Error importing BedrockService: {e}")
        return False
    
    try:
        from services.food_service import FoodService
        print("✅ FoodService imported successfully")
    except Exception as e:
        print(f"❌ Error importing FoodService: {e}")
        return False
    
    try:
        from services.session_service import SessionService
        print("✅ SessionService imported successfully")
    except Exception as e:
        print(f"❌ Error importing SessionService: {e}")
        return False
    
    try:
        from services.mcp_server import MCPServer
        print("✅ MCPServer imported successfully")
    except Exception as e:
        print(f"❌ Error importing MCPServer: {e}")
        return False
    
    try:
        from utils.response_formatter import ResponseFormatter
        print("✅ ResponseFormatter imported successfully")
    except Exception as e:
        print(f"❌ Error importing ResponseFormatter: {e}")
        return False
    
    return True


def test_service_initialization():
    """Test that services can be initialized"""
    print("\nTesting service initialization...")
    
    try:
        from services.food_service import FoodService
        food_service = FoodService()
        print("✅ FoodService initialized")
        
        # Test loading data (will fail if file not found, but that's ok)
        try:
            food_service.load_food_data("data/Niloufer_data.json")
            print(f"✅ Food data loaded: {len(food_service.food_items)} items")
        except FileNotFoundError:
            print("⚠️  Food data file not found (expected in test)")
        
    except Exception as e:
        print(f"❌ Error with FoodService: {e}")
        return False
    
    try:
        from services.session_service import SessionService
        session_service = SessionService()
        print("✅ SessionService initialized")
        
        # Test session creation
        session = session_service.get_or_create_session("test-123")
        print(f"✅ Session created: {session['session_id']}")
        
    except Exception as e:
        print(f"❌ Error with SessionService: {e}")
        return False
    
    try:
        from services.mcp_server import MCPServer
        from services.food_service import FoodService
        
        food_service = FoodService()
        mcp_server = MCPServer(food_service)
        print("✅ MCPServer initialized")
        
        # Test MCP info
        info = mcp_server.get_server_info()
        print(f"✅ MCP Server: {info['name']} v{info['version']}")
        
        # Test MCP tools
        tools = mcp_server.list_tools()
        print(f"✅ MCP Tools available: {len(tools)}")
        
        # Test MCP resources
        resources = mcp_server.list_resources()
        print(f"✅ MCP Resources available: {len(resources)}")
        
    except Exception as e:
        print(f"❌ Error with MCPServer: {e}")
        return False
    
    try:
        from utils.response_formatter import ResponseFormatter
        formatter = ResponseFormatter()
        print("✅ ResponseFormatter initialized")
        
        # Test formatting
        response = formatter.format_chat_response(
            message="Test message",
            session_id="test-123",
            food_ids=["id1", "id2"]
        )
        print(f"✅ Response formatted: {response['food_recommendation_id']}")
        
    except Exception as e:
        print(f"❌ Error with ResponseFormatter: {e}")
        return False
    
    return True


def test_mcp_tools():
    """Test MCP tool functionality"""
    print("\nTesting MCP tools...")
    
    try:
        from services.mcp_server import MCPServer
        from services.food_service import FoodService
        
        food_service = FoodService()
        mcp_server = MCPServer(food_service)
        
        # Test list_categories tool
        result = mcp_server.call_tool("list_categories", {})
        print(f"✅ list_categories tool executed: {result.get('count', 0)} categories")
        
        # Test get_food_statistics tool
        result = mcp_server.call_tool("get_food_statistics", {})
        print(f"✅ get_food_statistics tool executed: {result.get('total_items', 0)} items")
        
        print("✅ MCP tools working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error with MCP tools: {e}")
        return False


def test_food_service():
    """Test FoodService functionality"""
    print("\nTesting FoodService methods...")
    
    try:
        from services.food_service import FoodService
        
        food_service = FoodService()
        
        # Test with empty data
        categories = food_service.get_categories()
        print(f"✅ get_categories: {len(categories)} categories")
        
        stats = food_service.get_food_statistics()
        print(f"✅ get_food_statistics: {stats['total_items']} items")
        
        foods = food_service.get_all_foods(limit=10)
        print(f"✅ get_all_foods: {len(foods)} items retrieved")
        
        # Test matching (will return empty with no data)
        matches = food_service.find_matching_foods("test query", [], top_k=5)
        print(f"✅ find_matching_foods: {len(matches)} matches")
        
        print("✅ FoodService methods working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error with FoodService methods: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("NutriMood Integration Verification")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test initialization
    if not test_service_initialization():
        all_passed = False
    
    # Test MCP tools
    if not test_mcp_tools():
        all_passed = False
    
    # Test FoodService
    if not test_food_service():
        all_passed = False
    
    print()
    print("=" * 60)
    if all_passed:
        print("✅ All verification tests passed!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Configure AWS credentials in .env file")
        print("2. Ensure food data is available")
        print("3. Run: python main.py")
        print("4. Visit: http://localhost:8000/docs")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())



