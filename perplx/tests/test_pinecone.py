"""
Test script for Pinecone vector search integration
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))


def test_embedding_service():
    """Test Google Gemini embedding generation"""
    print("\n" + "="*60)
    print("Testing Embedding Service (Google Gemini)")
    print("="*60)
    
    try:
        from services.embedding_service import EmbeddingService
        
        embedding_service = EmbeddingService()
        
        # Test embedding generation
        test_queries = [
            "something spicy",
            "healthy breakfast",
            "comfort food"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            embedding = embedding_service.generate_embedding(query)
            
            if embedding:
                print(f"✅ Generated embedding with {len(embedding)} dimensions")
                print(f"   First 5 values: {embedding[:5]}")
            else:
                print(f"❌ Failed to generate embedding")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing embedding service: {e}")
        return False


def test_pinecone_service():
    """Test Pinecone connection and search"""
    print("\n" + "="*60)
    print("Testing Pinecone Service")
    print("="*60)
    
    try:
        from services.pinecone_service import PineconeService
        from services.embedding_service import EmbeddingService
        
        pinecone_service = PineconeService()
        embedding_service = EmbeddingService()
        
        # Check connection
        if not pinecone_service.index:
            print("❌ Pinecone not connected")
            return False
        
        print(f"✅ Connected to Pinecone index")
        
        # Get stats
        stats = pinecone_service.get_index_stats()
        print(f"✅ Index stats:")
        print(f"   Name: {stats.get('index_name')}")
        print(f"   Total vectors: {stats.get('total_vectors')}")
        print(f"   Dimension: {stats.get('dimension')}")
        
        # Test search
        print(f"\nTesting vector search...")
        test_query = "something spicy"
        
        query_embedding = embedding_service.generate_embedding(test_query)
        
        if not query_embedding:
            print("❌ Could not generate embedding for search")
            return False
        
        results = pinecone_service.search_foods(
            query_embedding=query_embedding,
            top_k=5
        )
        
        print(f"\n✅ Search results for '{test_query}':")
        for idx, (food, score) in enumerate(results, 1):
            print(f"   {idx}. {food.get('ProductName')} (similarity: {score:.3f})")
            print(f"      Category: {food.get('KioskCategoryName')}")
            print(f"      Calories: {food.get('calories')} cal")
        
        # Test with filters
        print(f"\nTesting filtered search (max 300 calories)...")
        results_filtered = pinecone_service.search_foods(
            query_embedding=query_embedding,
            top_k=3,
            filters={"max_calories": 300}
        )
        
        print(f"✅ Filtered results:")
        for idx, (food, score) in enumerate(results_filtered, 1):
            print(f"   {idx}. {food.get('ProductName')} ({food.get('calories')} cal)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Pinecone service: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_food_service_integration():
    """Test FoodService with Pinecone integration"""
    print("\n" + "="*60)
    print("Testing FoodService Integration")
    print("="*60)
    
    try:
        from services.food_service import FoodService
        
        food_service = FoodService()
        
        # Check if vector search is enabled
        print(f"\nVector search enabled: {food_service.use_vector_search}")
        
        if not food_service.use_vector_search:
            print("⚠️  Vector search not enabled - check API keys")
            return False
        
        # Test find_matching_foods
        test_queries = [
            "something spicy",
            "healthy breakfast",
            "comfort food for rainy day",
            "light snack"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            matches = food_service.find_matching_foods(
                query=query,
                conversation_history=[],
                top_k=3
            )
            
            if matches:
                print(f"✅ Found {len(matches)} matches:")
                for idx, (food, score) in enumerate(matches, 1):
                    print(f"   {idx}. {food.get('ProductName')} (score: {score:.3f})")
            else:
                print(f"⚠️  No matches found")
        
        # Test food context building
        print(f"\n📝 Testing context building...")
        matches = food_service.find_matching_foods("spicy", [], top_k=2)
        context = food_service.build_food_context(matches)
        
        print(f"✅ Context built ({len(context)} characters)")
        print(f"   Preview: {context[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing FoodService: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end():
    """Test complete flow"""
    print("\n" + "="*60)
    print("Testing End-to-End Flow")
    print("="*60)
    
    try:
        print("\n📊 Simulating: User asks 'I want something spicy'")
        
        from services.food_service import FoodService
        from services.bedrock_service import BedrockService
        
        food_service = FoodService()
        
        # Step 1: Find matching foods
        print("\n1️⃣  Finding matching foods...")
        matches = food_service.find_matching_foods(
            query="I want something spicy",
            conversation_history=[],
            top_k=5
        )
        
        print(f"✅ Found {len(matches)} matches via {'vector search' if food_service.use_vector_search else 'keyword matching'}")
        for idx, (food, score) in enumerate(matches, 1):
            print(f"   {idx}. {food.get('ProductName')} ({score:.3f})")
        
        # Step 2: Build context
        print("\n2️⃣  Building food context...")
        context = food_service.build_food_context(matches)
        print(f"✅ Context built ({len(context)} chars)")
        
        # Step 3: Extract IDs
        print("\n3️⃣  Testing ID extraction...")
        test_response = f"Try the {matches[0][0].get('ProductName')}! It's amazing!"
        ids = food_service.extract_food_ids_from_response(test_response, matches)
        print(f"✅ Extracted {len(ids)} food IDs")
        print(f"   IDs: {ids}")
        
        print("\n✅ End-to-end flow working!")
        return True
        
    except Exception as e:
        print(f"❌ Error in end-to-end test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("🧪 Pinecone Integration Test Suite")
    print("="*60)
    
    all_passed = True
    
    # Check environment variables
    print("\n📋 Checking environment variables...")
    required_vars = [
        "PINECONE_API_KEY",
        "GOOGLE_API_KEY",
        "PINECONE_INDEX_NAME"
    ]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            masked = value[:10] + "..." if len(value) > 10 else value
            print(f"✅ {var}: {masked}")
        else:
            print(f"❌ {var}: Not set!")
            all_passed = False
    
    if not all_passed:
        print("\n⚠️  Please set required environment variables in .env file")
        print("   See env.example for reference")
        return 1
    
    # Run tests
    print("\n" + "="*60)
    print("Running Tests")
    print("="*60)
    
    tests = [
        ("Embedding Service", test_embedding_service),
        ("Pinecone Service", test_pinecone_service),
        ("FoodService Integration", test_food_service_integration),
        ("End-to-End Flow", test_end_to_end)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed! Pinecone integration is working!")
        print("\nYou can now run: python main.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

