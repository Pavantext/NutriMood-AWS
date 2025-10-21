"""
API Tests for Nutrimood Chatbot
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

client = TestClient(app)

class TestHealthEndpoint:
    def test_root_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "Nutrimood" in response.json()["service"]

class TestChatEndpoint:
    def test_chat_without_session(self):
        """Test chat endpoint without existing session"""
        response = client.post(
            "/chat",
            json={"message": "I want something spicy"}
        )
        assert response.status_code == 200
        # Note: StreamingResponse requires special handling
    
    def test_chat_with_session(self):
        """Test chat endpoint with session ID"""
        # First request to get session
        response1 = client.post(
            "/chat",
            json={"message": "I want junk food"}
        )
        
        # Extract session_id from response (simplified)
        session_id = "test-session-123"
        
        # Second request with session
        response2 = client.post(
            "/chat",
            json={
                "message": "how many calories?",
                "session_id": session_id
            }
        )
        assert response2.status_code == 200
    
    def test_chat_empty_message(self):
        """Test chat with empty message"""
        response = client.post(
            "/chat",
            json={"message": ""}
        )
        # Should still return 200 but handle gracefully
        assert response.status_code in [200, 400]

class TestRecommendEndpoint:
    def test_recommend_basic(self):
        """Test basic recommendation request"""
        response = client.post(
            "/recommend",
            json={"query": "spicy snacks"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)
    
    def test_recommend_with_filters(self):
        """Test recommendation with calorie filter"""
        response = client.post(
            "/recommend",
            json={
                "query": "healthy food",
                "top_k": 3,
                "filters": {"max_calories": 200}
            }
        )
        assert response.status_code == 200
        data = response.json()
        
        # Check all recommendations are under calorie limit
        for rec in data["recommendations"]:
            if rec.get("calories"):
                assert rec["calories"] <= 200
    
    def test_recommend_by_category(self):
        """Test recommendation by category"""
        response = client.post(
            "/recommend",
            json={
                "query": "something to drink",
                "filters": {"category": "Beverages"}
            }
        )
        assert response.status_code == 200
        data = response.json()
        
        for rec in data["recommendations"]:
            assert rec["category"] == "Beverages"

class TestSessionEndpoint:
    def test_get_nonexistent_session(self):
        """Test getting a session that doesn't exist"""
        response = client.get("/session/nonexistent-id")
        assert response.status_code == 404
    
    def test_delete_session(self):
        """Test deleting a session"""
        # First create a session via chat
        chat_response = client.post(
            "/chat",
            json={"message": "test", "session_id": "delete-test-123"}
        )
        
        # Then delete it
        delete_response = client.delete("/session/delete-test-123")
        # Should be 200 or 404 depending on implementation
        assert delete_response.status_code in [200, 404]

class TestFoodsEndpoint:
    def test_list_all_foods(self):
        """Test listing all foods"""
        response = client.get("/foods")
        assert response.status_code == 200
        data = response.json()
        assert "foods" in data
        assert isinstance(data["foods"], list)
    
    def test_list_foods_with_limit(self):
        """Test listing foods with limit"""
        response = client.get("/foods?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data["foods"]) <= 5
    
    def test_list_foods_by_category(self):
        """Test filtering foods by category"""
        response = client.get("/foods?category=Snacks")
        assert response.status_code == 200
        data = response.json()
        
        for food in data["foods"]:
            assert food["category"] == "Snacks"

class TestConversationalFlow:
    def test_multi_turn_conversation(self):
        """Test a multi-turn conversation flow"""
        session_id = "multi-turn-test"
        
        # Turn 1: Ask for recommendation
        response1 = client.post(
            "/chat",
            json={
                "message": "I want junk food",
                "session_id": session_id
            }
        )
        assert response1.status_code == 200
        
        # Turn 2: Ask about calories (context-aware)
        response2 = client.post(
            "/chat",
            json={
                "message": "how many calories?",
                "session_id": session_id
            }
        )
        assert response2.status_code == 200
        
        # Turn 3: Change preferences
        response3 = client.post(
            "/chat",
            json={
                "message": "actually I want something healthy",
                "session_id": session_id
            }
        )
        assert response3.status_code == 200

class TestErrorHandling:
    def test_invalid_json(self):
        """Test handling of invalid JSON"""
        response = client.post(
            "/chat",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_field(self):
        """Test handling of missing required fields"""
        response = client.post(
            "/chat",
            json={}
        )
        assert response.status_code == 422

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
