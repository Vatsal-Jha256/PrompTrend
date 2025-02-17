import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json
from main import app
from services.intent_classifier import IntentClassifier
from services.recommender import ContextualBandit
import time
client = TestClient(app)

def generate_chat_history(num_messages=3):
    """Helper function to generate test chat history"""
    base_time = datetime.now()
    messages = []
    test_contents = [
        "How do I use this API?",
        "I'm getting an error when calling the endpoint",
        "Can you help me debug this issue?",
        "What are the best practices for prompt engineering?",
        "Is there a way to improve response quality?"
    ]
    
    for i in range(num_messages):
        messages.append({
            "content": test_contents[i % len(test_contents)],
            "timestamp": (base_time + timedelta(minutes=i)).isoformat()
        })
    return {"messages": messages}

class TestIntentClassifier:
    @pytest.fixture
    def initialized_classifier(self):
        """Initialize classifier once for all tests"""
        response = client.post(
            "/initialize",
            json={
                "model_path": "bert-base-uncased",
                "num_labels": 10,
                "use_pretrained": True
            }
        )
        assert response.status_code == 200
        return response
    
    def test_classifier_initialization(self, initialized_classifier):
        """Test basic initialization"""
        assert initialized_classifier.json()["status"] == "success"
    
    def test_initialization_with_invalid_model(self):
        """Test initialization with invalid model path"""
        response = client.post(
            "/initialize",
            json={
                "model_path": "invalid-model-path",
                "num_labels": 10,
                "use_pretrained": True
            }
        )
        assert response.status_code == 500
    
    def test_training_with_empty_data(self, initialized_classifier):
        """Test training with empty dataset"""
        response = client.post(
            "/train",
            json={
                "training_data": [],
                "epochs": 1
            }
        )
        assert response.status_code == 400
    
    def test_training_with_invalid_labels(self, initialized_classifier):
        """Test training with invalid label values"""
        response = client.post(
            "/train",
            json={
                "training_data": [
                    {"text": "Test text", "label": 999}  # Invalid label
                ],
                "epochs": 1
            }
        )
        assert response.status_code == 400

class TestRecommendationSystem:
    @pytest.fixture
    def initialized_system(self):
        """Initialize the complete system"""
        response = client.post(
            "/initialize",
            json={
                "model_path": "bert-base-uncased",
                "num_labels": 10,
                "use_pretrained": True
            }
        )
        assert response.status_code == 200
        return response
    
    def test_basic_recommendation(self, initialized_system):
        """Test basic recommendation functionality"""
        response = client.post(
            "/recommendations",
            json={
                "user_id": "test_user",
                "chat_history": generate_chat_history(3),
                "categories": ["technical", "support", "general"],
                "num_recommendations": 2
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["recommendations"]) == 2
        assert all(isinstance(score, float) for score in data["confidence_scores"].values())
    
    def test_empty_chat_history(self, initialized_system):
        """Test recommendations with empty chat history"""
        response = client.post(
            "/recommendations",
            json={
                "user_id": "test_user",
                "chat_history": {"messages": []},
                "categories": ["technical", "support"],
                "num_recommendations": 2
            }
        )
        assert response.status_code == 400
    
    def test_long_chat_history(self, initialized_system):
        """Test with a long chat history"""
        response = client.post(
            "/recommendations",
            json={
                "user_id": "test_user",
                "chat_history": generate_chat_history(10),
                "categories": ["technical", "support"],
                "num_recommendations": 2
            }
        )
        assert response.status_code == 200
        
        # tests/test_api.py
    def test_multiple_users(initialized_system):
        """Test recommendations for multiple users"""
        users = ["user1", "user2", "user3"]
        responses = []
        recommendations_before = []
        
        # Get initial recommendations for all users
        for user in users:
            try:
                response = client.post(
                    "/recommendations",
                    json={
                        "user_id": user,
                        "chat_history": generate_chat_history(3),
                        "categories": ["technical", "support"],
                        "num_recommendations": 2
                    }
                )
                recommendations_before.append(response.json()["recommendations"])
                
                # Simulate different feedback for each user
                feedback_response = client.post(
                    "/feedback",
                    json={
                        "user_id": user,
                        "recommendation_id": 1,
                        "feedback_score": (hash(user) % 4) + 1  # Scores between 1-5
                    }
                )
                assert feedback_response.status_code == 200, f"Feedback failed with status {feedback_response.status_code}: {feedback_response.text}"
                
                # Add some delay to ensure feedback is processed
                time.sleep(0.1)
                
            except Exception as e:
                pytest.fail(f"Error processing user {user}: {str(e)}")
def test_edge_cases():
    """Test various edge cases"""
    client.post("/initialize", json={
        "model_path": "bert-base-uncased",
        "num_labels": 10,
        "use_pretrained": True
    })
    
    # Test extremely short message
    response = client.post(
        "/recommendations",
        json={
            "user_id": "test_user",
            "chat_history": {
                "messages": [{"content": "Hi", "timestamp": datetime.now().isoformat()}]
            },
            "categories": ["technical", "support"],
            "num_recommendations": 2
        }
    )
    assert response.status_code == 200
    
    # Test message with special characters
    response = client.post(
        "/recommendations",
        json={
            "user_id": "test_user",
            "chat_history": {
                "messages": [{
                    "content": "!@#$%^&*()_+ Test message with špęćïął characters",
                    "timestamp": datetime.now().isoformat()
                }]
            },
            "categories": ["technical", "support"],
            "num_recommendations": 2
        }
    )
    assert response.status_code == 200
    
    # Test extremely large number of recommendations
    response = client.post(
        "/recommendations",
        json={
            "user_id": "test_user",
            "chat_history": generate_chat_history(3),
            "categories": ["technical", "support"],
            "num_recommendations": 999  # Exceeds Field(le=10)
        }
    )
    assert response.status_code == 422  # Changed from 400 to 422 for Pydantic validation error