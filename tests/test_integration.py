# tests/test_integration.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import logging
from main import app
from services.question_generator import QuestionGenerator
from services.error_handler import handle_error, ModelError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = TestClient(app)

def test_complete_flow():
    """Test the complete flow with actual recommendations"""
    try:
        # 1. Initialize the system
        logger.info("Initializing the system...")
        init_response = client.post(
            "/initialize",
            json={
                "model_path": "bert-base-uncased",
                "num_labels": 10,
                "use_pretrained": True
            }
        )
        assert init_response.status_code == 200
        logger.info("System initialized successfully")

        # 2. Train with sample data
        logger.info("Training the system...")
        training_data = [
            {"text": "How do I use the API authentication?", "label": 0},
            {"text": "Getting errors with token validation", "label": 1},
            {"text": "Best practices for API security?", "label": 2},
            {"text": "How to handle rate limiting?", "label": 3},
            {"text": "Need help with error responses", "label": 1}
        ]
        
        # Convert to proper TrainingData objects
        from core.models import TrainingData
        formatted_data = [TrainingData(**item) for item in training_data]
        
        train_response = client.post(
            "/train",
            json={
                "training_data": [item.model_dump() for item in formatted_data],
                "epochs": 1
            }
        )
        assert train_response.status_code == 200
        logger.info("Training completed successfully")

        # 3. Get recommendations for different scenarios
        test_scenarios = [
            {
                "name": "API Authentication Issues",
                "messages": [
                    "I'm having trouble with API authentication",
                    "The token validation keeps failing",
                    "I've tried different API keys"
                ]
            },
            {
                "name": "Error Handling",
                "messages": [
                    "Getting lots of 500 errors",
                    "How should I handle these exceptions?",
                    "Need better error messages"
                ]
            }
        ]

        # Initialize question generator
        question_gen = QuestionGenerator()

        for scenario in test_scenarios:
            logger.info(f"\nTesting scenario: {scenario['name']}")
            
            # Create chat history
            chat_history = {
                "messages": [
                    {
                        "content": msg,
                        "timestamp": datetime.now().isoformat()
                    } for msg in scenario["messages"]
                ]
            }

            # Get recommendations
            response = client.post(
                "/recommendations",
                json={
                    "user_id": f"test_user_{scenario['name'].lower().replace(' ', '_')}",
                    "chat_history": chat_history,
                    "categories": ["technical", "support", "documentation"],
                    "num_recommendations": 3
                }
            )
            
            assert response.status_code == 200
            result = response.json()
            
            # Print recommendations and confidence scores
            print(f"\nRecommendations for {scenario['name']}:")
            for i, rec in enumerate(result["recommendations"], 1):
                confidence = result["confidence_scores"][rec]
                print(f"{i}. {rec} (confidence: {confidence:.2f})")
                
                # Generate and print a relevant question
                question = question_gen.generate_question(rec, context=" ".join(scenario["messages"]))
                print(f"   Generated question: {question}")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

def test_database_integration():
    """Test database operations"""
    from core.database import init_db, get_db, User, Recommendation
    db = next(get_db())
    
    try:

        # Cleanup existing test user
        existing_user = db.query(User).filter(User.id == "test_user_1").first()
        if existing_user:
            db.delete(existing_user)
            db.commit()
        # Test user creation
        test_user = User(id="test_user_1")
        db.add(test_user)
        db.commit()
        
        # Test recommendation creation
        test_rec = Recommendation(
            user_id="test_user_1",
            category="test",
            recommendation="Test recommendation",
            confidence_score=0.95
        )
        db.add(test_rec)
        db.commit()
        
        # Verify data
        user = db.query(User).filter(User.id == "test_user_1").first()
        assert user is not None
        
        rec = db.query(Recommendation).filter(
            Recommendation.user_id == "test_user_1"
        ).first()
        assert rec is not None
        assert rec.confidence_score == 0.95
        
    finally:
        # Cleanup
        #db.rollback()
        db.close()

@pytest.mark.asyncio
async def test_cache_integration():
    """Test that caching is working properly"""
    from core.cache import RedisCache
    cache = RedisCache()
    
    # Test basic cache operations
    test_key = "test_key"
    test_value = {"data": "test"}
    
    # Set value
    assert cache.set(test_key, test_value)
    
    # Get value
    cached_value = cache.get(test_key)
    assert cached_value == test_value
    
    # Delete value
    assert cache.delete(test_key)
    assert cache.get(test_key) is None
@pytest.fixture
def clean_db():
    from core.database import init_db, get_db, User
    init_db()
    db = next(get_db())
    try:
        # Clean all test data
        db.query(User).delete()
        db.commit()
        yield db
    finally:
        db.close()

def test_error_handling():
    """Test error handling with invalid inputs"""
    try:
        # Test with invalid model path
        response = client.post(
            "/initialize",
            json={
                "model_path": "invalid-model-path",
                "num_labels": 10,
                "use_pretrained": True
            }
        )
        assert response.status_code == 500
        print("\nError handling test - Invalid model path:")
        print(response.json())

        # Test with invalid chat history
        response = client.post(
            "/recommendations",
            json={
                "user_id": "test_user",
                "chat_history": {"messages": []},  # Empty chat history
                "categories": ["technical"],
                "num_recommendations": 3
            }
        )
        assert response.status_code == 400
        print("\nError handling test - Empty chat history:")
        print(response.json())

    except Exception as e:
        logger.error(f"Error handling test failed: {str(e)}")
        raise

if __name__ == "__main__":
    print("Running integration tests with actual recommendations...")
    test_complete_flow()
    print("\nTesting error handling...")
    test_error_handling()