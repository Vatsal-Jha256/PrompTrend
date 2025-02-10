import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from api.main import app

client = TestClient(app)

def test_initialize_classifier():
    response = client.post(
        "/initialize",
        json={
            "model_path": "bert-base-uncased",
            "num_labels": 10,
            "use_pretrained": True
        }
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_train_classifier():
    # First initialize
    client.post(
        "/initialize",
        json={
            "model_path": "bert-base-uncased",
            "num_labels": 10,
            "use_pretrained": True
        }
    )
    
    # Then train
    response = client.post(
        "/train",
        json={
            "training_data": [
                {"text": "How do I fix my device?", "label": 0},
                {"text": "What are the features?", "label": 1}
            ],
            "epochs": 1
        }
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_get_recommendations():
    # First initialize
    client.post(
        "/initialize",
        json={
            "model_path": "bert-base-uncased",
            "num_labels": 10,
            "use_pretrained": True
        }
    )
    
    response = client.post(
        "/recommendations",
        json={
            "user_id": "test_user",
            "chat_history": {
                "messages": [
                    {
                        "content": "How do I fix my device?",
                        "timestamp": datetime.now().isoformat()
                    }
                ]
            },
            "categories": ["technical", "support"],
            "num_recommendations": 2
        }
    )
    assert response.status_code == 200
    assert "recommendations" in response.json()
    assert "confidence_scores" in response.json()