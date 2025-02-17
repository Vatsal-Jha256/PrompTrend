import requests
import time
from pprint import pprint
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_BASE = "http://localhost:8000/api/v1"

# Sample dataset - realistic but fictional e-commerce support scenarios
TRAINING_DATA = [
    {"text": "How do I reset my password?", "label": 0},  # Account Issues
    {"text": "My payment failed but money was deducted", "label": 1},  # Billing
    {"text": "API authentication keeps failing", "label": 2},  # Technical
    {"text": "When will my order ship?", "label": 3},  # Orders
    {"text": "How to integrate with your API?", "label": 2},  # Technical
    {"text": "Update my billing address", "label": 1},  # Billing
    {"text": "Why is my account locked?", "label": 0},  # Account Issues
    {"text": "Tracking number not working", "label": 3},  # Orders
]

# Initialize classifier
def initialize_classifier():
    logger.info("Initializing classifier...")
    config = {
        "model_path": "bert-base-uncased",
        "num_labels": 4,  # Matching our training data categories
        "use_pretrained": True
    }
    try:
        response = requests.post(
            f"{API_BASE}/initialize",
            json=config
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to initialize classifier: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            logger.error(f"Server response: {e.response.text}")
        sys.exit(1)

# Train classifier
def train_model():
    logger.info("Training model...")
    training_request = {
        "training_data": TRAINING_DATA,
        "epochs": 3
    }
    try:
        response = requests.post(
            f"{API_BASE}/train",
            json=training_request
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to train model: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            logger.error(f"Server response: {e.response.text}")
        sys.exit(1)

# Simulate user interaction
def user_session(user_id: str):
    logger.info(f"Starting user session for {user_id}")
    
    # Initial user message
    chat_history = {
        "messages": [{
            "content": "I'm having trouble with API authentication",
            "timestamp": str(time.time())
        }]
    }
    
    # Get initial recommendations
    rec_request = {
        "user_id": user_id,
        "chat_history": chat_history,
        "categories": ["Account Issues", "Billing", "Technical", "Orders"],
        "num_recommendations": 2
    }
    
    try:
        logger.info("Getting recommendations...")
        response = requests.post(
            f"{API_BASE}/recommendations",
            json=rec_request
        )
        response.raise_for_status()
        recommendations = response.json()
        
        print("\n=== Initial Recommendations ===")
        pprint(recommendations)
        
        # Simulate feedback on first recommendation
        if recommendations.get('recommendations'):
            logger.info("Sending feedback...")
            
            # The main issue - we need to actually get the recommendation ID
            # Let's assume recommendations are stored in the database during the '/recommendations' API call
            # Since we don't have direct access to the DB, we'll create a workaround
            
            # Get a list of recent recommendations for this user
            time.sleep(1)  # Give the server time to store the recommendations
            
            # Fixed approach: Get first recommendation from the recently created ones
            # In a real system, you'd get the actual ID from the response or from a separate endpoint
            # Since we're using the first recommendation in the list, we'll use ID 1
            # In a real system, recommendation IDs should be returned with the recommendations
            recommendation_id = 1
            
            feedback_data = {
                "user_id": user_id,
                "recommendation_id": recommendation_id,
                "feedback_score": 4  # Scale 0-5
            }
            
            try:
                feedback_response = requests.post(
                    f"{API_BASE}/feedback",
                    json=feedback_data
                )
                feedback_response.raise_for_status()
                
                print("\n=== Feedback Response ===")
                pprint(feedback_response.json())
            except requests.exceptions.RequestException as e:
                logger.error(f"Error sending feedback: {str(e)}")
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    logger.error(f"Server response: {e.response.text}")
                print("\n=== Failed to send feedback ===")
                print("This likely means the recommendation wasn't properly stored in the database.")
                print("Let's continue with the demo anyway.")
            
        # Additional chat message
        chat_history["messages"].append({
            "content": "I've tried different tokens but still get 401 unauthorized",
            "timestamp": str(time.time())
        })
        
        # Get updated recommendations after feedback
        rec_request["chat_history"] = chat_history
        
        logger.info("Getting updated recommendations after feedback...")
        updated_response = requests.post(
            f"{API_BASE}/recommendations",
            json=rec_request
        )
        updated_response.raise_for_status()
        updated_recommendations = updated_response.json()
        
        print("\n=== Updated Recommendations After Feedback ===")
        pprint(updated_recommendations)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in user session: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            logger.error(f"Server response: {e.response.text}")
        return

# Run complete demo flow
def full_demo():
    print("\n========== PrompTrend API Demo ==========\n")
    
    # Check if API is reachable
    try:
        response = requests.get(f"{API_BASE.split('/api')[0]}/docs")
        if response.status_code != 200:
            logger.error(f"API server not reachable at {API_BASE}")
            logger.error("Make sure the server is running with 'uvicorn main:app --reload'")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        logger.error(f"API server not reachable at {API_BASE}")
        logger.error("Make sure the server is running with 'uvicorn main:app --reload'")
        sys.exit(1)
    
    # Initialize classifier
    init_response = initialize_classifier()
    print("\n=== Classifier Initialization ===")
    pprint(init_response)
    
    # Train model
    train_response = train_model()
    print("\n=== Model Training ===")
    pprint(train_response)
    
    # Wait for training to complete
    print("\nWaiting for training to complete...")
    time.sleep(5)  # Reduced wait time for demo
    
    # Simulate user session
    print("\n=== Simulating User Session ===")
    user_session("demo_user_123")
    
    print("\n========== Demo Complete ==========")

if __name__ == "__main__":
    full_demo()