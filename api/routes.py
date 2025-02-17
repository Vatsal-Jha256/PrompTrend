from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Optional
from core.models import (
    RecommendationRequest,
    RecommendationResponse,
    TrainingRequest,
    TrainingResponse,
    IntentClassifierConfig,
    FeedbackRequest,
    FeedbackResponse
)
from services.intent_classifier import IntentClassifier
from services.recommender import ContextualBandit
from services.recommendation_service import RecommendationService
from sqlalchemy.orm import Session
from core.database import get_db
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize services
intent_classifier = None
bandit = ContextualBandit()
recommendation_service = RecommendationService()

@router.post("/initialize", response_model=dict)
async def initialize_classifier(config: IntentClassifierConfig):
    """Initialize or load the intent classifier"""
    global intent_classifier
    try:
        logger.info(f"Initializing classifier with config: {config}")
        intent_classifier = IntentClassifier(
            model_path=config.model_path,
            num_labels=config.num_labels,
            use_pretrained=config.use_pretrained
        )
        return {"status": "success", "message": "Classifier initialized"}
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback", response_model=FeedbackResponse)
async def store_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db)
):
    """Store and process user feedback for recommendations"""
    try:
        logger.info(f"Storing feedback for user {request.user_id}, recommendation {request.recommendation_id}")
        await recommendation_service.store_feedback(
            db=db,
            user_id=request.user_id,
            recommendation_id=request.recommendation_id,
            feedback_score=request.feedback_score
        )
        return FeedbackResponse(status="success", message="Feedback stored successfully")
    except Exception as e:
        logger.error(f"Failed to store feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train", response_model=TrainingResponse)
async def train_classifier(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train the classifier with new data"""
    global intent_classifier
    
    if not request.training_data:
        raise HTTPException(status_code=400, detail="Training data cannot be empty")
    
    if intent_classifier is None:
        raise HTTPException(status_code=400, detail="Classifier not initialized")
    
    # Validate training data before queuing
    labels = [item.label for item in request.training_data]
    num_labels = intent_classifier.model.config.num_labels
    if any(label < 0 or label >= num_labels for label in labels):
        raise HTTPException(
            status_code=400,
            detail=f"Labels must be between 0 and {num_labels - 1}"
        )
    
    try:
        logger.info(f"Starting training with {len(request.training_data)} examples for {request.epochs} epochs")
        # Convert training data to the format expected by IntentClassifier
        formatted_training_data = [
            {"text": item.text, "label": item.label}
            for item in request.training_data
        ]
        
        background_tasks.add_task(
            intent_classifier.train,
            formatted_training_data,
            request.epochs
        )
        return TrainingResponse(status="success", message="Training started in background")
    except Exception as e:
        logger.error(f"Failed to train classifier: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest, db: Session = Depends(get_db)):
    """Get recommendations based on chat history and generate relevant questions"""
    global intent_classifier
    
    if not request.chat_history.messages:
        raise HTTPException(
            status_code=400, 
            detail="Chat history cannot be empty"
        )
    
    if intent_classifier is None:
        raise HTTPException(status_code=400, detail="Classifier not initialized")
        
    try:
        logger.info(f"Getting recommendations for user {request.user_id}")
        
        # Classify intents
        intents_and_scores = [
            intent_classifier.classify(msg.content)
            for msg in request.chat_history.messages
        ]
        
        intents, confidence_scores = zip(*intents_and_scores)
        
        # Get context vector
        context = bandit.get_context_vector(
            request.user_id,
            [msg.content for msg in request.chat_history.messages],
            list(confidence_scores)
        )
        
        # Use default categories if none provided
        categories = request.categories or ["general", "technical", "support"]
        
        # Get recommendations from service
        result = await recommendation_service.get_recommendations(
            db=db,
            user_id=request.user_id,
            chat_history=[msg.dict() for msg in request.chat_history.messages],
            categories=categories,
            num_recommendations=request.num_recommendations
        )
        
        # Generate questions using the question generator
        from services.question_generator import QuestionGenerator
        question_generator = QuestionGenerator()
        generated_questions = question_generator.generate_questions(
            result["recommendations"],
            [msg.content for msg in request.chat_history.messages],
            num_questions=request.num_recommendations
        )
        
        # Add recommendation_ids to result if not present
        if "recommendation_ids" not in result:
            result["recommendation_ids"] = []
        
        # Include generated questions in response
        result["generated_questions"] = generated_questions
        
        return RecommendationResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to get recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))