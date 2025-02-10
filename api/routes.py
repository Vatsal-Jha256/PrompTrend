from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional
from core.models import (
    RecommendationRequest,
    RecommendationResponse,
    TrainingRequest,
    TrainingResponse,
    IntentClassifierConfig
)
from services.intent_classifier import IntentClassifier
from services.recommender import ContextualBandit
import json

router = APIRouter()
intent_classifier = None
bandit = ContextualBandit()

@router.post("/initialize", response_model=dict)
async def initialize_classifier(config: IntentClassifierConfig):
    """Initialize or load the intent classifier"""
    global intent_classifier
    try:
        intent_classifier = IntentClassifier(
            model_path=config.model_path,
            num_labels=config.num_labels,
            use_pretrained=config.use_pretrained
        )
        return {"status": "success", "message": "Classifier initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train", response_model=TrainingResponse)
async def train_classifier(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train the intent classifier on new data"""
    if intent_classifier is None:
        raise HTTPException(status_code=400, detail="Classifier not initialized")
    
    try:
        # Add training task to background
        background_tasks.add_task(
            intent_classifier.train,
            request.training_data,
            request.epochs
        )
        return TrainingResponse(
            status="success",
            message="Training started in background"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get recommendations based on chat history"""
    if intent_classifier is None:
        raise HTTPException(status_code=400, detail="Classifier not initialized")
    
    try:
        # Classify intents
        intents_and_scores = [
            intent_classifier.classify(msg.content)
            for msg in request.chat_history.messages
        ]
        intents, confidence_scores = zip(*intents_and_scores)
        
        # Get context vector
        context = bandit.get_context_vector(
            [msg.content for msg in request.chat_history.messages],
            confidence_scores
        )
        
        # Use default categories if none provided
        categories = request.categories or ["general", "technical", "support"]
        
        # Get predictions
        predictions = {
            category: bandit.predict(request.user_id, category, context)
            for category in categories
        }
        
        # Sort and get top recommendations
        sorted_categories = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_recommendations = [
            cat for cat, _ in sorted_categories[:request.num_recommendations]
        ]
        
        return RecommendationResponse(
            recommendations=top_recommendations,
            confidence_scores=predictions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
