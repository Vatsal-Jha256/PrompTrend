# services/recommendation_service.py
from core.database import get_db, User, Recommendation, ChatHistory
from core.cache import RedisCache, cached
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from services.intent_classifier import IntentClassifier
from services.recommender import ContextualBandit
import numpy as np
from fastapi import HTTPException

class RecommendationService:
    def __init__(self):
        self.cache = RedisCache()
        self.intent_classifier = IntentClassifier()
        self.bandit = ContextualBandit()

    async def ensure_user_exists(self, db: Session, user_id: str):
        """Ensure user exists in database, create if not"""
        user = db.query(User).filter_by(id=user_id).first()
        if not user:
            user = User(id=user_id)
            db.add(user)
            try:
                db.commit()
            except Exception as e:
                db.rollback()
                raise HTTPException(
                    status_code=500,
                    detail=f"Error creating user: {str(e)}"
                )
        return user

    async def store_chat_history(self, db: Session, user_id: str, messages: List[Dict]):
        """Store chat history with user context"""
        try:
            # Ensure user exists before storing chat history
            await self.ensure_user_exists(db, user_id)
            
            db.add(ChatHistory(
                user_id=user_id,
                messages=messages
            ))
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Error storing chat history: {str(e)}"
            )

    @cached(ttl=3600)
    async def get_recommendations(self, db: Session, user_id: str, 
                                chat_history: List[Dict], categories: List[str],
                                num_recommendations: int) -> Dict:
        """Get personalized recommendations"""
        try:
            # Ensure user exists
            await self.ensure_user_exists(db, user_id)
            
            # Store user context
            await self.store_chat_history(db, user_id, chat_history)
            
            # Generate user-aware context
            contents = [msg["content"] for msg in chat_history]
            intents, scores = zip(*[self.intent_classifier.classify(c) for c in contents])
            context = self.bandit.get_context_vector(user_id, contents, scores)
            
            # Get predictions with user context
            predictions = {
                cat: self.bandit.predict(user_id, cat, context)
                for cat in categories
            }
            
            # Store recommendations with context
            recommendation_ids = []
            for cat, score in predictions.items():
                rec = Recommendation(
                    user_id=user_id,
                    category=cat,
                    confidence_score=score,
                    context=context.tolist(),
                    recommendation=cat
                )
                db.add(rec)
                db.flush()
                recommendation_ids.append(rec.id)
            db.commit()
            
            sorted_categories = sorted(
                predictions.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            top_recommendations = [
                category for category, _ in sorted_categories[:num_recommendations]
            ]
            
            return {
                "recommendations": top_recommendations,
                "confidence_scores": predictions,
                "recommendation_ids": recommendation_ids
            }
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Error generating recommendations: {str(e)}"
            )

    async def store_feedback(self, db: Session, user_id: str, 
                           recommendation_id: int, feedback_score: int):
        """Handle feedback with user context"""
        try:
            # Ensure user exists
            await self.ensure_user_exists(db, user_id)
            
            # Get the recommendation
            rec = db.query(Recommendation).filter_by(
                id=recommendation_id,
                user_id=user_id
            ).first()
            
            if not rec:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Recommendation {recommendation_id} not found for user {user_id}"
                )

            # Update the bandit with feedback
            context = np.array(rec.context)
            self.bandit.train(
                user_id=user_id,
                category=rec.category,
                context=context,
                reward=feedback_score / 5.0
            )
            
            # Store feedback in database
            rec.feedback = feedback_score
            db.commit()
            
            # Invalidate cache for this user
            self.cache.delete(f"get_recommendations:{user_id}")
            
        except HTTPException:
            db.rollback()
            raise
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Error processing feedback: {str(e)}"
            )