import numpy as np
from sklearn.linear_model import SGDRegressor
from typing import List, Dict

class ContextualBandit:
    def __init__(self):
        self.learners: Dict[str, Dict[str, SGDRegressor]] = {}
        
    def get_context_vector(self, chat_history: List[str], intent_scores: List[float]) -> np.ndarray:
        """Create context vector from chat history and intent scores"""
        # Simple context: average intent score and number of messages
        return np.array([
            np.mean(intent_scores),
            len(chat_history)
        ]).reshape(1, -1)
        
    def train(self, user_id: str, category: str, context: np.ndarray, reward: float):
        """Train bandit for a specific user and category"""
        if user_id not in self.learners:
            self.learners[user_id] = {}
            
        if category not in self.learners[user_id]:
            self.learners[user_id][category] = SGDRegressor(
                learning_rate='constant',
                eta0=0.01
            )
            
        learner = self.learners[user_id][category]
        learner.partial_fit(context, [reward])
        
    def predict(self, user_id: str, category: str, context: np.ndarray) -> float:
        """Predict reward for a category given context"""
        if user_id not in self.learners or category not in self.learners[user_id]:
            return 0.0
            
        learner = self.learners[user_id][category]
        return learner.predict(context)[0]