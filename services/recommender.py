# services/recommender.py
import numpy as np
from sklearn.linear_model import SGDRegressor
from typing import List, Dict
import hashlib
class ContextualBandit:
    def __init__(self):
        self.learners: Dict[str, Dict[str, SGDRegressor]] = {}
        
    def get_context_vector(self, user_id: str, chat_history: List[str], intent_scores: List[float]) -> np.ndarray:
        """
        Create a distinctive user context vector while ensuring consistent dimensions
        """
        # Create deterministic but unique user features
        user_hash = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
        
        # Basic user features (always length 3)
        user_features = np.array([
            (user_hash % 100) / 100,
            ((user_hash // 100) % 100) / 100,
            ((user_hash // 10000) % 100) / 100
        ])
        
        # Context features (always length 4)
        context_features = np.array([
            float(np.mean(intent_scores)) if intent_scores else 0.0,
            float(len(chat_history)),
            float(np.std(intent_scores)) if intent_scores else 0.0,
            float(max(intent_scores)) if intent_scores else 0.0
        ])
        
        # Combine and ensure proper shape
        combined = np.concatenate([user_features, context_features])
        return combined.reshape(1, -1)
    
    def train(self, user_id: str, category: str, context: np.ndarray, reward: float):
        """Train bandit with proper initialization and type handling"""
        if user_id not in self.learners:
            self.learners[user_id] = {}
            
        if category not in self.learners[user_id]:
            # Initialize model with user-specific seed
            user_specific_seed = hash(user_id) % 2**32
            
            self.learners[user_id][category] = SGDRegressor(
                learning_rate='constant',
                eta0=0.01,
                random_state=user_specific_seed
            )
            
            # Initialize with some random data for better starting point
            n_features = context.shape[1]
            np.random.seed(user_specific_seed)
            init_contexts = np.random.randn(5, n_features)
            init_rewards = np.random.rand(5)
            self.learners[user_id][category].partial_fit(init_contexts, init_rewards)
        
        # Ensure context and reward are the right type
        context_array = np.asarray(context, dtype=np.float64)
        reward_value = float(reward)
        
        # Train the model
        self.learners[user_id][category].partial_fit(context_array, [reward_value])
    
    def predict(self, user_id: str, category: str, context: np.ndarray) -> float:
        """Predict with proper type handling and exploration"""
        if user_id not in self.learners or category not in self.learners[user_id]:
            # Deterministic but different initialization for new users/categories
            np.random.seed(hash(user_id + category) % 2**32)
            base_score = float(np.random.random() * 0.5 + 0.25)  # Between 0.25 and 0.75
            return base_score
        
        # Ensure context is the right type
        context_array = np.asarray(context, dtype=np.float64)
        
        # Get base prediction
        prediction = self.learners[user_id][category].predict(context_array)
        
        # Add small random variation to encourage exploration
        np.random.seed(hash(user_id + category + str(context.sum())) % 2**32)
        noise = np.random.normal(0, 0.1)
        
        # Ensure result is a float between 0 and 1
        final_prediction = float(np.clip(prediction[0] + noise, 0, 1))
        return final_prediction