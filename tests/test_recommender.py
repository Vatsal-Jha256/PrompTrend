import pytest
from services.recommender import ContextualBandit
import numpy as np

def test_bandit_training():
    bandit = ContextualBandit()
    user_id = "test_user"
    category = "test_category"
    context = np.array([[0.5, 5]])  # Example context vector
    reward = 1.0
    
    # Train the bandit
    bandit.train(user_id, category, context, reward)
    
    # Test prediction
    prediction = bandit.predict(user_id, category, context)
    assert isinstance(prediction, float)
    assert 0 <= prediction <= 1