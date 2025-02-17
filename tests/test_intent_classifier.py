import pytest
from services.intent_classifier import IntentClassifier

def test_intent_classification():
    classifier = IntentClassifier()
    text = "How do I fix my device?"
    
    intent, confidence = classifier.classify(text)
    
    assert isinstance(intent, int)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1