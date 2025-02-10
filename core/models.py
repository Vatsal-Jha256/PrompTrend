from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from pydantic.config import ConfigDict
class ChatMessage(BaseModel):
    content: str
    timestamp: str

class ChatHistory(BaseModel):
    messages: List[ChatMessage]

class TrainingData(BaseModel):
    text: str
    label: int

class IntentClassifierConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_path: str = "bert-base-uncased"
    num_labels: int = 10
    use_pretrained: bool = True

class TrainingRequest(BaseModel):
    training_data: List[TrainingData]
    epochs: int = Field(default=3, ge=1, le=10)

class TrainingResponse(BaseModel):
    status: str
    message: str

class RecommendationRequest(BaseModel):
    user_id: str
    chat_history: ChatHistory
    categories: Optional[List[str]] = None
    num_recommendations: int = Field(default=3, ge=1, le=10)

class RecommendationResponse(BaseModel):
    recommendations: List[str]
    confidence_scores: Dict[str, float]