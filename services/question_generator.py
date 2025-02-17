from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from typing import List, Optional
from functools import lru_cache

#TODO: Simple LLM based question generation for now - add option to just retrieve cached mechanisms or templates if user wants faster response


class QuestionGenerator:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """Initialize the question generator with a T5 model"""
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    @lru_cache(maxsize=1000)
    def generate_question(self, topic: str, context: Optional[str] = None) -> str:
        """Generate a user-perspective question about a topic, considering context"""
        # Prepare input text with user-centric framing
        if context:
            input_text = f"generate a question that a user would ask an AI assistant about {topic}. Consider this context: {context}"
        else:
            input_text = f"generate a question that a user would ask an AI assistant about {topic}"
            
        # Tokenize and generate
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=64,
            num_beams=4,
            no_repeat_ngram_size=2,
            temperature=0.7,
            do_sample=True  # Enable sampling for more natural questions
        )
        
        question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Ensure the question ends with a question mark
        if not question.endswith('?'):
            question += '?'
            
        return question

    def generate_questions(
        self,
        recommendations: List[str],
        chat_history: Optional[List[str]] = None,
        num_questions: int = 3
    ) -> List[str]:
        """Generate multiple user-perspective questions based on recommendations"""
        questions = []
        
        # Use the last few messages as context if available
        context = None
        if chat_history and len(chat_history) > 0:
            context = " ".join(chat_history[-3:])  # Use last 3 messages as context
            
        for rec in recommendations[:num_questions]:
            question = self.generate_question(rec, context)
            questions.append(question)
            
        return questions

# Example usage and testing
def test_question_generator():
    generator = QuestionGenerator()
    
    # Test with technical recommendations
    recommendations = [
        "API authentication",
        "error handling",
        "response formatting"
    ]
    
    # Test with user context
    chat_history = [
        "I'm having trouble with the API",
        "The authentication keeps failing",
        "I've tried using different tokens"
    ]
    
    questions_with_context = generator.generate_questions(recommendations, chat_history)
    print("\nUser-centric questions with context:")
    for q in questions_with_context:
        print(f"- {q}")

if __name__ == "__main__":
    test_question_generator()