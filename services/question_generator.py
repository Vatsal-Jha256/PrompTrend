#TODO: Use llms or slms to generate questions
#TODO: : Questions should be from user perspective

class QuestionGenerator:
    def __init__(self):
        self.templates = {
            "general": "How can I help you with {topic}?",
            "problem": "Are you experiencing any issues with {topic}?",
            "usage": "Would you like to learn more about using {topic}?"
        }
        
    def generate(self, recommendations):
        questions = []
        for rec in recommendations:
            template = self.templates["general"]  # Can be made more sophisticated
            question = template.format(topic=rec)
            questions.append(question)
        return questions