# PrompTrend: Intelligent Chat Support System

[![PyPI version](https://badge.fury.io/py/promptrend.svg)](https://badge.fury.io/py/promptrend)

PrompTrend is an advanced chat support system that combines intent classification and contextual bandit algorithms to provide personalized recommendations and responses. The system uses BERT for intent classification and implements a contextual multi-armed bandit approach for dynamic learning from user interactions.

## 🌟 Key Features

- Intent classification using BERT
- Contextual bandit-based recommendation system
- Real-time user feedback processing
- Automatic question generation
- Redis caching for improved performance
- Comprehensive API documentation
- Robust error handling
- Database persistence with PostgreSQL

## 🛠️ Technology Stack

- **Framework**: FastAPI
- **ML Models**: BERT (Transformers), T5
- **Database**: PostgreSQL
- **Caching**: Redis
- **ML Libraries**: PyTorch, Scikit-learn
- **Testing**: Pytest

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL
- Redis
- CUDA-compatible GPU (optional, for faster model training)

## 💻 Usage as a Library

You can now easily integrate PrompTrend's AI components directly into your own applications:

```python
from promptrend import IntentClassifier, Recommender, QuestionGenerator

# Initialize components
classifier = IntentClassifier(model_name="bert-base-uncased")
recommender = Recommender()
generator = QuestionGenerator()

# Example: Get intent
text = "How do I reset my password?"
intent = classifier.predict(text)

# Example: Get recommendation
context = recommender.get_context_vector(user_id="123", chat_history=[text], intent_scores=[0.9])
score = recommender.predict(user_id="123", category="account_help", context=context)
```


## ⚙️ Installation

You can install PrompTrend directly via pip (once published):
```bash
pip install promptrend
```

For development, clone the repository and install in editable mode:
```bash
git clone https://github.com/Vatsal-Jha256/PrompTrend.git
cd promptrend
pip install -e .[test]
```

Set up environment variables:
```bash
python scripts/setup_env.py --env development
```

## 🚀 Running the Application

1. Start the Redis server:
```bash
redis-server
```

2. Start the application using the CLI:
```bash
promptrend-server
```

The API will be available at `http://localhost:8000`


## 🧪 Testing

The project includes comprehensive tests for all components. To run the tests:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=app tests/

# Run specific test file
pytest tests/test_intent_classifier.py
```

## 📂 Project Structure

```
promptrend/
├── api/
│   └── routes.py          # API endpoints
├── core/
│   ├── config.py          # Configuration management
│   ├── database.py        # Database setup
│   ├── models.py          # Data models
│   └── cache.py           # Redis cache implementation
├── services/
│   ├── intent_classifier.py    # BERT classifier
│   ├── recommender.py          # Contextual bandit
│   ├── recommendation_service.py
│   ├── question_generator.py   # T5 question generator
│   └── error_handler.py        # Error handling service
├── tests/
│   ├── test_intent_classifier.py
│   ├── test_recommender.py
│   ├── test_api.py
│   └── test_integration.py 
├── scripts/
│   └── setup_env.py
├── main.py
├── requirements.txt
└── README.md
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- FastAPI team for the amazing framework
- The open-source community for various dependencies

## 📞 Contact

For questions and feedback, please create an issue in the GitHub repository.
