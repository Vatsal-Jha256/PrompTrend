from fastapi import FastAPI
from api.routes import router
from core.models import IntentClassifierConfig
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
#TODO: create comprehensive documentation
#TODO: add few more tests and check for coverage
#TODO: add caching and databasing
#TODO: add more error handling
#TODO: generalized question generation add
app = FastAPI(
    title="PrompTrend API",
    description="Intent Classification and Contextual Bandit Recommendation System",
    version="1.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)