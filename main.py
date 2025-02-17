from fastapi import FastAPI
from api.routes import router as api_router  # Note the import alias
from core.config import get_settings
from core.database import init_db
import logging
import os

# Configure logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get app settings
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title="PrompTrend API",
    description="Intent Classification and Contextual Bandit Recommendation System",
    version="1.0.0",
)

# Setup middleware for exception handling
from fastapi import Request
from fastapi.responses import JSONResponse
from services.error_handler import PrompTrendError, handle_error

@app.middleware("http")
async def handle_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"detail": str(e)}
        )
    except PrompTrendError as e:
        return handle_error(e)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

# Include API router with proper prefix
app.include_router(api_router, prefix=settings.API_V1_STR)

# Initialize database
@app.on_event("startup")
def startup_event():
    try:
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)