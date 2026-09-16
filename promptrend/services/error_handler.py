# services/error_handler.py
from typing import Optional, Dict, Any, Union
from fastapi import HTTPException
from enum import Enum
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorCode(Enum):
    """Standardized error codes for the application"""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    CACHE_ERROR = "CACHE_ERROR"
    AUTHENTICATION_ERROR = "AUTH_ERROR"
    AUTHORIZATION_ERROR = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    RATE_LIMIT_ERROR = "RATE_LIMIT"
    INTERNAL_ERROR = "INTERNAL_ERROR"

class PrompTrendError(Exception):
    """Base error class for PrompTrend with enhanced error tracking"""
    def __init__(
        self, 
        message: str,
        error_code: Union[ErrorCode, str],
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code if isinstance(error_code, str) else error_code.value
        self.status_code = status_code
        self.details = details or {}
        self.traceback = traceback.format_exc()
        super().__init__(self.message)

class ModelError(PrompTrendError):
    """Errors related to ML model operations"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.MODEL_ERROR,
            status_code=500,
            details=details
        )

class ValidationError(PrompTrendError):
    """Errors related to input validation"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=400,
            details=details
        )

class DatabaseError(PrompTrendError):
    """Errors related to database operations"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.DATABASE_ERROR,
            status_code=500,
            details=details
        )

def handle_error(error: Exception) -> HTTPException:
    """Enhanced error handler with logging and error tracking"""
    if isinstance(error, PrompTrendError):
        logger.error(
            f"PrompTrend error: {error.error_code} - {error.message}\n"
            f"Details: {error.details}\n"
            f"Traceback: {error.traceback}"
        )
        return HTTPException(
            status_code=error.status_code,
            detail={
                "error_code": error.error_code,
                "message": error.message,
                "details": error.details
            }
        )

    # Handle unexpected errors
    logger.error(
        f"Unexpected error: {str(error)}\n"
        f"Traceback: {traceback.format_exc()}"
    )
    return HTTPException(
        status_code=500,
        detail={
            "error_code": ErrorCode.INTERNAL_ERROR.value,
            "message": "An unexpected error occurred",
            "details": {
                "error": str(error),
                "type": type(error).__name__
            }
        }
    )