# scripts/setup_env.py
import os
import argparse
from pathlib import Path

def create_env_file(environment: str = "development"):
    """Create .env file with appropriate settings"""
    # Get project root directory
    root_dir = Path(__file__).parent.parent
    env_file = root_dir / ".env"
    
    # Default configurations for different environments
    configs = {
        "development": {
            "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/promptrend",
            "REDIS_URL": "redis://localhost:6379/0",
            "MODEL_PATH": "bert-base-uncased",
            "ENVIRONMENT": "development",
            "LOG_LEVEL": "DEBUG"
        },
        "production": {
            "DATABASE_URL": "postgresql://user:password@prod-db:5432/promptrend",
            "REDIS_URL": "redis://prod-redis:6379/0",
            "MODEL_PATH": "bert-base-uncased",
            "ENVIRONMENT": "production",
            "LOG_LEVEL": "INFO"
        },
        "testing": {
            "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/promptrend_test",
            "REDIS_URL": "redis://localhost:6379/1",
            "MODEL_PATH": "bert-base-uncased",
            "ENVIRONMENT": "testing",
            "LOG_LEVEL": "DEBUG"
        }
    }
    
    # Get configuration for specified environment
    config = configs.get(environment)
    if not config:
        raise ValueError(f"Invalid environment: {environment}")
    
    # Create .env file
    with open(env_file, "w") as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")
    
    print(f"Created .env file for {environment} environment at {env_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create environment configuration")
    parser.add_argument(
        "--env",
        choices=["development", "production", "testing"],
        default="development",
        help="Environment to configure"
    )
    args = parser.parse_args()
    create_env_file(args.env)