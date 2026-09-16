from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# Get database URL from environment variable or use default
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/promptrend")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    preferences = Column(JSON, nullable=True)
    recommendations = relationship("Recommendation", back_populates="user")

class Recommendation(Base):
    __tablename__ = "recommendations"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    category = Column(String)
    recommendation = Column(String)
    confidence_score = Column(Float)
    context = Column(JSON, nullable=True)  # Added context field
    timestamp = Column(DateTime, default=datetime.utcnow)
    feedback = Column(Integer, nullable=True)
    user = relationship("User", back_populates="recommendations")

class ChatHistory(Base):
    __tablename__ = "chat_histories"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    messages = Column(JSON)  # Store chat messages as JSON
    timestamp = Column(DateTime, default=datetime.utcnow)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)
