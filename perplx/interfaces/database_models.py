"""
Database models for AWS RDS PostgreSQL
"""

from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime


class UserProfile(BaseModel):
    """User profile model"""
    id: Optional[str] = None
    user_id: str
    email: Optional[str] = None
    name: Optional[str] = None
    preferences: Optional[Dict] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ConversationRecord(BaseModel):
    """Conversation record model"""
    id: Optional[str] = None
    session_id: str
    user_id: Optional[str] = None
    user_message: str
    bot_response: str
    recommendations: Optional[List[str]] = None
    alternative_suggestions: Optional[List[str]] = None
    query_intent: Optional[str] = None
    response_time_ms: Optional[int] = None
    user_rating: Optional[int] = None
    created_at: Optional[datetime] = None


class SessionAnalytics(BaseModel):
    """Session analytics model"""
    id: Optional[str] = None
    session_id: str
    user_id: Optional[str] = None
    total_messages: int = 0
    total_recommendations: int = 0
    session_duration_minutes: Optional[int] = None
    first_message_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


class UserFeedback(BaseModel):
    """User feedback model"""
    id: Optional[str] = None
    conversation_id: str
    user_id: Optional[str] = None
    rating: int
    feedback_text: Optional[str] = None
    created_at: Optional[datetime] = None

