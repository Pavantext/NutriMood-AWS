"""
Database Service - AWS RDS PostgreSQL operations for storing conversations and analytics
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Optional, Dict
import json
from datetime import datetime
import os
from interfaces.database_models import (
    UserProfile,
    ConversationRecord,
    SessionAnalytics,
    UserFeedback
)


class DatabaseService:
    """Service class for AWS RDS PostgreSQL operations"""
    
    def __init__(self):
        """Initialize database service with connection parameters"""
        self.connection_params = {
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }
        
        # Check if DB is configured
        if not all([self.connection_params['host'], self.connection_params['database'], 
                    self.connection_params['user'], self.connection_params['password']]):
            print("⚠️  Database not configured, running without persistent storage")
            self.enabled = False
            return
        
        self.enabled = True
        try:
            self._ensure_tables_exist()
            print("✅ Database service initialized (AWS RDS PostgreSQL)")
        except Exception as e:
            print(f"⚠️  Database connection failed: {e}")
            self.enabled = False
    
    def _get_connection(self):
        """Get database connection"""
        if not self.enabled:
            return None
        
        try:
            return psycopg2.connect(**self.connection_params)
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            return None
    
    def _ensure_tables_exist(self):
        """Ensure required tables exist in RDS PostgreSQL"""
        conn = self._get_connection()
        if not conn:
            return
        
        try:
            cursor = conn.cursor()
            
            # Tables are already created, just verify
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            required_tables = ['user_profiles', 'conversations', 'session_analytics', 'user_feedback']
            
            for table in required_tables:
                if table in tables:
                    print(f"   ✓ Table '{table}' exists")
                else:
                    print(f"   ⚠️  Table '{table}' not found")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error checking tables: {e}")
            if conn:
                conn.close()
            raise
    
    # Conversation Operations
    def save_conversation(
        self,
        session_id: str,
        user_id: Optional[str],
        user_message: str,
        bot_response: str,
        recommendations: Optional[List[str]] = None,
        query_intent: Optional[str] = None,
        response_time_ms: Optional[int] = None
    ) -> bool:
        """
        Save conversation record to database
        
        Args:
            session_id: Session identifier
            user_id: User identifier (optional)
            user_message: User's message
            bot_response: Bot's response
            recommendations: List of recommended food IDs
            query_intent: Detected intent of query
            response_time_ms: Response time in milliseconds
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        conn = self._get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            insert_sql = """
                INSERT INTO conversations (session_id, user_id, user_message, bot_response, 
                                        recommendations, query_intent, response_time_ms, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """
            
            cursor.execute(insert_sql, (
                session_id,
                user_id,
                user_message,
                bot_response,
                json.dumps(recommendations) if recommendations else None,
                query_intent,
                response_time_ms,
                datetime.utcnow()
            ))
            
            result = cursor.fetchone()
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return result is not None
            
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")
            if conn:
                conn.close()
            return False
    
    def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get conversation history for a session"""
        if not self.enabled:
            return []
        
        conn = self._get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM conversations 
                WHERE session_id = %s 
                ORDER BY created_at ASC 
                LIMIT %s
            """, (session_id, limit))
            
            results = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            print(f"❌ Error getting conversation history: {e}")
            if conn:
                conn.close()
            return []
    
    # Session Analytics Operations
    def update_session_analytics(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        total_messages: int = 0,
        total_recommendations: int = 0,
        session_duration_minutes: Optional[int] = None,
        first_message_at: Optional[datetime] = None,
        last_message_at: Optional[datetime] = None
    ) -> bool:
        """Update session analytics"""
        if not self.enabled:
            return False
        
        conn = self._get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Use UPSERT (INSERT ... ON CONFLICT)
            upsert_sql = """
                INSERT INTO session_analytics (session_id, user_id, total_messages, total_recommendations,
                                            session_duration_minutes, first_message_at, last_message_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id) 
                DO UPDATE SET 
                    user_id = EXCLUDED.user_id,
                    total_messages = EXCLUDED.total_messages,
                    total_recommendations = EXCLUDED.total_recommendations,
                    session_duration_minutes = EXCLUDED.session_duration_minutes,
                    first_message_at = EXCLUDED.first_message_at,
                    last_message_at = EXCLUDED.last_message_at
                RETURNING *;
            """
            
            cursor.execute(upsert_sql, (
                session_id,
                user_id,
                total_messages,
                total_recommendations,
                session_duration_minutes,
                first_message_at,
                last_message_at
            ))
            
            result = cursor.fetchone()
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return result is not None
            
        except Exception as e:
            print(f"❌ Error updating session analytics: {e}")
            if conn:
                conn.close()
            return False
    
    def get_session_analytics(self, session_id: str) -> Optional[Dict]:
        """Get session analytics"""
        if not self.enabled:
            return None
        
        conn = self._get_connection()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("SELECT * FROM session_analytics WHERE session_id = %s", (session_id,))
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return dict(result) if result else None
            
        except Exception as e:
            print(f"❌ Error getting session analytics: {e}")
            if conn:
                conn.close()
            return None
    
    # User Profile Operations
    def create_or_update_user_profile(
        self,
        user_id: str,
        email: Optional[str] = None,
        name: Optional[str] = None,
        preferences: Optional[Dict] = None
    ) -> bool:
        """Create or update user profile"""
        if not self.enabled:
            return False
        
        conn = self._get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Try to update first
            update_sql = """
                UPDATE user_profiles 
                SET email = %s, name = %s, preferences = %s, updated_at = %s
                WHERE user_id = %s
                RETURNING *;
            """
            
            cursor.execute(update_sql, (
                email,
                name,
                json.dumps(preferences) if preferences else None,
                datetime.utcnow(),
                user_id
            ))
            
            result = cursor.fetchone()
            
            if result:
                conn.commit()
                cursor.close()
                conn.close()
                return True
            
            # If no existing record, insert new one
            insert_sql = """
                INSERT INTO user_profiles (user_id, email, name, preferences, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING *;
            """
            
            cursor.execute(insert_sql, (
                user_id,
                email,
                name,
                json.dumps(preferences) if preferences else None,
                datetime.utcnow(),
                datetime.utcnow()
            ))
            
            result = cursor.fetchone()
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return result is not None
            
        except Exception as e:
            print(f"❌ Error creating/updating user profile: {e}")
            if conn:
                conn.close()
            return False
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile by user_id"""
        if not self.enabled:
            return None
        
        conn = self._get_connection()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("SELECT * FROM user_profiles WHERE user_id = %s", (user_id,))
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return dict(result) if result else None
            
        except Exception as e:
            print(f"❌ Error getting user profile: {e}")
            if conn:
                conn.close()
            return None
    
    # User Feedback Operations
    def save_user_feedback(
        self,
        conversation_id: str,
        user_id: Optional[str],
        rating: int,
        feedback_text: Optional[str] = None
    ) -> bool:
        """Save user feedback"""
        if not self.enabled:
            return False
        
        conn = self._get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            insert_sql = """
                INSERT INTO user_feedback (conversation_id, user_id, rating, feedback_text, created_at)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING *;
            """
            
            cursor.execute(insert_sql, (
                conversation_id,
                user_id,
                rating,
                feedback_text,
                datetime.utcnow()
            ))
            
            result = cursor.fetchone()
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return result is not None
            
        except Exception as e:
            print(f"❌ Error saving user feedback: {e}")
            if conn:
                conn.close()
            return False
    
    def get_feedback_stats(self, user_id: Optional[str] = None) -> Dict:
        """Get feedback statistics"""
        if not self.enabled:
            return {"total_feedback": 0, "average_rating": 0}
        
        conn = self._get_connection()
        if not conn:
            return {"total_feedback": 0, "average_rating": 0}
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if user_id:
                cursor.execute("SELECT rating FROM user_feedback WHERE user_id = %s", (user_id,))
            else:
                cursor.execute("SELECT rating FROM user_feedback")
            
            results = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            if not results:
                return {"total_feedback": 0, "average_rating": 0}
            
            ratings = [row['rating'] for row in results]
            total_feedback = len(ratings)
            average_rating = sum(ratings) / total_feedback if total_feedback > 0 else 0
            
            return {
                "total_feedback": total_feedback,
                "average_rating": round(average_rating, 2),
                "ratings_distribution": {
                    "5_stars": ratings.count(5),
                    "4_stars": ratings.count(4),
                    "3_stars": ratings.count(3),
                    "2_stars": ratings.count(2),
                    "1_star": ratings.count(1)
                }
            }
            
        except Exception as e:
            print(f"❌ Error getting feedback stats: {e}")
            if conn:
                conn.close()
            return {"total_feedback": 0, "average_rating": 0}
    
    def get_user_conversations(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Get all conversations for a user"""
        if not self.enabled:
            return []
        
        conn = self._get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM conversations 
                WHERE user_id = %s 
                ORDER BY created_at DESC 
                LIMIT %s
            """, (user_id, limit))
            
            results = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            print(f"❌ Error getting user conversations: {e}")
            if conn:
                conn.close()
            return []

