import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Optional, Dict
import json
from datetime import datetime, timezone, timedelta
import os
from interfaces.database_models import (
    UserProfile,
    ConversationRecord,
    SessionAnalytics,
    UserFeedback
)


class DatabaseService:
    """Service class for AWS RDS PostgreSQL operations"""
    
    # IST timezone offset: UTC+5:30
    IST = timezone(timedelta(hours=5, minutes=30))
    
    @staticmethod
    def _to_ist(dt: datetime) -> datetime:
        """Convert UTC datetime to IST (Indian Standard Time)"""
        if dt is None:
            return None
        # If datetime is naive (no timezone info), assume it's UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # Convert to IST
        ist_dt = dt.astimezone(DatabaseService.IST)
        return ist_dt
    
    @staticmethod
    def _format_ist_datetime(dt: datetime) -> str:
        """Format datetime in IST timezone"""
        if dt is None:
            return 'N/A'
        ist_dt = DatabaseService._to_ist(dt)
        return ist_dt.strftime('%Y-%m-%d %H:%M:%S IST')
    
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
    
    def _get_connection(self, autocommit=False):
        """Get database connection"""
        if not self.enabled:
            return None
        
        try:
            conn = psycopg2.connect(**self.connection_params)
            if autocommit:
                conn.autocommit = True
            return conn
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
            
            # Create chatbot_sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chatbot_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    total_time_seconds INTEGER,
                    user_name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create indexes for chatbot_sessions
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chatbot_sessions_session_id 
                ON chatbot_sessions(session_id);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chatbot_sessions_start_time 
                ON chatbot_sessions(start_time);
            """)
            
            # Create chatbot_food_orders table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chatbot_food_orders (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    product_id VARCHAR(255) NOT NULL,
                    product_name VARCHAR(255) NOT NULL,
                    event_type VARCHAR(50) NOT NULL,
                    order_id VARCHAR(255),
                    quantity INTEGER,
                    timestamp TIMESTAMP NOT NULL,
                    user_name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create indexes for chatbot_food_orders
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chatbot_food_orders_session_id 
                ON chatbot_food_orders(session_id);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chatbot_food_orders_event_type 
                ON chatbot_food_orders(event_type);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chatbot_food_orders_order_id 
                ON chatbot_food_orders(order_id);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chatbot_food_orders_timestamp 
                ON chatbot_food_orders(timestamp);
            """)
            
            # Create chatbot_ratings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chatbot_ratings (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    message_id VARCHAR(255),
                    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                    timestamp TIMESTAMP NOT NULL,
                    user_name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Add message_id column if it doesn't exist (for existing tables)
            cursor.execute("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name='chatbot_ratings' AND column_name='message_id'
                    ) THEN
                        ALTER TABLE chatbot_ratings ADD COLUMN message_id VARCHAR(255);
                    END IF;
                END $$;
            """)
            
            # Create indexes for chatbot_ratings
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chatbot_ratings_session_id 
                ON chatbot_ratings(session_id);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chatbot_ratings_message_id 
                ON chatbot_ratings(message_id);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chatbot_ratings_rating 
                ON chatbot_ratings(rating);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chatbot_ratings_timestamp 
                ON chatbot_ratings(timestamp);
            """)
            
            conn.commit()
            
            # Verify tables exist
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            required_tables = ['user_profiles', 'conversations', 'session_analytics', 'user_feedback',
                             'chatbot_sessions', 'chatbot_food_orders', 'chatbot_ratings']
            
            for table in required_tables:
                if table in tables:
                    print(f"   ✓ Table '{table}' exists")
                else:
                    print(f"   ⚠️  Table '{table}' not found")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error checking/creating tables: {e}")
            if conn:
                conn.rollback()
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
    
    def get_all_users(self) -> Dict[str, Dict]:
        """
        Get all sessions with their aggregated data for admin dashboard (OPTIMIZED)
        Uses a single efficient query with JOINs instead of N+1 queries
        
        Returns:
            Dictionary mapping session_id to session data with conversations, analytics, etc.
        """
        if not self.enabled:
            return {}
        
        # Use autocommit mode to avoid transaction issues
        conn = self._get_connection(autocommit=True)
        if not conn:
            return {}
        
        sessions_data = {}
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Single optimized query to get all session data at once
            cursor.execute("""
                WITH session_stats AS (
                    SELECT 
                        c.session_id,
                        MIN(c.user_id) as user_id,
                        COUNT(*) as conversation_count,
                        MIN(c.created_at) as first_message_at,
                        MAX(c.created_at) as last_message_at,
                        -- Get all recommendations as an array for processing
                        ARRAY_AGG(c.recommendations) FILTER (WHERE c.recommendations IS NOT NULL) as all_recommendations
                    FROM conversations c
                    GROUP BY c.session_id
                )
                SELECT 
                    ss.session_id,
                    ss.user_id,
                    ss.conversation_count,
                    ss.first_message_at,
                    ss.last_message_at,
                    ss.all_recommendations,
                    up.name as user_name,
                    up.email as user_email,
                    sa.total_messages,
                    sa.total_recommendations
                FROM session_stats ss
                LEFT JOIN user_profiles up ON ss.user_id = up.user_id
                LEFT JOIN session_analytics sa ON ss.session_id = sa.session_id
                ORDER BY ss.last_message_at DESC
            """)
            
            rows = cursor.fetchall()
            cursor.close()
            
            # Process results
            for row in rows:
                session_id = row['session_id']
                user_id = row['user_id']
                
                # Count total recommendations from the array
                total_recs = 0
                if row.get('all_recommendations'):
                    for rec in row['all_recommendations']:
                        if rec:
                            try:
                                if isinstance(rec, str):
                                    if rec and rec != '' and rec != '[]':
                                        rec_list = json.loads(rec)
                                        if isinstance(rec_list, list):
                                            total_recs += len(rec_list)
                                        else:
                                            total_recs += 1
                                elif isinstance(rec, list):
                                    total_recs += len(rec)
                                else:
                                    total_recs += 1
                            except (json.JSONDecodeError, TypeError):
                                if rec and rec != '' and rec != '[]':
                                    total_recs += 1
                
                # Use analytics total if available, otherwise use our count
                if row.get('total_recommendations') is not None:
                    total_recs = row['total_recommendations']
                
                # Format first login time in IST
                first_login = 'N/A'
                if row.get('first_message_at'):
                    first_login = self._format_ist_datetime(row['first_message_at'])
                
                # Determine display name
                display_name = session_id
                if row.get('user_name'):
                    display_name = f"{row['user_name']} ({session_id[:8]}...)"
                elif row.get('user_email'):
                    display_name = f"{row['user_email']} ({session_id[:8]}...)"
                elif user_id:
                    display_name = f"User {user_id[:8]}... ({session_id[:8]}...)"
                
                # Build session data
                sessions_data[session_id] = {
                    'session_id': session_id,
                    'user_id': user_id,
                    'display_name': display_name,
                    'login_time': first_login,
                    'conversation_count': row.get('conversation_count', 0),
                    'total_recommendations': total_recs
                }
            
            conn.close()
            return sessions_data
            
        except Exception as e:
            print(f"❌ Error getting all sessions: {e}")
            if conn:
                try:
                    conn.rollback()
                    conn.close()
                except:
                    pass
            return {}
    
    def get_user_details(self, session_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session details including all conversations
        """
        if not self.enabled:
            return None
        
        conn = self._get_connection()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get session analytics
            cursor.execute("SELECT * FROM session_analytics WHERE session_id = %s LIMIT 1", (session_id,))
            analytics = cursor.fetchone()
            
            # Get user_id from conversations (if available)
            cursor.execute("""
                SELECT DISTINCT user_id 
                FROM conversations 
                WHERE session_id = %s AND user_id IS NOT NULL
                LIMIT 1
            """, (session_id,))
            user_result = cursor.fetchone()
            user_id = user_result['user_id'] if user_result and user_result.get('user_id') else None
            
            # Get user profile if user_id exists
            profile = None
            if user_id:
                cursor.execute("SELECT * FROM user_profiles WHERE user_id = %s", (user_id,))
                profile = cursor.fetchone()
            
            # Get all conversations for this session
            cursor.execute("""
                SELECT * FROM conversations 
                WHERE session_id = %s 
                ORDER BY created_at DESC
            """, (session_id,))
            
            conversations = cursor.fetchall()
            
            if not conversations:
                cursor.close()
                conn.close()
                return None
            
            # Get first login
            cursor.execute("""
                SELECT MIN(created_at) as first_login 
                FROM conversations 
                WHERE session_id = %s
            """, (session_id,))
            first_result = cursor.fetchone()
            first_login = None
            if first_result and first_result['first_login']:
                first_login = self._format_ist_datetime(first_result['first_login'])
            
            cursor.close()
            conn.close()
            
            # Format conversations - food details will be added by the calling code
            formatted_conversations = []
            for conv in conversations:
                recommendations = []
                if conv.get('recommendations'):
                    try:
                        rec_ids = json.loads(conv['recommendations']) if isinstance(conv['recommendations'], str) else conv['recommendations']
                    except:
                        rec_ids = []
                else:
                    rec_ids = []
                
                formatted_conv = {
                    'timestamp': self._format_ist_datetime(conv['created_at']) if conv.get('created_at') else 'N/A',
                    'user_input': conv.get('user_message', ''),
                    'ai_response': conv.get('bot_response', ''),
                    'recommended_food_ids': rec_ids if isinstance(rec_ids, list) else [],
                    'recommended_foods': [],  # Will be populated by caller with food details
                    'is_followup': False  # Can enhance this later
                }
                formatted_conversations.append(formatted_conv)
            
            # Determine display name
            display_name = session_id
            if profile:
                if profile.get('name'):
                    display_name = f"{profile['name']} ({session_id})"
                elif profile.get('email'):
                    display_name = f"{profile['email']} ({session_id})"
            elif user_id:
                display_name = f"User {user_id} ({session_id})"
            
            return {
                'session_id': session_id,
                'user_id': user_id,
                'display_name': display_name,
                'username': display_name,  # For template compatibility
                'login_time': first_login or 'N/A',
                'conversations': formatted_conversations,
                'profile': dict(profile) if profile else None
            }
            
        except Exception as e:
            print(f"❌ Error getting session details: {e}")
            if conn:
                conn.close()
            return None
    
    # Chatbot Tracking Operations
    def track_chatbot_session(
        self,
        session_id: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        total_time_seconds: Optional[int] = None,
        user_name: Optional[str] = None
    ) -> bool:
        """
        Track chatbot session time
        
        Args:
            session_id: Session identifier
            start_time: Session start time
            end_time: Session end time (optional, only when session ends)
            total_time_seconds: Total session duration in seconds (optional)
            user_name: User's name (optional)
        
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
            
            # If end_time is provided, update existing session or insert new one
            if end_time:
                # Try to update existing session
                update_sql = """
                    UPDATE chatbot_sessions 
                    SET end_time = %s, total_time_seconds = %s
                    WHERE session_id = %s AND end_time IS NULL
                    RETURNING id;
                """
                cursor.execute(update_sql, (end_time, total_time_seconds, session_id))
                result = cursor.fetchone()
                
                if not result:
                    # No existing session, insert new one
                    insert_sql = """
                        INSERT INTO chatbot_sessions (session_id, start_time, end_time, 
                                                    total_time_seconds, user_name)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id;
                    """
                    cursor.execute(insert_sql, (
                        session_id, start_time, end_time, total_time_seconds, user_name
                    ))
            else:
                # Session start - check if session already exists
                check_sql = """
                    SELECT id FROM chatbot_sessions 
                    WHERE session_id = %s AND end_time IS NULL
                """
                cursor.execute(check_sql, (session_id,))
                existing = cursor.fetchone()
                
                if not existing:
                    # Insert new session
                    insert_sql = """
                        INSERT INTO chatbot_sessions (session_id, start_time, user_name)
                        VALUES (%s, %s, %s)
                        RETURNING id;
                    """
                    cursor.execute(insert_sql, (session_id, start_time, user_name))
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error tracking chatbot session: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False
    
    def track_food_order(
        self,
        session_id: str,
        product_id: str,
        product_name: str,
        timestamp: datetime,
        event_type: str,
        user_name: Optional[str] = None,
        order_id: Optional[str] = None,
        quantity: Optional[int] = None
    ) -> bool:
        """
        Track food order events from chatbot
        
        Args:
            session_id: Session identifier
            product_id: Product/item ID
            product_name: Product name
            timestamp: Event timestamp
            event_type: Either 'added_to_cart' or 'order_placed'
            user_name: User's name (optional)
            order_id: Order ID (optional, only for 'order_placed')
            quantity: Quantity ordered (optional, only for 'order_placed')
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        if event_type not in ['added_to_cart', 'order_placed']:
            print(f"❌ Invalid event_type: {event_type}")
            return False
        
        conn = self._get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            insert_sql = """
                INSERT INTO chatbot_food_orders (session_id, product_id, product_name, 
                                                event_type, order_id, quantity, timestamp, user_name)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """
            
            cursor.execute(insert_sql, (
                session_id, product_id, product_name, event_type,
                order_id, quantity, timestamp, user_name
            ))
            
            result = cursor.fetchone()
            conn.commit()
            cursor.close()
            conn.close()
            
            return result is not None
            
        except Exception as e:
            print(f"❌ Error tracking food order: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False
    
    def track_chatbot_rating(
        self,
        session_id: str,
        rating: int,
        timestamp: datetime,
        message_id: Optional[str] = None,
        user_name: Optional[str] = None
    ) -> bool:
        """
        Track chatbot rating/feedback
        
        Args:
            session_id: Session identifier
            rating: Rating value (1-5)
            timestamp: Rating timestamp
            message_id: ID of the specific bot message being rated (optional)
            user_name: User's name (optional)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        if rating < 1 or rating > 5:
            print(f"❌ Invalid rating: {rating}. Must be between 1 and 5.")
            return False
        
        conn = self._get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # If message_id is provided, check if rating exists for this specific message
            # Otherwise, check if rating exists for this session (backward compatibility)
            if message_id:
                check_sql = """
                    SELECT id FROM chatbot_ratings 
                    WHERE session_id = %s AND message_id = %s
                """
                cursor.execute(check_sql, (session_id, message_id))
            else:
                check_sql = """
                    SELECT id FROM chatbot_ratings WHERE session_id = %s
                """
                cursor.execute(check_sql, (session_id,))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing rating
                if message_id:
                    update_sql = """
                        UPDATE chatbot_ratings 
                        SET rating = %s, timestamp = %s, user_name = %s, message_id = %s
                        WHERE session_id = %s AND message_id = %s
                        RETURNING id;
                    """
                    cursor.execute(update_sql, (rating, timestamp, user_name, message_id, session_id, message_id))
                else:
                    update_sql = """
                        UPDATE chatbot_ratings 
                        SET rating = %s, timestamp = %s, user_name = %s
                        WHERE session_id = %s
                        RETURNING id;
                    """
                    cursor.execute(update_sql, (rating, timestamp, user_name, session_id))
            else:
                # Insert new rating
                insert_sql = """
                    INSERT INTO chatbot_ratings (session_id, message_id, rating, timestamp, user_name)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id;
                """
                cursor.execute(insert_sql, (session_id, message_id, rating, timestamp, user_name))
            
            result = cursor.fetchone()
            conn.commit()
            cursor.close()
            conn.close()
            
            return result is not None
            
        except Exception as e:
            print(f"❌ Error tracking chatbot rating: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False
    
    def get_chatbot_analytics(self) -> Dict:
        """
        Get comprehensive chatbot analytics for admin dashboard
        
        Returns:
            Dictionary with analytics data
        """
        if not self.enabled:
            return {
                "total_sessions": 0,
                "total_ratings": 0,
                "average_rating": 0,
                "total_food_orders": 0,
                "total_added_to_cart": 0,
                "average_session_duration": 0
            }
        
        conn = self._get_connection()
        if not conn:
            return {
                "total_sessions": 0,
                "total_ratings": 0,
                "average_rating": 0,
                "total_food_orders": 0,
                "total_added_to_cart": 0,
                "average_session_duration": 0
            }
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get session statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_sessions,
                    COUNT(CASE WHEN end_time IS NOT NULL THEN 1 END) as completed_sessions,
                    AVG(total_time_seconds) as avg_duration_seconds
                FROM chatbot_sessions
            """)
            session_stats = cursor.fetchone()
            
            # Get rating statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_ratings,
                    AVG(rating) as avg_rating,
                    COUNT(CASE WHEN rating = 5 THEN 1 END) as rating_5,
                    COUNT(CASE WHEN rating = 4 THEN 1 END) as rating_4,
                    COUNT(CASE WHEN rating = 3 THEN 1 END) as rating_3,
                    COUNT(CASE WHEN rating = 2 THEN 1 END) as rating_2,
                    COUNT(CASE WHEN rating = 1 THEN 1 END) as rating_1
                FROM chatbot_ratings
            """)
            rating_stats = cursor.fetchone()
            
            # Get food order statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_events,
                    COUNT(CASE WHEN event_type = 'order_placed' THEN 1 END) as total_orders,
                    COUNT(CASE WHEN event_type = 'added_to_cart' THEN 1 END) as total_added_to_cart,
                    COUNT(DISTINCT product_id) as unique_products,
                    COUNT(DISTINCT session_id) as sessions_with_orders
                FROM chatbot_food_orders
            """)
            order_stats = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return {
                "total_sessions": session_stats['total_sessions'] or 0,
                "completed_sessions": session_stats['completed_sessions'] or 0,
                "average_session_duration": round(session_stats['avg_duration_seconds'] or 0, 2),
                "total_ratings": rating_stats['total_ratings'] or 0,
                "average_rating": round(rating_stats['avg_rating'] or 0, 2),
                "ratings_distribution": {
                    "5_stars": rating_stats['rating_5'] or 0,
                    "4_stars": rating_stats['rating_4'] or 0,
                    "3_stars": rating_stats['rating_3'] or 0,
                    "2_stars": rating_stats['rating_2'] or 0,
                    "1_star": rating_stats['rating_1'] or 0
                },
                "total_food_orders": order_stats['total_orders'] or 0,
                "total_added_to_cart": order_stats['total_added_to_cart'] or 0,
                "unique_products_ordered": order_stats['unique_products'] or 0,
                "sessions_with_orders": order_stats['sessions_with_orders'] or 0
            }
            
        except Exception as e:
            print(f"❌ Error getting chatbot analytics: {e}")
            if conn:
                conn.close()
            return {
                "total_sessions": 0,
                "total_ratings": 0,
                "average_rating": 0,
                "total_food_orders": 0,
                "total_added_to_cart": 0,
                "average_session_duration": 0
            }
    
    def get_orders_analytics(self) -> Dict:
        """Get detailed orders analytics (added to cart vs placed)"""
        if not self.enabled:
            return {
                "total_orders": 0,
                "total_added_to_cart": 0,
                "total_items_ordered": 0,
                "total_items_in_cart": 0,
                "conversion_rate": 0,
                "top_products": []
            }
        
        conn = self._get_connection()
        if not conn:
            return {
                "total_orders": 0,
                "total_added_to_cart": 0,
                "total_items_ordered": 0,
                "total_items_in_cart": 0,
                "conversion_rate": 0,
                "top_products": []
            }
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get order statistics
            cursor.execute("""
                SELECT 
                    COUNT(CASE WHEN event_type = 'order_placed' THEN 1 END) as total_orders,
                    COUNT(CASE WHEN event_type = 'added_to_cart' THEN 1 END) as total_added_to_cart,
                    SUM(CASE WHEN event_type = 'order_placed' AND quantity IS NOT NULL THEN quantity ELSE 0 END) as total_items_ordered,
                    SUM(CASE WHEN event_type = 'added_to_cart' THEN 1 ELSE 0 END) as total_items_in_cart
                FROM chatbot_food_orders
            """)
            stats = cursor.fetchone()
            
            # Get top products ordered
            cursor.execute("""
                SELECT 
                    product_id,
                    product_name,
                    COUNT(*) as order_count,
                    SUM(quantity) as total_quantity
                FROM chatbot_food_orders
                WHERE event_type = 'order_placed'
                GROUP BY product_id, product_name
                ORDER BY order_count DESC, total_quantity DESC
                LIMIT 10
            """)
            top_products = cursor.fetchall()
            
            total_added = stats['total_added_to_cart'] or 0
            total_ordered = stats['total_orders'] or 0
            conversion_rate = 0
            if total_added > 0:
                conversion_rate = round((total_ordered / total_added) * 100, 2)
            
            cursor.close()
            conn.close()
            
            return {
                "total_orders": total_ordered,
                "total_added_to_cart": total_added,
                "total_items_ordered": stats['total_items_ordered'] or 0,
                "total_items_in_cart": stats['total_items_in_cart'] or 0,
                "conversion_rate": conversion_rate,
                "top_products": [{
                    "product_id": row['product_id'],
                    "product_name": row['product_name'],
                    "order_count": row['order_count'],
                    "total_quantity": row['total_quantity'] or 0
                } for row in top_products]
            }
            
        except Exception as e:
            print(f"❌ Error getting orders analytics: {e}")
            if conn:
                conn.close()
            return {
                "total_orders": 0,
                "total_added_to_cart": 0,
                "total_items_ordered": 0,
                "total_items_in_cart": 0,
                "conversion_rate": 0,
                "top_products": []
            }
    
    def get_users_analytics(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict:
        """Get users analytics with optional date range"""
        if not self.enabled:
            return {
                "total_users": 0,
                "users_by_date": [],
                "new_users": 0,
                "active_users": 0
            }
        
        conn = self._get_connection()
        if not conn:
            return {
                "total_users": 0,
                "users_by_date": [],
                "new_users": 0,
                "active_users": 0
            }
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Build date filter
            date_filter = ""
            params = []
            if start_date or end_date:
                conditions = []
                if start_date:
                    conditions.append("DATE(created_at) >= %s")
                    params.append(start_date.date())
                if end_date:
                    conditions.append("DATE(created_at) <= %s")
                    params.append(end_date.date())
                if conditions:
                    date_filter = "WHERE " + " AND ".join(conditions)
            
            # Get total unique users
            cursor.execute(f"""
                SELECT COUNT(DISTINCT session_id) as total_users
                FROM chatbot_sessions
                {date_filter}
            """, params)
            total_result = cursor.fetchone()
            
            # Get users by date
            cursor.execute(f"""
                SELECT 
                    DATE(created_at) as date,
                    COUNT(DISTINCT session_id) as user_count
                FROM chatbot_sessions
                {date_filter}
                GROUP BY DATE(created_at)
                ORDER BY date DESC
                LIMIT 30
            """, params)
            users_by_date = cursor.fetchall()
            
            # Get new users (first time sessions) - simplified approach
            if date_filter:
                cursor.execute(f"""
                    SELECT COUNT(DISTINCT session_id) as new_users
                    FROM chatbot_sessions cs
                    WHERE cs.created_at = (
                        SELECT MIN(cs2.created_at)
                        FROM chatbot_sessions cs2
                        WHERE cs2.session_id = cs.session_id
                    )
                    {date_filter.replace('created_at', 'cs.created_at')}
                """, params)
            else:
                cursor.execute("""
                    SELECT COUNT(DISTINCT session_id) as new_users
                    FROM chatbot_sessions cs
                    WHERE cs.created_at = (
                        SELECT MIN(cs2.created_at)
                        FROM chatbot_sessions cs2
                        WHERE cs2.session_id = cs.session_id
                    )
                """)
            new_users_result = cursor.fetchone()
            
            # Get active users (sessions with activity in last 7 days)
            cursor.execute("""
                SELECT COUNT(DISTINCT session_id) as active_users
                FROM chatbot_sessions
                WHERE created_at >= NOW() - INTERVAL '7 days'
            """)
            active_result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return {
                "total_users": total_result['total_users'] or 0,
                "users_by_date": [{
                    "date": str(row['date']),
                    "count": row['user_count']
                } for row in users_by_date],
                "new_users": new_users_result['new_users'] or 0 if new_users_result else 0,
                "active_users": active_result['active_users'] or 0
            }
            
        except Exception as e:
            print(f"❌ Error getting users analytics: {e}")
            if conn:
                conn.close()
            return {
                "total_users": 0,
                "users_by_date": [],
                "new_users": 0,
                "active_users": 0
            }
    
    def get_feedback_with_conversations(
        self, 
        limit: int = 50,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        rating: Optional[int] = None
    ) -> List[Dict]:
        """Get feedback with associated user queries and bot responses, with optional filters"""
        if not self.enabled:
            return []
        
        conn = self._get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Build WHERE clause for filters
            where_conditions = []
            params = []
            
            if start_date:
                where_conditions.append("DATE(cr.timestamp) >= %s")
                params.append(start_date.date())
                print(f"🔍 Adding start_date filter: {start_date.date()}")
            
            if end_date:
                where_conditions.append("DATE(cr.timestamp) <= %s")
                params.append(end_date.date())
                print(f"🔍 Adding end_date filter: {end_date.date()}")
            
            if rating is not None:
                where_conditions.append("cr.rating = %s")
                params.append(rating)
                print(f"🔍 Adding rating filter: {rating}")
            
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            print(f"🔍 SQL WHERE clause: {where_clause}")
            print(f"🔍 SQL params: {params}")
            
            # Get ratings with message_id and join with conversations if possible
            # Use DISTINCT ON to avoid duplicates from JOIN
            # Note: DISTINCT ON requires the columns to be first in ORDER BY
            query = f"""
                SELECT DISTINCT ON (cr.id)
                    cr.id,
                    cr.session_id,
                    cr.message_id,
                    cr.rating,
                    cr.timestamp,
                    cr.user_name,
                    c.user_message,
                    c.bot_response,
                    c.created_at as conversation_time
                FROM chatbot_ratings cr
                LEFT JOIN conversations c ON cr.session_id = c.session_id
                {where_clause}
                ORDER BY cr.id, cr.timestamp DESC, c.created_at DESC NULLS LAST
                LIMIT %s
            """
            params.append(limit)
            
            print(f"🔍 Final SQL query: {query}")
            print(f"🔍 Final params: {params}")
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            print(f"🔍 Query returned {len(results)} results")
            
            cursor.close()
            conn.close()
            
            feedback_list = []
            for row in results:
                feedback_list.append({
                    "id": row['id'],
                    "session_id": row['session_id'],
                    "message_id": row['message_id'],
                    "rating": row['rating'],
                    "timestamp": self._format_ist_datetime(row['timestamp']) if row['timestamp'] else None,
                    "user_name": row['user_name'],
                    "user_query": row['user_message'],
                    "bot_response": row['bot_response'],
                    "conversation_time": self._format_ist_datetime(row['conversation_time']) if row.get('conversation_time') else None
                })
            
            return feedback_list
            
        except Exception as e:
            print(f"❌ Error getting feedback with conversations: {e}")
            if conn:
                conn.close()
            return []
    
    def get_session_times_analytics(self) -> Dict:
        """Get session time analytics per user"""
        if not self.enabled:
            return {
                "average_time": 0,
                "total_time": 0,
                "sessions_by_time": [],
                "top_users_by_time": []
            }
        
        conn = self._get_connection()
        if not conn:
            return {
                "average_time": 0,
                "total_time": 0,
                "sessions_by_time": [],
                "top_users_by_time": []
            }
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get session time statistics
            cursor.execute("""
                SELECT 
                    AVG(total_time_seconds) as avg_time,
                    SUM(total_time_seconds) as total_time,
                    COUNT(*) as completed_sessions
                FROM chatbot_sessions
                WHERE total_time_seconds IS NOT NULL
            """)
            stats = cursor.fetchone()
            
            # Get sessions grouped by time ranges
            cursor.execute("""
                SELECT 
                    time_range,
                    COUNT(*) as session_count
                FROM (
                    SELECT 
                        CASE
                            WHEN total_time_seconds < 60 THEN '0-1 min'
                            WHEN total_time_seconds < 300 THEN '1-5 min'
                            WHEN total_time_seconds < 600 THEN '5-10 min'
                            WHEN total_time_seconds < 1800 THEN '10-30 min'
                            ELSE '30+ min'
                        END as time_range
                    FROM chatbot_sessions
                    WHERE total_time_seconds IS NOT NULL
                ) as time_ranges
                GROUP BY time_range
                ORDER BY 
                    CASE time_range
                        WHEN '0-1 min' THEN 1
                        WHEN '1-5 min' THEN 2
                        WHEN '5-10 min' THEN 3
                        WHEN '10-30 min' THEN 4
                        ELSE 5
                    END
            """)
            sessions_by_time = cursor.fetchall()
            
            # Get top users by total session time
            cursor.execute("""
                SELECT 
                    session_id,
                    user_name,
                    SUM(total_time_seconds) as total_time,
                    COUNT(*) as session_count
                FROM chatbot_sessions
                WHERE total_time_seconds IS NOT NULL
                GROUP BY session_id, user_name
                ORDER BY total_time DESC
                LIMIT 10
            """)
            top_users = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return {
                "average_time": round(stats['avg_time'] or 0, 2),
                "total_time": stats['total_time'] or 0,
                "completed_sessions": stats['completed_sessions'] or 0,
                "sessions_by_time": [{
                    "range": row['time_range'],
                    "count": row['session_count']
                } for row in sessions_by_time],
                "top_users_by_time": [{
                    "session_id": row['session_id'],
                    "user_name": row['user_name'] or f"User {row['session_id'][:8]}",
                    "total_time": row['total_time'],
                    "session_count": row['session_count']
                } for row in top_users]
            }
            
        except Exception as e:
            print(f"❌ Error getting session times analytics: {e}")
            if conn:
                conn.close()
            return {
                "average_time": 0,
                "total_time": 0,
                "sessions_by_time": [],
                "top_users_by_time": []
            }
    
    def get_all_users_filtered(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Dict]:
        """
        Get all sessions with date filtering for admin dashboard
        """
        if not self.enabled:
            return {}
        
        conn = self._get_connection(autocommit=True)
        if not conn:
            return {}
        
        sessions_data = {}
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Build date filter
            date_filter = ""
            params = []
            if start_date or end_date:
                conditions = []
                if start_date:
                    conditions.append("DATE(c.created_at) >= %s")
                    params.append(start_date.date())
                if end_date:
                    conditions.append("DATE(c.created_at) <= %s")
                    params.append(end_date.date())
                if conditions:
                    date_filter = "WHERE " + " AND ".join(conditions)
            
            # Single optimized query to get all session data at once with date filter
            cursor.execute(f"""
                WITH session_stats AS (
                    SELECT 
                        c.session_id,
                        MIN(c.user_id) as user_id,
                        COUNT(*) as conversation_count,
                        MIN(c.created_at) as first_message_at,
                        MAX(c.created_at) as last_message_at,
                        ARRAY_AGG(c.recommendations) FILTER (WHERE c.recommendations IS NOT NULL) as all_recommendations
                    FROM conversations c
                    {date_filter}
                    GROUP BY c.session_id
                )
                SELECT 
                    ss.session_id,
                    ss.user_id,
                    ss.conversation_count,
                    ss.first_message_at,
                    ss.last_message_at,
                    ss.all_recommendations,
                    up.name as user_name,
                    up.email as user_email,
                    sa.total_messages,
                    sa.total_recommendations
                FROM session_stats ss
                LEFT JOIN user_profiles up ON ss.user_id = up.user_id
                LEFT JOIN session_analytics sa ON ss.session_id = sa.session_id
                ORDER BY ss.last_message_at DESC
            """, params)
            
            rows = cursor.fetchall()
            cursor.close()
            
            # Process results (same as get_all_users)
            for row in rows:
                session_id = row['session_id']
                user_id = row['user_id']
                
                total_recs = 0
                if row.get('all_recommendations'):
                    for rec in row['all_recommendations']:
                        if rec:
                            try:
                                if isinstance(rec, str):
                                    if rec and rec != '' and rec != '[]':
                                        rec_list = json.loads(rec)
                                        if isinstance(rec_list, list):
                                            total_recs += len(rec_list)
                                        else:
                                            total_recs += 1
                                elif isinstance(rec, list):
                                    total_recs += len(rec)
                                else:
                                    total_recs += 1
                            except (json.JSONDecodeError, TypeError):
                                if rec and rec != '' and rec != '[]':
                                    total_recs += 1
                
                if row.get('total_recommendations') is not None:
                    total_recs = row['total_recommendations']
                
                first_login = 'N/A'
                if row.get('first_message_at'):
                    first_login = self._format_ist_datetime(row['first_message_at'])
                
                display_name = session_id
                if row.get('user_name'):
                    display_name = f"{row['user_name']} ({session_id[:8]}...)"
                elif row.get('user_email'):
                    display_name = f"{row['user_email']} ({session_id[:8]}...)"
                elif user_id:
                    display_name = f"User {user_id[:8]}... ({session_id[:8]}...)"
                
                sessions_data[session_id] = {
                    'session_id': session_id,
                    'user_id': user_id,
                    'display_name': display_name,
                    'login_time': first_login,
                    'conversation_count': row.get('conversation_count', 0),
                    'total_recommendations': total_recs
                }
            
            conn.close()
            return sessions_data
            
        except Exception as e:
            print(f"❌ Error getting filtered sessions: {e}")
            if conn:
                try:
                    conn.rollback()
                    conn.close()
                except:
                    pass
            return {}

