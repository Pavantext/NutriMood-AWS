"""
Session Service - Manages user sessions, conversation history, and preferences
"""

from typing import Dict, List, Optional
from datetime import datetime
import json

class SessionService:
    def __init__(self):
        """Initialize session storage (in-memory for now)"""
        self.sessions = {}  # session_id -> session_data
        self.max_history_length = 50  # Maximum messages to keep per session
    
    def get_or_create_session(self, session_id: str) -> Dict:
        """Get existing session or create a new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat(),
                "messages": [],
                "recommendations": [],
                "preferences": {},
                "metadata": {}
            }
        else:
            # Update last activity
            self.sessions[session_id]["last_activity"] = datetime.now().isoformat()
        
        return self.sessions[session_id]
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data by ID"""
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to session history"""
        session = self.get_or_create_session(session_id)
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        session["messages"].append(message)
        
        # Trim history if too long
        if len(session["messages"]) > self.max_history_length:
            session["messages"] = session["messages"][-self.max_history_length:]
        
        session["last_activity"] = datetime.now().isoformat()
    
    def get_conversation_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Get conversation history for a session"""
        session = self.get_session(session_id)
        
        if not session:
            return []
        
        messages = session.get("messages", [])
        
        if limit:
            return messages[-limit:]
        
        return messages
    
    def add_recommendations(self, session_id: str, food_ids: List[str]):
        """Add recommended food IDs to session"""
        session = self.get_or_create_session(session_id)
        
        recommendation_entry = {
            "food_ids": food_ids,
            "timestamp": datetime.now().isoformat()
        }
        
        session["recommendations"].append(recommendation_entry)
        session["last_activity"] = datetime.now().isoformat()
    
    def update_preferences(self, session_id: str, preferences: Dict):
        """Update user preferences for the session"""
        session = self.get_or_create_session(session_id)
        session["preferences"].update(preferences)
        session["last_activity"] = datetime.now().isoformat()
    
    def get_preferences(self, session_id: str) -> Dict:
        """Get user preferences for a session"""
        session = self.get_session(session_id)
        
        if not session:
            return {}
        
        return session.get("preferences", {})
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all active sessions (for admin/debugging)"""
        return list(self.sessions.values())
    
    def clear_old_sessions(self, hours: int = 24):
        """Clear sessions older than specified hours"""
        from datetime import timedelta
        
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            last_activity = datetime.fromisoformat(session["last_activity"])
            if current_time - last_activity > timedelta(hours=hours):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        return len(expired_sessions)
    
    def export_session(self, session_id: str) -> Optional[str]:
        """Export session data as JSON string"""
        session = self.get_session(session_id)
        
        if not session:
            return None
        
        return json.dumps(session, indent=2)
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a session"""
        session = self.get_session(session_id)
        
        if not session:
            return {}
        
        messages = session.get("messages", [])
        recommendations = session.get("recommendations", [])
        
        user_messages = [m for m in messages if m.get("role") == "user"]
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        
        total_recommendations = sum(len(r.get("food_ids", [])) for r in recommendations)
        
        return {
            "session_id": session_id,
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "total_recommendations": total_recommendations,
            "unique_food_items": len(set(
                food_id 
                for r in recommendations 
                for food_id in r.get("food_ids", [])
            )),
            "session_duration": self._calculate_duration(session),
            "created_at": session.get("created_at"),
            "last_activity": session.get("last_activity")
        }
    
    def _calculate_duration(self, session: Dict) -> str:
        """Calculate session duration in human-readable format"""
        try:
            created = datetime.fromisoformat(session["created_at"])
            last_activity = datetime.fromisoformat(session["last_activity"])
            duration = last_activity - created
            
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            
            if hours > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{minutes}m"
        except:
            return "unknown"