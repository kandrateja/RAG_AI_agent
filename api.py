"""
FastAPI Backend for the Deficiency Data Chatbot.
Production-ready API with SSO user session support.
Designed to be embedded in enterprise applications.
"""

import os
import time
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from backend.data_service import get_data_service, initialize_data_service
from backend.llm_service import LLMService
from backend.feedback_store import get_feedback_store

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ MODELS ============

class UserInfo(BaseModel):
    """User information from SSO."""
    user_id: str = Field(..., description="Unique user ID from SSO")
    user_name: Optional[str] = Field(None, description="User display name")
    user_email: Optional[str] = Field(None, description="User email")
    department: Optional[str] = Field(None, description="User department")


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    """Chat request with user context."""
    message: str = Field(..., description="User's question")
    user_id: str = Field(..., description="User ID from SSO")
    user_name: Optional[str] = Field(None, description="User display name")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")


class ChatResponse(BaseModel):
    """Chat response with session info."""
    response: str
    session_id: str
    message_id: str
    tools_used: List[Dict[str, Any]]
    has_data: bool
    data_summary: Optional[str] = None
    execution_time_ms: int


class FeedbackRequest(BaseModel):
    """Feedback submission."""
    message_id: str
    user_id: str
    feedback_type: str  # "positive" or "negative"
    comment: Optional[str] = None


class SessionHistoryResponse(BaseModel):
    """User's conversation history."""
    session_id: str
    messages: List[ChatMessage]
    created_at: str
    last_activity: str


# ============ SESSION STORE ============

class UserSessionStore:
    """In-memory store for user sessions with conversation history."""
    
    def __init__(self, max_history: int = 20, session_timeout_hours: int = 24):
        self.sessions: Dict[str, Dict] = {}
        self.max_history = max_history
        self.session_timeout = timedelta(hours=session_timeout_hours)
    
    def get_or_create_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Get existing session or create new one for user."""
        # Clean expired sessions
        self._cleanup_expired()
        
        # If session_id provided and valid, use it
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            if session["user_id"] == user_id:
                session["last_activity"] = datetime.now()
                return session_id
        
        # Create new session
        new_session_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
        self.sessions[new_session_id] = {
            "user_id": user_id,
            "messages": [],
            "chat_history": [],  # For LLM context
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        }
        logger.info(f"Created new session for user {user_id}: {new_session_id}")
        return new_session_id
    
    def add_message(self, session_id: str, role: str, content: str, message_id: str = None):
        """Add a message to session history."""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "message_id": message_id
        }
        session["messages"].append(message)
        
        # Also maintain LLM chat history format
        session["chat_history"].append({"role": role, "content": content})
        
        # Trim to max history
        if len(session["chat_history"]) > self.max_history:
            session["chat_history"] = session["chat_history"][-self.max_history:]
        
        session["last_activity"] = datetime.now()
    
    def get_chat_history(self, session_id: str) -> List[Dict]:
        """Get LLM-formatted chat history for a session."""
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id]["chat_history"]
    
    def get_session_messages(self, session_id: str) -> List[Dict]:
        """Get all messages for a session."""
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id]["messages"]
    
    def get_user_sessions(self, user_id: str) -> List[str]:
        """Get all session IDs for a user."""
        return [
            sid for sid, session in self.sessions.items()
            if session["user_id"] == user_id
        ]
    
    def clear_session(self, session_id: str):
        """Clear a session's history."""
        if session_id in self.sessions:
            self.sessions[session_id]["messages"] = []
            self.sessions[session_id]["chat_history"] = []
    
    def _cleanup_expired(self):
        """Remove expired sessions."""
        now = datetime.now()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session["last_activity"] > self.session_timeout
        ]
        for sid in expired:
            del self.sessions[sid]
            logger.info(f"Cleaned up expired session: {sid}")


# ============ GLOBAL INSTANCES ============

session_store = UserSessionStore()
llm_service: Optional[LLMService] = None


# ============ LIFESPAN ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global llm_service
    
    logger.info("Starting Deficiency Data Chatbot API...")
    
    # Initialize data service
    data_path = os.environ.get("DATA_FILE_PATH")
    initialize_data_service(data_path)
    logger.info("Data service initialized")
    
    # Initialize LLM service
    try:
        llm_service = LLMService()
        logger.info("LLM service initialized")
    except ValueError as e:
        logger.warning(f"LLM service not initialized: {e}")
    
    yield
    
    logger.info("Shutting down API...")


# ============ APP SETUP ============

app = FastAPI(
    title="Deficiency Data Chatbot API",
    description="""
    LLM-powered chatbot API for querying FDA deficiency data.
    
    ## Features
    - Natural language queries about deficiency data
    - Per-user session management (SSO compatible)
    - Conversation history
    - Feedback collection
    
    ## Authentication
    Pass user information via `X-User-ID` header or in request body.
    Your SSO system should provide the user_id after authentication.
    """,
    version="2.0.0",
    lifespan=lifespan
)

# CORS - Configure for your domain in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: ["https://your-app.company.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ DEPENDENCY ============

async def get_user_from_header(
    x_user_id: Optional[str] = Header(None),
    x_user_name: Optional[str] = Header(None),
    x_user_email: Optional[str] = Header(None)
) -> Optional[UserInfo]:
    """Extract user info from headers (set by SSO/gateway)."""
    if x_user_id:
        return UserInfo(
            user_id=x_user_id,
            user_name=x_user_name,
            user_email=x_user_email
        )
    return None


# ============ ENDPOINTS ============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    data_service = get_data_service()
    schema = data_service.get_schema_info()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": schema.get("total_records", 0) > 0,
        "total_records": schema.get("total_records", 0),
        "llm_available": llm_service is not None
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    header_user: Optional[UserInfo] = Depends(get_user_from_header)
):
    """
    Send a chat message and get a response.
    
    - **message**: The user's question in natural language
    - **user_id**: User identifier from SSO (required)
    - **session_id**: Optional session ID for conversation continuity
    
    Returns the AI response with session info for subsequent requests.
    """
    if llm_service is None:
        raise HTTPException(status_code=503, detail="LLM service not available")
    
    # Use header user_id if available, otherwise use request body
    user_id = header_user.user_id if header_user else request.user_id
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    # Get or create session
    session_id = session_store.get_or_create_session(user_id, request.session_id)
    message_id = f"{session_id}_{uuid.uuid4().hex[:8]}"
    
    # Get conversation history for context
    chat_history = session_store.get_chat_history(session_id)
    
    # Process with LLM
    start_time = time.time()
    try:
        result = llm_service.chat(request.message, chat_history)
        execution_time = int((time.time() - start_time) * 1000)
        
        # Log query
        feedback_store = get_feedback_store()
        feedback_store.log_query(
            query_id=message_id,
            query=request.message,
            response=result["response"],
            tools_used=result.get("tool_calls", []),
            tool_results=result.get("data", []),
            execution_time_ms=execution_time,
            success=True
        )
        
    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        logger.error(f"Chat error for user {user_id}: {e}")
        
        # Log error
        feedback_store = get_feedback_store()
        feedback_store.log_query(
            query_id=message_id,
            query=request.message,
            response=str(e),
            tools_used=[],
            tool_results=[],
            execution_time_ms=execution_time,
            success=False,
            error_message=str(e)
        )
        
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    
    # Store messages in session
    session_store.add_message(session_id, "user", request.message, message_id)
    session_store.add_message(session_id, "assistant", result["response"], message_id)
    
    # Prepare response
    tools_used = result.get("tool_calls", [])
    has_data = len(result.get("data", [])) > 0
    data_summary = None
    if has_data:
        summaries = [tc.get("result_summary", "") for tc in tools_used if tc.get("result_summary")]
        data_summary = "; ".join(summaries) if summaries else "Data retrieved"
    
    return ChatResponse(
        response=result["response"],
        session_id=session_id,
        message_id=message_id,
        tools_used=tools_used,
        has_data=has_data,
        data_summary=data_summary,
        execution_time_ms=execution_time
    )


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for a chat response.
    
    - **message_id**: The message ID from the chat response
    - **user_id**: User identifier
    - **feedback_type**: "positive" or "negative"
    - **comment**: Optional feedback comment
    """
    feedback_store = get_feedback_store()
    
    success = feedback_store.save_feedback(
        query_id=request.message_id,
        query="",  # Will be filled from logs if needed
        response="",
        tools_used=[],
        feedback_type=request.feedback_type,
        comment=request.comment
    )
    
    if success:
        return {"status": "ok", "message": "Feedback recorded"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save feedback")


@app.get("/api/session/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(
    session_id: str,
    user_id: str,
    header_user: Optional[UserInfo] = Depends(get_user_from_header)
):
    """Get conversation history for a session."""
    # Verify user owns session
    actual_user_id = header_user.user_id if header_user else user_id
    
    messages = session_store.get_session_messages(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = session_store.sessions.get(session_id, {})
    if session.get("user_id") != actual_user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return SessionHistoryResponse(
        session_id=session_id,
        messages=[ChatMessage(**m) for m in messages],
        created_at=session.get("created_at", datetime.now()).isoformat(),
        last_activity=session.get("last_activity", datetime.now()).isoformat()
    )


@app.delete("/api/session/{session_id}")
async def clear_session(
    session_id: str,
    user_id: str,
    header_user: Optional[UserInfo] = Depends(get_user_from_header)
):
    """Clear conversation history for a session."""
    actual_user_id = header_user.user_id if header_user else user_id
    
    session = session_store.sessions.get(session_id, {})
    if session.get("user_id") != actual_user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    session_store.clear_session(session_id)
    return {"status": "ok", "message": "Session cleared"}


@app.get("/api/user/{user_id}/sessions")
async def get_user_sessions(
    user_id: str,
    header_user: Optional[UserInfo] = Depends(get_user_from_header)
):
    """Get all sessions for a user."""
    actual_user_id = header_user.user_id if header_user else user_id
    
    if actual_user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    sessions = session_store.get_user_sessions(user_id)
    return {"user_id": user_id, "sessions": sessions}


# ============ DATA ENDPOINTS (Direct Access) ============

@app.get("/api/data/summary")
async def get_data_summary():
    """Get summary statistics of the deficiency data."""
    data_service = get_data_service()
    return data_service.get_summary_statistics()


@app.get("/api/data/unique/{column}")
async def get_unique_values(column: str):
    """Get unique values for a column."""
    data_service = get_data_service()
    valid_columns = ["CATEGORY", "SUBCATEGORY", "Markets", "GEOGRAPHY", "VERTICAL", "PLANT", "DosageForm"]
    if column not in valid_columns:
        raise HTTPException(status_code=400, detail=f"Invalid column. Must be one of: {valid_columns}")
    return {"column": column, "values": data_service.get_unique_values(column)}


# ============ EMBEDDABLE WIDGET ============

@app.get("/widget", response_class=HTMLResponse)
async def get_chat_widget():
    """
    Get the embeddable chat widget HTML/JS.
    
    Include this in your application:
    ```html
    <script src="https://your-api-host/widget"></script>
    <script>
        DeficiencyChatbot.init({
            apiUrl: 'https://your-api-host',
            userId: 'user-from-sso',
            userName: 'User Name'
        });
    </script>
    ```
    """
    widget_html = """
<!DOCTYPE html>
<html>
<head>
<style>
/* Chat Widget Styles */
.dc-widget-container {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 10000;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.dc-chat-button {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform 0.2s;
}

.dc-chat-button:hover {
    transform: scale(1.1);
}

.dc-chat-button svg {
    width: 28px;
    height: 28px;
    fill: white;
}

.dc-chat-window {
    display: none;
    position: fixed;
    bottom: 90px;
    right: 20px;
    width: 380px;
    height: 520px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    flex-direction: column;
    overflow: hidden;
}

.dc-chat-window.open {
    display: flex;
}

.dc-chat-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.dc-chat-header h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
}

.dc-close-btn {
    background: none;
    border: none;
    color: white;
    cursor: pointer;
    font-size: 20px;
    padding: 0;
}

.dc-chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    background: #f8f9fa;
}

.dc-message {
    margin-bottom: 12px;
    max-width: 85%;
}

.dc-message.user {
    margin-left: auto;
}

.dc-message-content {
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 14px;
    line-height: 1.4;
}

.dc-message.user .dc-message-content {
    background: #667eea;
    color: white;
    border-bottom-right-radius: 4px;
}

.dc-message.assistant .dc-message-content {
    background: white;
    color: #333;
    border-bottom-left-radius: 4px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

.dc-feedback-btns {
    display: flex;
    gap: 8px;
    margin-top: 8px;
}

.dc-feedback-btn {
    background: #f0f0f0;
    border: none;
    padding: 4px 8px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
}

.dc-feedback-btn:hover {
    background: #e0e0e0;
}

.dc-feedback-btn.selected {
    background: #667eea;
    color: white;
}

.dc-chat-input-area {
    padding: 12px;
    border-top: 1px solid #eee;
    display: flex;
    gap: 8px;
    background: white;
}

.dc-chat-input {
    flex: 1;
    padding: 10px 14px;
    border: 1px solid #ddd;
    border-radius: 20px;
    font-size: 14px;
    outline: none;
}

.dc-chat-input:focus {
    border-color: #667eea;
}

.dc-send-btn {
    background: #667eea;
    color: white;
    border: none;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
}

.dc-send-btn:disabled {
    background: #ccc;
    cursor: not-allowed;
}

.dc-typing {
    display: flex;
    gap: 4px;
    padding: 10px;
}

.dc-typing span {
    width: 8px;
    height: 8px;
    background: #667eea;
    border-radius: 50%;
    animation: dc-bounce 1.4s infinite ease-in-out;
}

.dc-typing span:nth-child(1) { animation-delay: -0.32s; }
.dc-typing span:nth-child(2) { animation-delay: -0.16s; }

@keyframes dc-bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
}
</style>
</head>
<body>
<script>
(function() {
    'use strict';
    
    window.DeficiencyChatbot = {
        config: {
            apiUrl: '',
            userId: '',
            userName: '',
            sessionId: null
        },
        
        init: function(options) {
            this.config = { ...this.config, ...options };
            this.createWidget();
            this.bindEvents();
        },
        
        createWidget: function() {
            const container = document.createElement('div');
            container.className = 'dc-widget-container';
            container.innerHTML = `
                <div class="dc-chat-window" id="dc-chat-window">
                    <div class="dc-chat-header">
                        <h3>🔬 Deficiency Assistant</h3>
                        <button class="dc-close-btn" id="dc-close-btn">&times;</button>
                    </div>
                    <div class="dc-chat-messages" id="dc-messages">
                        <div class="dc-message assistant">
                            <div class="dc-message-content">
                                Hello${this.config.userName ? ' ' + this.config.userName : ''}! I can help you query deficiency data. Ask me anything!
                            </div>
                        </div>
                    </div>
                    <div class="dc-chat-input-area">
                        <input type="text" class="dc-chat-input" id="dc-input" placeholder="Ask about deficiencies...">
                        <button class="dc-send-btn" id="dc-send-btn">
                            <svg viewBox="0 0 24 24" width="20" height="20" fill="white">
                                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                            </svg>
                        </button>
                    </div>
                </div>
                <button class="dc-chat-button" id="dc-toggle-btn">
                    <svg viewBox="0 0 24 24">
                        <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>
                    </svg>
                </button>
            `;
            document.body.appendChild(container);
        },
        
        bindEvents: function() {
            const self = this;
            
            document.getElementById('dc-toggle-btn').addEventListener('click', function() {
                document.getElementById('dc-chat-window').classList.toggle('open');
            });
            
            document.getElementById('dc-close-btn').addEventListener('click', function() {
                document.getElementById('dc-chat-window').classList.remove('open');
            });
            
            document.getElementById('dc-send-btn').addEventListener('click', function() {
                self.sendMessage();
            });
            
            document.getElementById('dc-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') self.sendMessage();
            });
        },
        
        addMessage: function(content, role, messageId) {
            const messagesDiv = document.getElementById('dc-messages');
            const msgDiv = document.createElement('div');
            msgDiv.className = 'dc-message ' + role;
            
            let html = '<div class="dc-message-content">' + this.escapeHtml(content) + '</div>';
            
            if (role === 'assistant' && messageId) {
                html += `
                    <div class="dc-feedback-btns" data-msg-id="${messageId}">
                        <button class="dc-feedback-btn" data-feedback="positive">👍</button>
                        <button class="dc-feedback-btn" data-feedback="negative">👎</button>
                    </div>
                `;
            }
            
            msgDiv.innerHTML = html;
            messagesDiv.appendChild(msgDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            // Bind feedback buttons
            if (messageId) {
                msgDiv.querySelectorAll('.dc-feedback-btn').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        this.submitFeedback(messageId, e.target.dataset.feedback);
                        msgDiv.querySelectorAll('.dc-feedback-btn').forEach(b => b.classList.remove('selected'));
                        e.target.classList.add('selected');
                    });
                });
            }
        },
        
        showTyping: function() {
            const messagesDiv = document.getElementById('dc-messages');
            const typingDiv = document.createElement('div');
            typingDiv.id = 'dc-typing';
            typingDiv.className = 'dc-message assistant';
            typingDiv.innerHTML = '<div class="dc-typing"><span></span><span></span><span></span></div>';
            messagesDiv.appendChild(typingDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        },
        
        hideTyping: function() {
            const typing = document.getElementById('dc-typing');
            if (typing) typing.remove();
        },
        
        async sendMessage: function() {
            const input = document.getElementById('dc-input');
            const message = input.value.trim();
            if (!message) return;
            
            input.value = '';
            this.addMessage(message, 'user');
            this.showTyping();
            
            try {
                const response = await fetch(this.config.apiUrl + '/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-User-ID': this.config.userId,
                        'X-User-Name': this.config.userName || ''
                    },
                    body: JSON.stringify({
                        message: message,
                        user_id: this.config.userId,
                        session_id: this.config.sessionId
                    })
                });
                
                const data = await response.json();
                this.hideTyping();
                
                if (response.ok) {
                    this.config.sessionId = data.session_id;
                    this.addMessage(data.response, 'assistant', data.message_id);
                } else {
                    this.addMessage('Sorry, an error occurred. Please try again.', 'assistant');
                }
            } catch (error) {
                this.hideTyping();
                this.addMessage('Connection error. Please check your network.', 'assistant');
            }
        },
        
        async submitFeedback: function(messageId, feedbackType) {
            try {
                await fetch(this.config.apiUrl + '/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message_id: messageId,
                        user_id: this.config.userId,
                        feedback_type: feedbackType
                    })
                });
            } catch (error) {
                console.error('Feedback error:', error);
            }
        },
        
        escapeHtml: function(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    };
})();
</script>
</body>
</html>
    """
    return widget_html


# ============ RUN ============

if __name__ == "__main__":
    import uvicorn
    
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8000))
    
    uvicorn.run("backend.api:app", host=host, port=port, reload=True)
