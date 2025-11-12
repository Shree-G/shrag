# app/models.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class DocumentSource(BaseModel):
    """
    A Pydantic model for a single source document.
    """
    page_content: str
    metadata: Dict[str, Any] = {}

class ChatRequest(BaseModel):
    """
    Defines the JSON structure for a request to the /chat endpoint.
    """
    # The user's new question
    query: str
    
    # The unique ID for the conversation
    # We make it optional with a default for easier testing
    session_id: Optional[str] = "default-session"

class ChatResponse(BaseModel):
    """
    Defines the JSON structure for a response from the /chat endpoint.
    """
    # The chatbot's final answer
    answer: str
    
    # A list of source documents used to generate the answer.
    # This is a very impressive feature to show recruiters.
    source_documents: List[DocumentSource] = []