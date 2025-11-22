from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Import your Pydantic models
from app.models import ChatRequest, ChatResponse, DocumentSource

# Import your final, history-aware RAG chain
from core.chain import final_rag_chain

# Initialize the FastAPI app
app = FastAPI(
    title="Shree's Personal RAG API",
    description="A chatbot API for Shree's personal website, "
                "powered by LangChain, Groq, and Chroma."
)

# --- CORS Configuration ---
origins = [
    "http://localhost:3000",  # Your local frontend dev server
    "http://localhost:5173",  # Another common local dev port
    #"https://your-website.com", # Your *deployed* website URL
    # Add any other origins you need
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# --- API Endpoints ---

@app.get("/")
def read_root():
    """
    A simple "health check" endpoint to confirm the API is running.
    """
    return {"status": "ok", "message": "Welcome to Shree's RAG API!"}


@app.post("/chat", response_model=ChatResponse)
async def chat_handler(request: ChatRequest):
    """
    The main chat endpoint.
    
    Receives a user's query and session_id, calls the
    conversational RAG chain, and returns the answer
    along with the source documents.
    """
    
    # 1. Create the config object for the chain
    # This tells the chain which session to use for memory
    config = {"configurable": {"session_id": request.session_id}}
    
    # 2. Create the input for the chain
    # This must match the 'input_messages_key' from your chain.py
    input_data = {"input": request.query}
    
    # 3. Invoke the chain
    # The chain will automatically:
    # - Load history (using get_session_history)
    # - Re-phrase the question
    # - Retrieve documents
    # - Generate the answer
    # - Save the new messages to history
    response = final_rag_chain.invoke(input_data, config=config)
    
    # 4. Format the response
    # The chain's output is a dictionary. We extract
    # the 'answer' and 'context' (which are the documents).
    
    # Convert the LangChain Document objects into Pydantic models
    # source_docs = [
    #     DocumentSource(
    #         page_content=doc.page_content,
    #         metadata=doc.metadata
    #     ) for doc in response.get("context", [])
    # ]

    source_docs = []
    for doc in response.get("context", []):
        # Flashrank adds metadata values as 'numpy.float32', which 
        # Pydantic/JSON cannot serialize. We must convert them to standard floats.
        safe_metadata = {}
        for key, value in doc.metadata.items():
            # We check the type string to avoid importing numpy directly
            if "numpy" in str(type(value)):
                safe_metadata[key] = float(value)
            else:
                safe_metadata[key] = value
        
        source_docs.append(DocumentSource(
            page_content=doc.page_content,
            metadata=safe_metadata
        ))
    
    return ChatResponse(
        answer=response.get("answer", "Error: No answer found."),
        source_documents=source_docs
    )