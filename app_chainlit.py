import chainlit as cl
import sys
import os
from dotenv import load_dotenv

# --- 1. Set up your Python path ---
load_dotenv()
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- 2. Import your RAG chain and the *correct* history class ---
from core.chain import conversational_rag_chain # <-- The base chain
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- THIS IS THE NEW, CORRECT IMPORT ---
# We use the standard in-memory history from langchain-core
from langchain_core.chat_history import InMemoryChatMessageHistory

@cl.on_chat_start
async def on_chat_start():
    """
    This function runs when a new user connects.
    We will create and store the chat history *inside* the
    chainlit user session.
    """
    
    # 1. Create a new, empty history object for this user
    history = InMemoryChatMessageHistory()
    
    # 2. Store this history object in Chainlit's user session
    cl.user_session.set("chat_history", history)
    
    # 3. Create the memory-wrapped chain
    memory_chain = RunnableWithMessageHistory(
        conversational_rag_chain,
        
        # This lambda function now correctly fetches the
        # history object we just stored in the user's session.
        lambda session_id: cl.user_session.get("chat_history"),
        
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    # 4. Store the runnable in the user's session
    cl.user_session.set("runnable", memory_chain)
    
    await cl.Message(
        content="Hello! I'm Shree's assistant. How can I help you today?"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """
    This function runs every time the user sends a message.
    """
    # 1. Get the runnable chain from the user's session
    runnable = cl.user_session.get("runnable")
    
    # 2. Get the unique session ID from chainlit
    session_id = cl.user_session.get("id")
    
    # 3. Create the config
    config = {"configurable": {"session_id": session_id}}
    
    # 4. Create the input for the chain
    input_data = {"input": message.content}
    
    # 5. Stream the response
    response_stream = cl.Message(content="")
    await response_stream.send()

    # 6. Run the chain asynchronously
    async for chunk in runnable.astream(input_data, config=config):
        if "answer" in chunk:
            await response_stream.stream_token(chunk["answer"])
    
    # 7. Send the final, streamed message
    await response_stream.update()