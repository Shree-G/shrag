from langchain.chains import (
	create_history_aware_retriever,
    create_retrieval_chain
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from core.vector_store import retriever
from core.llm import llm
from core.prompts import (
    REPHRASE_PROMPT,
    RAG_PROMPT
)

# LLM create a standalone question to use for embedding and similarity search
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    REPHRASE_PROMPT
)

# uses the context to answer the question from the LLM
question_answer_chain = create_stuff_documents_chain(
    llm,
    RAG_PROMPT
)

# takes the context (the included documents) as well as the 
# rephrased question with history and passes it into the llm to answer
conversational_rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain
)

chat_history_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]

final_rag_chain = RunnableWithMessageHistory(
    conversational_rag_chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    output_messages_key="answer",
    history_messages_key="chat_history"
)





