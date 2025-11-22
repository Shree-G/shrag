from core.llm import llm
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from core.loaders import load_all_documents
from app.settings import Settings
from core.chain import final_rag_chain

# template = "what is the capital of india? what is the population?"

# prompt = ChatPromptTemplate.from_template(template)

# chain = (
#     prompt | llm | StrOutputParser()
# )

# print(chain.invoke({}))

## ----------------

# settings = Settings()

# jsonObj = get_public_repo_urls(settings.GITHUB_USERNAME)

# print(jsonObj)

## ----------------

# all_docs = load_all_documents()
# print(all_docs)

## ----------------

def run_test():
    # --- 3. Define a session ID ---
    # This can be any string. It's how the chain remembers the conversation.
    my_session_id = "test-session-123"

    # --- 4. The 'config' object ---
    # This is required by RunnableWithMessageHistory
    config = {"configurable": {"session_id": my_session_id}}

    print("--- Test 1: First question ---")
    
    # --- 5. The 'input' object ---
    # This must be a dictionary with the key you specified
    input1 = {"input": "What projects are on Shree's resume?"}
    
    # --- 6. Call invoke ---
    answer1 = final_rag_chain.invoke(input1, config=config)
    
    # The output is a dictionary, and you want the "answer" key
    print(f"Query: {input1['input']}")
    print(f"Answer: {answer1['answer']}")
    print("-" * 30)
    
    # --- 7. Test memory (follow-up question) ---
    print("--- Test 2: Follow-up question ---")
    
    # Use the *same config* but a *new input*
    input2 = {"input": "Tell me more about the third one."}
    
    answer2 = final_rag_chain.invoke(input2, config=config)
    
    print(f"Query: {input2['input']}")
    print(f"Answer: {answer2['answer']}")
    print("-" * 30)

    input3 = {"input": "Give me a recipe for a cake!"}
    answer3 = final_rag_chain.invoke(input3, config=config)

    print(f"Query: {input3['input']}")
    print(f"Answer: {answer3['answer']}")
    print("-" * 30)

# run_test()

documents = load_all_documents()
print(documents)
