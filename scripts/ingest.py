import sys
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
print(f"Added project root to path: {PROJECT_ROOT}")

from core.loaders import load_all_documents
from core.vector_store import vector_store
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.settings import Settings

settings = Settings()

def main():
    documents = load_all_documents()

    if not documents:
        print("Could not load documents. Exiting ingestion script.")
        return
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = settings.CHUNK_SIZE,
        chunk_overlap = settings.CHUNK_OVERLAP,
        add_start_index=True
    )

    print(f"Splitting {len(documents)} documents into chunks (Chunk size: {settings.CHUNK_SIZE}, Overlap: {settings.CHUNK_OVERLAP})...")

    chunks = text_splitter.split_documents(documents)

    try:
        print(f"Clearing old data from collection: {vector_store._collection.name}...")
        vector_store.delete_collection()
        print("Old data cleared.")
    except Exception as e:
        # This is expected on the very first run when the
        # collection doesn't exist yet. We can safely ignore it.
        print(f"Info: Could not clear collection (this is normal on first run): {e}")

    vector_store.add_documents(chunks)
    
    print("\n--- Ingestion Complete ---")
    print(f"Total documents loaded: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Vector store populated and ready at: {settings.VECTOR_DB_PATH}")

if __name__ == "__main__":
    main()