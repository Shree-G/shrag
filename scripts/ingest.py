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
# from core.embeddings import embedding_model
# from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.settings import Settings
settings = Settings()

def preview_chunks(chunks, filename="chunks_debug.txt"):
    """
    Writes all chunks to a text file for inspection.
    """
    print(f"Writing {len(chunks)} chunks to {filename} for inspection...")
    
    with open(filename, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"{'='*20} CHUNK {i+1} {'='*20}\n")
            f.write(f"Source: {chunk.metadata.get('source_file', 'Unknown')}\n")
            f.write(f"Length: {len(chunk.page_content)} characters\n")
            f.write(f"{'-'*20}\n")
            f.write(chunk.page_content)
            f.write(f"\n\n")
    
    print(f"Done! Open {filename} to verify your chunking quality.")

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

    # text_splitter = SemanticChunker(
    #     embeddings=embedding_model,
    #     breakpoint_threshold_type="percentile",
    #     min_chunk_size=500
    #     # breakpoint_threshold_amount=95.0 # this is the default value
    # )

    # final_chunks = []

    # safety_cap = 4000

    # for doc in documents:
    #     if len(doc.page_content) < safety_cap:
    #         print(f"Keeping whole: {doc.metadata.get('source_file') or doc.metadata.get('repo_name')} ({len(doc.page_content)} chars)")
    #         final_chunks.append(doc)
    #     else:
    #         print(f"Splitting large doc: {doc.metadata.get('source_file') or doc.metadata.get('repo_name')} ({len(doc.page_content)} chars)")
    #         sub_chunks = text_splitter.split_documents([doc])
    #         final_chunks.extend(sub_chunks)

    # preview_chunks(final_chunks)

    final_chunks = text_splitter.split_documents(documents)

    try:
        print(f"Clearing old data from collection: {vector_store._collection.name}...")
        vector_store.reset_collection()
        print("Old data cleared.")
    except Exception as e:
        # This is expected on the very first run when the
        # collection doesn't exist yet. We can safely ignore it.
        print(f"Info: Could not clear collection (this is normal on first run): {e}")

    vector_store.add_documents(final_chunks)
    
    print("\n--- Ingestion Complete ---")
    print(f"Total documents loaded: {len(documents)}")
    print(f"Total chunks created: {len(final_chunks)}")
    print(f"Vector store populated and ready at: {settings.VECTOR_DB_PATH}")

if __name__ == "__main__":
    main()