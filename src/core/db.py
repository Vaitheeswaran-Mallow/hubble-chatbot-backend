import chromadb

def get_chroma_client():
    """Initialize and return a ChromaDB client."""
    return chromadb.PersistentClient(path="./chroma_db")
