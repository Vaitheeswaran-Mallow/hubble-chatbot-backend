import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import uuid
from db import get_chroma_client


class VectorStore:
    """Handles vector storage and retrieval using ChromaDB."""
    
    def __init__(self, collection_name: str = "documents"):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.collection_name = collection_name
        self.client = get_chroma_client()
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Get existing collection or create a new one."""
        try:
            return self.client.get_collection(name=self.collection_name)
        except Exception:
            # Collection doesn't exist, create it
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings for semantic search"}
            )
    
    def _recreate_collection_if_dimension_mismatch(self, expected_dimension: int):
        """Recreate collection if embedding dimension doesn't match."""
        try:
            # Check if collection exists and has the right dimension
            collection = self.client.get_collection(name=self.collection_name)
            
            # Try to get collection info to check dimension
            try:
                # This will fail if dimension mismatch
                collection.peek()
                return collection
            except Exception as e:
                if "dimension" in str(e).lower():
                    print(f"Dimension mismatch detected. Recreating collection...")
                    # Delete the old collection
                    self.client.delete_collection(name=self.collection_name)
                    # Create new collection
                    return self.client.create_collection(
                        name=self.collection_name,
                        metadata={"description": "Document embeddings for semantic search"}
                    )
                else:
                    raise e
                    
        except Exception:
            # Collection doesn't exist, create it
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings for semantic search"}
            )
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add document chunks with embeddings to the vector store.
        
        Args:
            chunks: List of chunks with embeddings and metadata
            
        Returns:
            Dictionary with operation results
        """
        if not chunks:
            return {"status": "error", "message": "No chunks provided"}
        
        # Check embedding dimension and recreate collection if needed
        if chunks and "embedding" in chunks[0]:
            expected_dimension = len(chunks[0]["embedding"])
            self.collection = self._recreate_collection_if_dimension_mismatch(expected_dimension)
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            # Generate unique ID if not present
            chunk_id = chunk.get("id", str(uuid.uuid4()))
            ids.append(chunk_id)
            
            # Extract embedding
            if "embedding" not in chunk:
                raise ValueError(f"Chunk {chunk_id} missing embedding")
            embeddings.append(chunk["embedding"])
            
            # Extract document text
            documents.append(chunk["text"])
            
            # Prepare metadata (exclude embedding and text)
            metadata = {k: v for k, v in chunk.items() 
                       if k not in ["id", "text", "embedding"]}
            metadatas.append(metadata)
        
        try:
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            return {
                "status": "success",
                "message": f"Added {len(chunks)} chunks to vector store",
                "chunks_added": len(chunks)
            }
            
        except Exception as e:
            # If still getting dimension error, try recreating collection
            if "dimension" in str(e).lower():
                try:
                    print("Attempting to recreate collection due to dimension mismatch...")
                    self.client.delete_collection(name=self.collection_name)
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata={"description": "Document embeddings for semantic search"}
                    )
                    
                    # Retry adding documents
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas
                    )
                    
                    return {
                        "status": "success",
                        "message": f"Added {len(chunks)} chunks to vector store (collection recreated)",
                        "chunks_added": len(chunks)
                    }
                except Exception as retry_error:
                    return {
                        "status": "error",
                        "message": f"Failed to add chunks after collection recreation: {str(retry_error)}"
                    }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to add chunks: {str(e)}"
                }
    
    def search_similar(self, query_embedding: List[float], n_results: int = 5, 
                      where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            List of similar documents with metadata
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
            
            # Format results
            similar_docs = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    doc = {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else None
                    }
                    similar_docs.append(doc)
            
            return similar_docs
            
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]
    
    def search_by_text(self, query_text: str, n_results: int = 5, 
                      where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using text query.
        
        Args:
            query_text: Text query to search for
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            List of similar documents with metadata
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where
            )
            
            # Format results
            similar_docs = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    doc = {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else None
                    }
                    similar_docs.append(doc)
            
            return similar_docs
            
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "status": "active"
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "status": "error",
                "error": str(e)
            }
    
    def delete_documents(self, document_name: str) -> Dict[str, Any]:
        """
        Delete all documents with a specific document name.
        
        Args:
            document_name: Name of the document to delete
            
        Returns:
            Dictionary with operation results
        """
        try:
            # Get all documents with the specified document name
            results = self.collection.get(
                where={"document_name": document_name}
            )
            
            if results["ids"]:
                # Delete the documents
                self.collection.delete(ids=results["ids"])
                return {
                    "status": "success",
                    "message": f"Deleted {len(results['ids'])} chunks for document '{document_name}'"
                }
            else:
                return {
                    "status": "info",
                    "message": f"No documents found with name '{document_name}'"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to delete documents: {str(e)}"
            }
    
    def clear_collection(self) -> Dict[str, Any]:
        """
        Clear all documents from the collection.
        
        Returns:
            Dictionary with operation results
        """
        try:
            # Get all document IDs
            results = self.collection.get()
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                return {
                    "status": "success",
                    "message": f"Cleared {len(results['ids'])} documents from collection"
                }
            else:
                return {
                    "status": "info",
                    "message": "Collection is already empty"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to clear collection: {str(e)}"
            }
