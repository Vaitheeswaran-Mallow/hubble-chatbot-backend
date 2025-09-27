from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Any
import asyncio
from ..config import config


class EmbeddingService:
    """Handles text embedding generation using OpenAI's embedding model."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initialize the embedding service.
        
        Args:
            model_name: OpenAI embedding model to use
        """
        self.embeddings = OpenAIEmbeddings(
            api_key=config.open_api_key,
            model=model_name
        )
        self.model_name = model_name
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of float values representing the embedding
        """
        return self.embeddings.embed_query(text)
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        return self.embeddings.embed_documents(texts)
    
    async def generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings asynchronously for better performance.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        # Run the synchronous embedding generation in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.generate_embeddings_batch, 
            texts
        )
    
    def process_chunks_with_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process chunks and add embeddings to each chunk.
        
        Args:
            chunks: List of text chunks with metadata
            
        Returns:
            List of chunks with embeddings added
        """
        # Extract texts for batch embedding
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings for all texts
        embeddings = self.generate_embeddings_batch(texts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]
            chunk["embedding_model"] = self.model_name
        
        return chunks
    
    async def process_chunks_with_embeddings_async(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process chunks and add embeddings asynchronously.
        
        Args:
            chunks: List of text chunks with metadata
            
        Returns:
            List of chunks with embeddings added
        """
        # Extract texts for batch embedding
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings for all texts asynchronously
        embeddings = await self.generate_embeddings_async(texts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]
            chunk["embedding_model"] = self.model_name
        
        return chunks
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Dimension of the embedding vectors
        """
        # Generate a test embedding to determine dimension
        test_embedding = self.generate_embedding("test")
        return len(test_embedding)
