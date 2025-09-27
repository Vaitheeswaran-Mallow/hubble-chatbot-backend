from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import tiktoken


class DocumentProcessor:
    """Handles document processing and text chunking for embeddings."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        # Initialize tokenizer for accurate token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def split_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks and return metadata for each chunk.
        
        Args:
            text: The text to split
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        chunks = self.text_splitter.split_text(text)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Count tokens for this chunk
            token_count = len(self.tokenizer.encode(chunk))
            
            processed_chunks.append({
                "id": f"chunk_{i}",
                "text": chunk,
                "chunk_index": i,
                "token_count": token_count,
                "char_count": len(chunk)
            })
        
        return processed_chunks
    
    def process_document(self, document_text: str, document_name: str) -> List[Dict[str, Any]]:
        """
        Process a complete document and return all chunks with metadata.
        
        Args:
            document_text: The full text of the document
            document_name: Name/identifier of the document
            
        Returns:
            List of processed chunks with metadata
        """
        chunks = self.split_text(document_text)
        
        # Add document-level metadata to each chunk and create unique IDs
        for i, chunk in enumerate(chunks):
            # Create unique ID that includes document name
            chunk["id"] = f"{document_name}_chunk_{i}"
            chunk["document_name"] = document_name
            chunk["source"] = document_name
        
        return chunks
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the processed chunks.
        
        Args:
            chunks: List of processed chunks
            
        Returns:
            Dictionary containing chunk statistics
        """
        if not chunks:
            return {"total_chunks": 0, "total_tokens": 0, "total_characters": 0}
        
        total_tokens = sum(chunk["token_count"] for chunk in chunks)
        total_characters = sum(chunk["char_count"] for chunk in chunks)
        
        return {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "total_characters": total_characters,
            "average_tokens_per_chunk": total_tokens / len(chunks),
            "average_characters_per_chunk": total_characters / len(chunks)
        }
