import os
import openai
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import uuid
from data_processor import EndorsementDataProcessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EndorsementRAGSystem:
    """RAG system for querying endorsement processes using OpenAI and ChromaDB"""
    
    def __init__(self, openai_api_key: str, excel_file_path: str):
        self.openai_api_key = openai_api_key
        self.excel_file_path = excel_file_path
        
        # Initialize OpenAI client
        openai.api_key = openai_api_key
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Initialize data processor
        self.data_processor = EndorsementDataProcessor(excel_file_path)
        
        # Collection for storing embeddings
        self.collection_name = "endorsement_processes"
        self.collection = None
        
    def initialize_system(self):
        """Initialize the RAG system by processing data and creating embeddings"""
        logger.info("Initializing RAG system...")
        
        # Load and process data
        logger.info("Loading and processing Excel data...")
        self.data_processor.load_data()
        processed_data = self.data_processor.clean_data()
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            logger.info(f"Found existing collection: {self.collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Endorsement process documents"},
                embedding_function=None  # We'll use OpenAI embeddings directly
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        # Generate embeddings and store in ChromaDB
        logger.info("Generating embeddings and storing in vector database...")
        self._create_embeddings(processed_data)
        
        logger.info("RAG system initialized successfully!")
        return True
    
    def _create_embeddings(self, processed_data: List[Dict[str, Any]]):
        """Create embeddings for all processes and store in ChromaDB"""
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        
        for process in processed_data:
            # Use the full text for embedding
            documents.append(process['full_text'])
            
            # Generate embedding
            embedding = self._get_embedding(process['full_text'])
            embeddings.append(embedding)
            
            # Store metadata
            metadata = {
                'id': process['id'],
                'management_of': process['management_of'],
                'process_type': process['metadata']['process_type'],
                'has_email_template': process['metadata']['has_email_template'],
                'actions': process['actions'][:500],  # Truncate for metadata
                'initiator': process['initiator'][:200],
                'approver': process['approver'][:200],
                'executer': process['executer'][:200]
            }
            metadatas.append(metadata)
            ids.append(str(process['id']))
        
        # Add documents to collection with embeddings
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        logger.info(f"Added {len(documents)} documents to vector database")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=1536  # Specify dimensions explicitly
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            return []
    
    def search_similar_processes(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar processes using vector similarity"""
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            if not query_embedding:
                return []
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            similar_processes = []
            for i in range(len(results['documents'][0])):
                similar_processes.append({
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'id': results['metadatas'][0][i]['id']
                })
            
            return similar_processes
            
        except Exception as e:
            logger.error(f"Error searching similar processes: {str(e)}")
            return []
    
    def generate_answer(self, query: str, context_processes: List[Dict[str, Any]]) -> str:
        """Generate answer using OpenAI based on retrieved context"""
        try:
            # Prepare context from retrieved processes
            context_text = ""
            for i, process in enumerate(context_processes[:3]):  # Use top 3 processes
                context_text += f"\nProcess {i+1}:\n{process['document']}\n"
            
            # Create prompt for OpenAI
            prompt = f"""You are an AI assistant that helps answer questions about endorsement processes based on the provided context.

Context from endorsement process documents:
{context_text}

Question: {query}

Please provide a helpful answer based on the context above. If the context doesn't contain enough information to answer the question, please say so. Focus on:
1. Who needs to initiate the request
2. Who needs to approve it
3. Who executes the action
4. What confirmation is needed
5. Any specific email templates or procedures

Answer:"""

            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions about endorsement processes."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Sorry, I encountered an error while generating the answer: {str(e)}"
    
    def query(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """Main query method that combines retrieval and generation"""
        logger.info(f"Processing query: {question}")
        
        # Search for similar processes
        similar_processes = self.search_similar_processes(question, n_results)
        
        if not similar_processes:
            return {
                'answer': "I couldn't find any relevant processes for your question. Please try rephrasing your question or ask about a different topic.",
                'similar_processes': [],
                'context_used': []
            }
        
        # Generate answer based on retrieved context
        answer = self.generate_answer(question, similar_processes)
        
        # Prepare response
        response = {
            'answer': answer,
            'similar_processes': similar_processes,
            'context_used': [p['document'] for p in similar_processes[:3]]
        }
        
        return response
    
    def get_process_by_id(self, process_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific process by ID"""
        try:
            results = self.collection.get(
                ids=[str(process_id)],
                include=['documents', 'metadatas']
            )
            
            if results['documents']:
                return {
                    'id': process_id,
                    'document': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting process by ID: {str(e)}")
            return None
    
    def get_all_processes(self) -> List[Dict[str, Any]]:
        """Get all processes from the collection"""
        try:
            results = self.collection.get(
                include=['documents', 'metadatas']
            )
            
            processes = []
            for i in range(len(results['documents'])):
                processes.append({
                    'id': results['metadatas'][i]['id'],
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
            
            return processes
            
        except Exception as e:
            logger.error(f"Error getting all processes: {str(e)}")
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            collection_count = self.collection.count()
            data_stats = self.data_processor.get_summary_stats()
            
            return {
                'vector_db_documents': collection_count,
                'data_stats': data_stats,
                'collection_name': self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {}


if __name__ == "__main__":
    # Test the RAG system
    import os
    
    # You'll need to set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    
    if api_key == "your-api-key-here":
        print("Please set your OPENAI_API_KEY environment variable")
    else:
        rag_system = EndorsementRAGSystem(
            openai_api_key=api_key,
            excel_file_path='/Users/yuvaraj/Downloads/Mallow Hackathon 2025/Endorsement Process Excel.xlsx'
        )
        
        # Initialize the system
        rag_system.initialize_system()
        
        # Test query
        test_query = "How do I request access to Bitbucket?"
        result = rag_system.query(test_query)
        
        print(f"Query: {test_query}")
        print(f"Answer: {result['answer']}")
        print(f"Found {len(result['similar_processes'])} similar processes")
