from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, Optional
from ..config import config
from .database_service import db_service
from .question_classifier import question_classifier
import json
import logging

logger = logging.getLogger(__name__)


class AnswerService:
    """Handles answer generation by combining document search with LLM responses."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1):
        """
        Initialize the answer service.
        
        Args:
            model_name: OpenAI model to use for answer generation
            temperature: Temperature for response generation (0.0 = deterministic, 1.0 = creative)
        """
        self.llm = ChatOpenAI(
            api_key=config.open_api_key,
            model=model_name,
            temperature=temperature
        )
        self.model_name = model_name
    
    def create_context_prompt(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Create a context-aware prompt for the LLM.
        
        Args:
            query: User's question
            search_results: Relevant document chunks from search
            
        Returns:
            Formatted prompt for the LLM
        """
        # Extract relevant text from search results
        context_parts = []
        sources = []
        
        for i, result in enumerate(search_results, 1):
            text = result.get('text', '')
            metadata = result.get('metadata', {})
            document_name = metadata.get('document_name', 'Unknown Document')
            chunk_index = metadata.get('chunk_index', 0)
            
            context_parts.append(f"[Source {i}] {text}")
            sources.append({
                "source": i,
                "document": document_name,
                "chunk": chunk_index,
                "similarity": 1 - result.get('distance', 0) if 'distance' in result else None
            })
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are a helpful assistant that answers questions based on the provided document context. 
Use only the information from the provided context to answer the user's question. Your bot that answers questions related Mallow Technologies. Ignore other questions.

Context from documents:
{context}

User Question: {query}

Instructions:
1. Answer the question based ONLY on the information provided in the context above
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question based on the available documents"
3. Be specific and cite relevant details from the context
4. If you reference information, mention which source it came from (e.g., "According to Source 1...")
5. Keep your answer concise but comprehensive
6. If the question is about policies, procedures, or rules, be precise about the specific requirements

Answer:"""
        
        return prompt, sources
    
    def generate_answer(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an answer based on search results.
        
        Args:
            query: User's question
            search_results: Relevant document chunks from search
            
        Returns:
            Dictionary containing the answer and metadata
        """
        if not search_results:
            return {
                "answer": "I don't have any relevant information to answer your question. Please try rephrasing your question or check if the document has been processed.",
                "sources": [],
                "confidence": "low",
                "model_used": self.model_name,
                "search_results_count": 0
            }
        
        # Create context-aware prompt
        prompt, sources = self.create_context_prompt(query, search_results)
        
        try:
            # Generate answer using LLM
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Calculate confidence based on search result quality
            confidence = self._calculate_confidence(search_results)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "model_used": self.model_name,
                "search_results_count": len(search_results)
            }
            
        except Exception as e:
            return {
                "answer": f"I encountered an error while generating an answer: {str(e)}",
                "sources": sources,
                "confidence": "error",
                "model_used": self.model_name,
                "error": str(e)
            }
    
    def _calculate_confidence(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Calculate confidence level based on search result quality.
        
        Args:
            search_results: List of search results
            
        Returns:
            Confidence level: "high", "medium", or "low"
        """
        if not search_results:
            return "low"
        
        # Check similarity scores if available
        distances = [result.get('distance', 1.0) for result in search_results if 'distance' in result]
        
        if distances:
            avg_distance = sum(distances) / len(distances)
            avg_similarity = 1 - avg_distance
            
            if avg_similarity > 0.8:
                return "high"
            elif avg_similarity > 0.6:
                return "medium"
            else:
                return "low"
        
        # If no distance information, base on number of results
        if len(search_results) >= 3:
            return "medium"
        elif len(search_results) >= 1:
            return "low"
        else:
            return "low"
    
    def generate_answer_with_sources(self, query: str, search_results: List[Dict[str, Any]], 
                                   include_full_context: bool = False) -> Dict[str, Any]:
        """
        Generate answer with detailed source information.
        
        Args:
            query: User's question
            search_results: Relevant document chunks from search
            include_full_context: Whether to include full context in response
            
        Returns:
            Dictionary containing answer and detailed source information
        """
        answer_data = self.generate_answer(query, search_results)
        
        # Add detailed source information
        detailed_sources = []
        for i, result in enumerate(search_results, 1):
            metadata = result.get('metadata', {})
            detailed_sources.append({
                "source_id": i,
                "document_name": metadata.get('document_name', 'Unknown'),
                "chunk_index": metadata.get('chunk_index', 0),
                "text_preview": result.get('text', '')[:200] + "..." if len(result.get('text', '')) > 200 else result.get('text', ''),
                "similarity_score": 1 - result.get('distance', 0) if 'distance' in result else None,
                "token_count": metadata.get('token_count', 0)
            })
        
        answer_data["detailed_sources"] = detailed_sources
        
        if include_full_context:
            answer_data["full_context"] = [result.get('text', '') for result in search_results]
        
        return answer_data
    
    def generate_follow_up_questions(self, query: str, search_results: List[Dict[str, Any]]) -> List[str]:
        """
        Generate follow-up questions based on the context.
        
        Args:
            query: Original user question
            search_results: Search results used for the answer
            
        Returns:
            List of suggested follow-up questions
        """
        if not search_results:
            return []
        
        # Extract key topics from search results
        context_text = " ".join([result.get('text', '') for result in search_results[:3]])
        
        follow_up_prompt = f"""Based on the following context about policies and procedures, suggest 3 relevant follow-up questions that a user might want to ask:

Context: {context_text[:1000]}

Original Question: {query}

Generate 3 specific, relevant follow-up questions that would help the user understand more about the policies mentioned. Make them practical and useful.

Format as a JSON array of strings:"""
        
        try:
            response = self.llm.invoke(follow_up_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse JSON response
            try:
                follow_ups = json.loads(response_text)
                if isinstance(follow_ups, list) and len(follow_ups) >= 1:
                    return follow_ups[:3]
            except json.JSONDecodeError:
                pass
            
            # Fallback: return generic follow-ups
            return [
                f"Can you tell me more about the specific requirements for {query.lower()}?",
                "What are the exceptions or special cases for this policy?",
                "How do I apply or request this policy?"
            ]
            
        except Exception:
            return [
                f"Can you provide more details about {query.lower()}?",
                "What are the specific requirements for this?",
                "How do I proceed with this policy?"
            ]
    
    def classify_and_answer(self, query: str, search_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Classify question and generate appropriate answer (database or document).
        
        Args:
            query: User's question
            search_results: Document search results (if available)
            
        Returns:
            Dictionary containing the answer and metadata
        """
        # Classify the question
        classification = question_classifier.classify_question(query)
        
        if classification["query_type"] == "database":
            return self._generate_database_answer(query, classification)
        else:
            return self._generate_document_answer(query, search_results or [])
    
    def _generate_database_answer(self, query: str, classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate answer based on database query.
        
        Args:
            query: User's question
            classification: Question classification results
            
        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            # Generate SQL query using LLM
            sql_query = self._generate_sql_query(query, classification)
            
            if not sql_query:
                return {
                    "answer": "I couldn't generate a suitable database query for your question. Please try rephrasing your question or ask about specific data like employee information, leave balances, or project details.",
                    "sources": [],
                    "confidence": "low",
                    "model_used": self.model_name,
                    "query_type": "database",
                    "search_results_count": 0
                }
            
            # Execute database query
            db_results = db_service.execute_query(sql_query)
            
            if not db_results:
                return {
                    "answer": "No data found matching your query. Please check if the information exists or try a different search term.",
                    "sources": [],
                    "confidence": "medium",
                    "model_used": self.model_name,
                    "query_type": "database",
                    "search_results_count": 0
                }
            
            # Generate human-readable answer from database results
            answer = self._format_database_results(query, db_results, classification)
            
            return {
                "answer": answer,
                "sources": [{"source": "database", "query": sql_query, "result_count": len(db_results)}],
                "confidence": classification.get("confidence", "medium"),
                "model_used": self.model_name,
                "query_type": "database",
                "search_results_count": len(db_results),
                "raw_data": db_results[:5]  # Include first 5 results for debugging
            }
            
        except Exception as e:
            logger.error(f"Database query error: {str(e)}")
            return {
                "answer": f"I encountered an error while querying the database: {str(e)}. Please try rephrasing your question or contact support.",
                "sources": [],
                "confidence": "error",
                "model_used": self.model_name,
                "query_type": "database",
                "error": str(e)
            }
    
    def _generate_document_answer(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate answer based on document search (existing functionality).
        
        Args:
            query: User's question
            search_results: Document search results
            
        Returns:
            Dictionary containing the answer and metadata
        """
        return self.generate_answer(query, search_results)
    
    def _generate_sql_query(self, query: str, classification: Dict[str, Any]) -> Optional[str]:
        """
        Generate SQL query using LLM based on the question and classification.
        
        Args:
            query: User's question
            classification: Question classification results
            
        Returns:
            SQL query string or None if generation fails
        """
        schema_info = db_service.get_schema_info()
        suggested_tables = classification.get("suggested_tables", [])
        
        prompt = f"""You are an expert SQL query generator for a company HR and project management database.
Generate a precise SQL query to answer the user's question.

Database Schema:
{schema_info}

User Question: "{query}"

Suggested Tables: {suggested_tables}

CRITICAL INSTRUCTIONS:
1. Generate ONLY a SQL query - no explanations, no markdown, no code blocks
2. Use proper SQLite syntax (not PostgreSQL)
3. Use LIKE for text searches (SQLite doesn't support ILIKE)
4. Use single quotes for string literals
6. Use appropriate JOINs to get complete information
7. Handle NULL values properly with COALESCE if needed
8. For user searches, try multiple fields: name, employee_id, email
9. Use proper date formatting for SQLite
10. Make sure the query is syntactically correct


Examples of good queries:
- SELECT u.name, u.employee_id, d.name as designation FROM users u LEFT JOIN designations d ON u.designation_id = d.id WHERE u.name LIKE '%john%';
- SELECT * FROM leaves WHERE user_id = (SELECT id FROM users WHERE name LIKE '%sarah%');
- SELECT name, date_of_holiday FROM holidays WHERE date_of_holiday >= '2024-01-01';

SQL Query:"""
        
        try:
            response = self.llm.invoke(prompt)
            sql_query = response.content if hasattr(response, 'content') else str(response)
            
            # Clean up the response - remove any markdown or code blocks
            sql_query = sql_query.strip()
            
            # Remove markdown code blocks
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            elif sql_query.startswith("```"):
                sql_query = sql_query[3:]
            
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            
            sql_query = sql_query.strip()
            
            # Remove any leading/trailing quotes or backticks
            sql_query = sql_query.strip('"\'`')
            
            # Basic validation
            if not sql_query.upper().startswith("SELECT"):
                logger.warning(f"Generated query doesn't start with SELECT: {sql_query}")
                return None
            
            # Ensure LIMIT is present for safety
            # if "LIMIT" not in sql_query.upper():
            #     sql_query += " LIMIT 20"
            
            logger.info(f"Generated SQL query: {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"SQL generation error: {str(e)}")
            return None
    
    def _format_database_results(self, query: str, results: List[Dict[str, Any]], classification: Dict[str, Any]) -> str:
        """
        Format database results into a human-readable answer.
        
        Args:
            query: Original user question
            results: Database query results
            classification: Question classification results
            
        Returns:
            Formatted answer string
        """
        if not results:
            return "No data found matching your query."
        
        # Create context for LLM to format the results
        results_text = json.dumps(results[:10], indent=2)  # Limit to first 10 results
        
        prompt = f"""You are a helpful assistant that formats database query results into a clear, human-readable answer.

User Question: "{query}"

Database Results:
{results_text}

Instructions:
1. Format the data in a clear, organized way
2. Highlight key information that answers the user's question
3. If there are multiple results, summarize them appropriately
4. Output should be a readable string without no markdown or code blocks.
5. Be concise but informative
6. If the data contains sensitive information, be careful about what you include

Formatted Answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Result formatting error: {str(e)}")
            # Fallback: simple formatting
            return f"Found {len(results)} result(s):\n" + "\n".join([str(result) for result in results[:5]])
