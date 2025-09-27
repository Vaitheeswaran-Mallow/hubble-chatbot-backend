from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, Optional
from settings import config
import json


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
Use only the information from the provided context to answer the user's question.

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
                "model_used": self.model_name
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
