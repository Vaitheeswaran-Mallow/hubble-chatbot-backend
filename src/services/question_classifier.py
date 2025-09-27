"""
Question classifier service to determine if a question requires database or document search
"""

from langchain_openai import ChatOpenAI
from typing import Dict, Any, List
from ..config import config
import json
import logging

logger = logging.getLogger(__name__)


class QuestionClassifier:
    """Service for classifying questions as database or document queries"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1):
        """
        Initialize the question classifier.
        
        Args:
            model_name: OpenAI model to use for classification
            temperature: Temperature for response generation
        """
        self.llm = ChatOpenAI(
            api_key=config.open_api_key,
            model=model_name,
            temperature=temperature
        )
        self.model_name = model_name
    
    def classify_question(self, question: str) -> Dict[str, Any]:
        """
        Classify a question to determine if it requires database or document search.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing classification results
        """
        prompt = self._create_classification_prompt(question)
        
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse JSON response
            try:
                classification = json.loads(response_text)
                return self._validate_classification(classification)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response, using fallback classification")
                return self._fallback_classification(question)
                
        except Exception as e:
            logger.error(f"Error in question classification: {str(e)}")
            return self._fallback_classification(question)
    
    def _create_classification_prompt(self, question: str) -> str:
        """Create prompt for question classification"""
        return f"""You are a question classifier for a company HR and project management system. 
Your job is to determine whether a user's question requires querying the database or searching through documents.

Database queries are needed for questions about:
- Employee information (names, IDs, roles, teams, departments)
- Leave balances, leave history, time-off requests
- Project assignments, timesheet entries, work hours
- Team members, reporting structures
- Company holidays, work schedules
- Personal details, contact information
- Project status, assignments, deadlines
- Any specific data that would be stored in database tables

Document queries are needed for questions about:
- Company policies, procedures, guidelines
- HR policies, leave policies, work from home policies
- Rules and regulations, office etiquette
- Process documentation, workflows
- General information that would be in policy documents
- How-to guides, procedures, best practices

Question: "{question}"

Analyze this question and respond with a JSON object containing:
{{
    "query_type": "database" or "document",
    "confidence": "high", "medium", or "low",
    "reasoning": "Brief explanation of why this classification was chosen",
    "suggested_tables": ["list of relevant database tables if query_type is database"],
    "suggested_document_types": ["list of relevant document types if query_type is document"]
}}

Respond only with the JSON object:"""
    
    def _validate_classification(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean classification results"""
        # Ensure required fields exist
        if "query_type" not in classification:
            classification["query_type"] = "document"
        
        if "confidence" not in classification:
            classification["confidence"] = "medium"
        
        if "reasoning" not in classification:
            classification["reasoning"] = "Classification completed"
        
        # Validate query_type
        if classification["query_type"] not in ["database", "document"]:
            classification["query_type"] = "document"
        
        # Validate confidence
        if classification["confidence"] not in ["high", "medium", "low"]:
            classification["confidence"] = "medium"
        
        # Ensure suggested fields exist
        if "suggested_tables" not in classification:
            classification["suggested_tables"] = []
        
        if "suggested_document_types" not in classification:
            classification["suggested_document_types"] = []
        
        return classification
    
    def _fallback_classification(self, question: str) -> Dict[str, Any]:
        """Fallback classification when LLM fails"""
        # Simple keyword-based classification as fallback
        db_keywords = [
            "employee", "user", "leave", "timesheet", "project", "team", 
            "balance", "history", "assignment", "hours", "holiday", "schedule",
            "reporting", "manager", "designation", "branch", "status"
        ]
        
        doc_keywords = [
            "policy", "procedure", "guideline", "rule", "regulation", 
            "process", "workflow", "how to", "what is", "explain"
        ]
        
        question_lower = question.lower()
        
        db_score = sum(1 for keyword in db_keywords if keyword in question_lower)
        doc_score = sum(1 for keyword in doc_keywords if keyword in question_lower)
        
        if db_score > doc_score:
            return {
                "query_type": "database",
                "confidence": "low",
                "reasoning": "Fallback classification based on keywords",
                "suggested_tables": [],
                "suggested_document_types": []
            }
        else:
            return {
                "query_type": "document",
                "confidence": "low",
                "reasoning": "Fallback classification based on keywords",
                "suggested_tables": [],
                "suggested_document_types": []
            }
    
    def classify_batch(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple questions at once.
        
        Args:
            questions: List of questions to classify
            
        Returns:
            List of classification results
        """
        return [self.classify_question(question) for question in questions]


# Global question classifier instance
question_classifier = QuestionClassifier()
