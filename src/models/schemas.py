"""
Pydantic models for API request and response schemas
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    """Request model for querying the RAG system"""
    question: str
    n_results: Optional[int] = 5


class QueryResponse(BaseModel):
    """Response model for RAG system queries"""
    answer: str
    similar_processes: List[Dict[str, Any]]
    context_used: List[str]


class ProcessResponse(BaseModel):
    """Response model for individual process data"""
    id: int
    document: str
    metadata: Dict[str, Any]


class SystemStatsResponse(BaseModel):
    """Response model for system statistics"""
    vector_db_documents: int
    data_stats: Dict[str, Any]
    collection_name: str


class ProcessMetadata(BaseModel):
    """Model for process metadata"""
    id: int
    management_of: str
    process_type: str
    has_email_template: bool
    actions: str
    initiator: str
    approver: str
    executer: str


class ProcessData(BaseModel):
    """Model for complete process data"""
    id: int
    management_of: str
    actions: str
    initiator: str
    approver: str
    executer: str
    post_execution_confirmation: str
    email_details: str
    full_text: str
    metadata: ProcessMetadata


class CategoryResponse(BaseModel):
    """Response model for process categories"""
    categories: List[str]


class ProcessesByCategoryResponse(BaseModel):
    """Response model for processes filtered by category"""
    category: str
    processes: List[ProcessData]
    count: int
