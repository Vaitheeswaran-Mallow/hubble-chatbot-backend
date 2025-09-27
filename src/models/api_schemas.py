from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class AskQuestion(BaseModel):
    question: str
    n_results: int = 3


class QueryRequest(BaseModel):
    question: str
    n_results: int = 5


class QueryResponse(BaseModel):
    answer: str
    similar_processes: List[Dict[str, Any]]
    context_used: List[str]


class ProcessResponse(BaseModel):
    id: int
    document: str
    metadata: Dict[str, Any]


class SystemStatsResponse(BaseModel):
    vector_db_documents: int
    data_stats: Dict[str, Any]
    collection_name: str


class CategoryResponse(BaseModel):
    categories: List[str]


class ProcessesByCategoryResponse(BaseModel):
    category: str
    processes: List[Dict[str, Any]]
    count: int
