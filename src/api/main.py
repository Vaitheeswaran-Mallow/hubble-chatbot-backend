from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any
import os
import logging
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ..config import config
from ..services.rag_system import EndorsementRAGSystem
from ..models.api_schemas import (
    QueryRequest, QueryResponse, ProcessResponse, 
    SystemStatsResponse, CategoryResponse, ProcessesByCategoryResponse
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=config.api_title,
    description=config.api_description,
    version=config.api_version
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global RAG system instance
rag_system: Optional[EndorsementRAGSystem] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global rag_system
    
    try:
        logger.info("Initializing RAG system...")
        
        # Check if OpenAI API key is available
        if not config.open_api_key or config.open_api_key == "your-api-key-here":
            logger.error("OpenAI API key not configured. Please set OPEN_API_KEY environment variable.")
            return
        
        # Initialize RAG system
        rag_system = EndorsementRAGSystem(
            openai_api_key=config.open_api_key,
            excel_file_path=config.excel_file_path
        )
        
        # Initialize the system
        rag_system.initialize_system()
        
        logger.info("RAG system initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        rag_system = None

@app.get("/")
def hello_world():
    return {
        "message": "Hubble Chatbot Backend - Endorsement Process RAG",
        "status": "running",
        "rag_system_initialized": rag_system is not None
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_system_initialized": rag_system is not None,
        "openai_configured": bool(config.open_api_key and config.open_api_key != "your-api-key-here")
    }

@app.post("/query", response_model=QueryResponse)
async def query_endorsement_processes(request: QueryRequest):
    """Query the endorsement process RAG system"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = rag_system.query(request.question, request.n_results)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/query")
async def query_endorsement_processes_get(
    question: str = Query(..., description="The question to ask about endorsement processes"),
    n_results: int = Query(5, description="Number of similar processes to retrieve")
):
    """Query the endorsement process RAG system (GET method)"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = rag_system.query(question, n_results)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/process/{process_id}", response_model=ProcessResponse)
async def get_process_by_id(process_id: int):
    """Get a specific endorsement process by ID"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        process = rag_system.get_process_by_id(process_id)
        if not process:
            raise HTTPException(status_code=404, detail=f"Process with ID {process_id} not found")
        return ProcessResponse(**process)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting process by ID: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting process: {str(e)}")

@app.get("/processes", response_model=List[ProcessResponse])
async def get_all_processes():
    """Get all endorsement processes"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        processes = rag_system.get_all_processes()
        return [ProcessResponse(**process) for process in processes]
    except Exception as e:
        logger.error(f"Error getting all processes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting processes: {str(e)}")

@app.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    """Get system statistics"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        stats = rag_system.get_system_stats()
        return SystemStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/categories")
async def get_process_categories():
    """Get all available process categories"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        categories = rag_system.data_processor.get_all_categories()
        return {"categories": categories}
    except Exception as e:
        logger.error(f"Error getting categories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting categories: {str(e)}")

@app.get("/processes/category/{category}")
async def get_processes_by_category(category: str):
    """Get all processes of a specific category"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        processes = rag_system.data_processor.get_processes_by_category(category)
        return {"category": category, "processes": processes, "count": len(processes)}
    except Exception as e:
        logger.error(f"Error getting processes by category: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting processes by category: {str(e)}")

@app.get("/config")
def get_config():
    return {
        "open_api_key_configured": bool(config.open_api_key and config.open_api_key != "your-api-key-here"),
        "rag_system_initialized": rag_system is not None
    }
