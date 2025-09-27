from fastapi import FastAPI, Depends, HTTPException
from settings import config
from langchain_openai import OpenAI
from db import get_chroma_client
from markitdown import MarkItDown
from document_processor import DocumentProcessor
from embedding_service import EmbeddingService
from vector_store import VectorStore
from answer_service import AnswerService
from schemas import AskQuestion
from fastapi.responses import FileResponse
import asyncio
from ws import router, manager
from ws.upstream import receiver_loop
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import date
from typing import Optional
import sqlite3

app = FastAPI(title="Hubble Chatbot Backend")

def get_markdown_converter():
    """Initialize and return a Markitdown instance."""
    return MarkItDown(enable_plugins=False)

def get_document_processor():
    """Initialize and return a DocumentProcessor instance."""
    return DocumentProcessor()

def get_embedding_service():
    """Initialize and return an EmbeddingService instance."""
    return EmbeddingService()

def get_vector_store():
    """Initialize and return a VectorStore instance."""
    return VectorStore()

def get_answer_service():
    """Initialize and return an AnswerService instance."""
    return AnswerService()


def extract_markdown_content(conversion_result):
    """Extract markdown content from MarkItDown conversion result."""
    if hasattr(conversion_result, 'text_content'):
        return conversion_result.text_content
    elif hasattr(conversion_result, 'markdown'):
        return conversion_result.markdown
    else:
        # If it's already a string, use it directly
        return str(conversion_result)


@app.on_event("startup")
async def startup_tasks():
    if config.ws_upstream_url:
        asyncio.create_task(receiver_loop(config.ws_upstream_url, manager))


def get_db():
    # Support both sqlite:/// and sqlite: URLs; strip prefix to path
    url = config.database_url
    if url.startswith("sqlite:///"):
        path = url.replace("sqlite:///", "", 1)
    elif url.startswith("sqlite:"):
        path = url.replace("sqlite:", "", 1)
    else:
        # Fall back to treating as a direct file path
        path = url
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    # Enforce foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


class TimesheetEntryCreate(BaseModel):
    user_id: int = Field(..., gt=0)
    project_id: int = Field(..., gt=0)
    module_id: int = Field(..., gt=0)
    task_id: int = Field(..., gt=0)
    team_id: int = Field(..., gt=0)
    description: str = Field(..., min_length=1)
    entry_date: date
    working_hours: float = Field(..., gt=0)
    approved_hours: Optional[float] = Field(default=0, ge=0)
    authorized_hours: Optional[float] = Field(default=0, ge=0)
    billed_hours: Optional[float] = Field(default=0, ge=0)
    admin_comments: Optional[str] = None

    @field_validator("approved_hours", "authorized_hours", "billed_hours", mode="before")
    @classmethod
    def default_zero_for_missing(cls, v):
        return 0 if v is None or v == "" else v

    @field_validator("working_hours", "approved_hours", "authorized_hours", "billed_hours")
    @classmethod
    def non_negative(cls, v):
        if v is None:
            return v
        if v < 0:
            raise ValueError("must be >= 0")
        return v

    @model_validator(mode="after")
    def validate_hours_monotonic(self):
        if self.working_hours is not None and self.working_hours > 24:
            raise ValueError("working_hours cannot exceed 24 hours for a single entry")
        if self.approved_hours is not None and self.approved_hours > self.working_hours:
            raise ValueError("approved_hours cannot exceed working_hours")
        if self.authorized_hours is not None and self.approved_hours is not None and self.authorized_hours > self.approved_hours:
            raise ValueError("authorized_hours cannot exceed approved_hours")
        if self.billed_hours is not None and self.approved_hours is not None and self.billed_hours > self.approved_hours:
            raise ValueError("billed_hours cannot exceed approved_hours")
        return self


@app.get("/")
def hello_world():
    return {"message": "Hello, World!"}


@app.get("/config")
def get_config():
    return {"open_api_key": config.open_api_key}


@app.get("/llm-test")
def llm_test():
    llm = OpenAI(api_key=config.open_api_key, temperature=0)
    response = llm.invoke("What is the capital of India?")
    return {"response": response}


@app.get("/db-test")
def db_test(chroma_db=Depends(get_chroma_client)):
    if chroma_db.heartbeat():
        return {"status": "ChromaDB is reachable"}
    else:
        return {"status": "ChromaDB is not reachable"}

@app.get("/convert-to-markdown/")
def convert_to_markdown(md=Depends(get_markdown_converter)):
    conversion_result = md.convert("./docs/New Leave policy.pdf")
    markdown_content = extract_markdown_content(conversion_result)
    return {"markdown": markdown_content}


@app.post("/process-document/")
async def process_document(
    document_path: str,
    document_name: str = None,
    md=Depends(get_markdown_converter),
    processor=Depends(get_document_processor),
    embedding_service=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store)
):
    """
    Process a document: convert to markdown, chunk, embed, and store in vector database.

    Args:
        document_path: Path to the document file
        document_name: Optional name for the document (defaults to filename)
    """
    try:
        # Convert document to markdown
        conversion_result = md.convert(document_path)
        markdown_content = extract_markdown_content(conversion_result)

        # Use filename as document name if not provided
        if not document_name:
            import os
            document_name = os.path.basename(document_path)

        # Process document into chunks
        chunks = processor.process_document(markdown_content, document_name)

        # Generate embeddings for chunks
        chunks_with_embeddings = await embedding_service.process_chunks_with_embeddings_async(chunks)

        # Store in vector database
        store_result = vector_store.add_documents(chunks_with_embeddings)

        # Get processing statistics
        stats = processor.get_chunk_stats(chunks)

        return {
            "status": "success",
            "document_name": document_name,
            "processing_stats": stats,
            "vector_store_result": store_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@app.post("/query-documents/")
async def query_documents(
    query: str,
    n_results: int = 5,
    embedding_service=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store)
):
    """
    Query documents using semantic search.

    Args:
        query: Search query text
        n_results: Number of results to return (default: 5)
    """
    try:
        # Generate embedding for the query
        query_embedding = embedding_service.generate_embedding(query)

        # Search for similar documents
        results = vector_store.search_similar(query_embedding, n_results)

        return {
            "query": query,
            "results": results,
            "total_results": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/search-documents/")
def search_documents(
    query: str,
    n_results: int = 5,
    vector_store=Depends(get_vector_store)
):
    """
    Search documents using text-based search (ChromaDB handles embedding internally).

    Args:
        query: Search query text
        n_results: Number of results to return (default: 5)
    """
    try:
        # Search using ChromaDB's built-in text search
        results = vector_store.search_by_text(query, n_results)

        return {
            "query": query,
            "results": results,
            "total_results": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/vector-store-info/")
def get_vector_store_info(vector_store=Depends(get_vector_store)):
    """Get information about the vector store collection."""
    return vector_store.get_collection_info()


@app.delete("/delete-document/")
def delete_document(
    document_name: str,
    vector_store=Depends(get_vector_store)
):
    """Delete all chunks for a specific document."""
    result = vector_store.delete_documents(document_name)
    return result


@app.delete("/clear-vector-store/")
def clear_vector_store(vector_store=Depends(get_vector_store)):
    """Clear all documents from the vector store."""
    result = vector_store.clear_collection()
    return result


@app.post("/recreate-collection/")
def recreate_collection(vector_store=Depends(get_vector_store)):
    """Recreate the collection to fix dimension mismatches."""
    try:
        # Delete existing collection
        vector_store.client.delete_collection(name=vector_store.collection_name)

        # Create new collection
        vector_store.collection = vector_store.client.create_collection(
            name=vector_store.collection_name,
            metadata={"description": "Document embeddings for semantic search"}
        )

        return {
            "status": "success",
            "message": "Collection recreated successfully",
            "collection_name": vector_store.collection_name
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to recreate collection: {str(e)}"
        }


@app.post("/ask-question/")
async def ask_question(
    params: AskQuestion,
    embedding_service=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store),
    answer_service=Depends(get_answer_service)
):
    """
    Ask a question and get an intelligent answer based on document content.

    Args:
        question: The question to ask about the documents
        n_results: Number of relevant chunks to retrieve (default: 5)
        include_follow_ups: Whether to include suggested follow-up questions
    """
    try:
        # Generate embedding for the question
        query_embedding = embedding_service.generate_embedding(params.question)

        # Search for relevant documents
        search_results = vector_store.search_similar(query_embedding, params.n_results)

        # Generate answer using LLM
        answer_data = answer_service.generate_answer(params.question, search_results)

        # Add follow-up questions if requested
        if answer_data.get("confidence") != "error":
            follow_ups = answer_service.generate_follow_up_questions(params.question, search_results)
            answer_data["follow_up_questions"] = follow_ups

        return {
            "question": params.question,
            "answer": answer_data["answer"],
            "confidence": answer_data["confidence"],
            "sources": answer_data["sources"],
            "search_results_count": answer_data["search_results_count"],
            "model_used": answer_data["model_used"],
            "follow_up_questions": answer_data.get("follow_up_questions", [])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")


@app.post("/ask-detailed/")
async def ask_detailed_question(
    question: str,
    n_results: int = 5,
    include_full_context: bool = False,
    embedding_service=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store),
    answer_service=Depends(get_answer_service)
):
    """
    Ask a question and get a detailed answer with full source information.

    Args:
        question: The question to ask about the documents
        n_results: Number of relevant chunks to retrieve (default: 5)
        include_full_context: Whether to include full context text in response
    """
    try:
        # Generate embedding for the question
        query_embedding = embedding_service.generate_embedding(question)

        # Search for relevant documents
        search_results = vector_store.search_similar(query_embedding, n_results)

        # Generate detailed answer
        answer_data = answer_service.generate_answer_with_sources(
            question, search_results, include_full_context
        )

        return {
            "question": question,
            "answer": answer_data["answer"],
            "confidence": answer_data["confidence"],
            "sources": answer_data["sources"],
            "detailed_sources": answer_data["detailed_sources"],
            "search_results_count": answer_data["search_results_count"],
            "model_used": answer_data["model_used"],
            "full_context": answer_data.get("full_context", [])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate detailed answer: {str(e)}")


@app.get("/chat/")
async def chat(
    message: str,
    n_results: int = 5,
    embedding_service=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store),
    answer_service=Depends(get_answer_service)
):
    """
    Simple chat interface for asking questions about documents.

    Args:
        message: The message/question to ask
        n_results: Number of relevant chunks to retrieve (default: 5)
    """
    try:
        # Generate embedding for the message
        query_embedding = embedding_service.generate_embedding(message)

        # Search for relevant documents
        search_results = vector_store.search_similar(query_embedding, n_results)

        # Generate answer
        answer_data = answer_service.generate_answer(message, search_results)

        # Generate follow-up questions
        follow_ups = answer_service.generate_follow_up_questions(message, search_results)

        return {
            "message": message,
            "response": answer_data["answer"],
            "confidence": answer_data["confidence"],
            "sources_count": len(answer_data["sources"]),
            "follow_up_suggestions": follow_ups
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/test/ws")
async def get_ws_test():
    return FileResponse("static/ws.html")


app.include_router(router)


@app.post("/timesheet-entries", status_code=201)
def create_timesheet_entry(payload: TimesheetEntryCreate):
    conn = get_db()
    try:
        cur = conn.cursor()

        # existence checks
        refs = {
            "users": payload.user_id,
            "projects": payload.project_id,
            "modules": payload.module_id,
            "tasks": payload.task_id,
            "teams": payload.team_id,
        }
        for table, value in refs.items():
            cur.execute(f"SELECT 1 FROM {table} WHERE id = ?", (value,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail=f"{table[:-1]} not found: {value}")

        # relationship checks
        cur.execute(
            "SELECT 1 FROM modules WHERE id = ? AND project_id = ?",
            (payload.module_id, payload.project_id),
        )
        if cur.fetchone() is None:
            raise HTTPException(status_code=422, detail="module_id does not belong to project_id")

        cur.execute(
            "SELECT 1 FROM tasks WHERE id = ? AND module_id = ?",
            (payload.task_id, payload.module_id),
        )
        if cur.fetchone() is None:
            raise HTTPException(status_code=422, detail="task_id does not belong to module_id")

        # allow multiple entries per user per day; daily cap enforced below

        # daily cap
        cur.execute(
            "SELECT COALESCE(SUM(working_hours), 0) FROM timesheet_entries WHERE user_id = ? AND entry_date = ?",
            (payload.user_id, payload.entry_date),
        )
        row = cur.fetchone()
        current_total = (row[0] if row and row[0] is not None else 0)
        if current_total + payload.working_hours > 24:
            raise HTTPException(status_code=422, detail="Total working_hours for the day would exceed 24 hours")

        # insert
        cur.execute(
            """
            INSERT INTO timesheet_entries
                (user_id, project_id, module_id, task_id, description, entry_date,
                 working_hours, approved_hours, authorized_hours, billed_hours, team_id, admin_comments, created_at, updated_at)
            VALUES
                (?, ?, ?, ?, ?, ?,
                 ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                payload.user_id,
                payload.project_id,
                payload.module_id,
                payload.task_id,
                payload.description,
                payload.entry_date,
                payload.working_hours,
                payload.approved_hours or 0,
                payload.authorized_hours or 0,
                payload.billed_hours or 0,
                payload.team_id,
                payload.admin_comments,
            ),
        )
        # SQLite returns the new rowid here
        rowid = cur.lastrowid
        # Try to read id; if NULL, sync id to rowid for this row
        cur.execute("SELECT id FROM timesheet_entries WHERE rowid = ?", (rowid,))
        got = cur.fetchone()
        db_id = got[0] if got else None
        if db_id is None:
            cur.execute("UPDATE timesheet_entries SET id = ? WHERE rowid = ?", (rowid, rowid))
            db_id = rowid

        conn.commit()
        return {"id": db_id}
    finally:
        conn.close()
