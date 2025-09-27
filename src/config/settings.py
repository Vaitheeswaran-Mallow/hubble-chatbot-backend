"""
Configuration settings for the Endorsement Process RAG System
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # OpenAI Configuration
    open_api_key: str
    
    # Database Configuration
    database_url: str = "sqlite:///./hubble.db"
    
    # File Paths
    excel_file_path: str = "data/Endorsement Process Excel.xlsx"
    chroma_db_path: str = "./chroma_db"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "Hubble Chatbot Backend - Endorsement Process RAG"
    api_description: str = "RAG system for querying endorsement processes using OpenAI"
    api_version: str = "1.0.0"
    
    # RAG Configuration
    collection_name: str = "endorsement_processes"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    generation_model: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.3
    
    # Logging Configuration
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
config = Settings()
