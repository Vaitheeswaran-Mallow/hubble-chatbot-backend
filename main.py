#!/usr/bin/env python3
"""
Main entry point for the Endorsement Process RAG System
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.api.main import app

if __name__ == "__main__":
    import uvicorn
    from src.config import config
    
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        reload=True
    )
