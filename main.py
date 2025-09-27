#!/usr/bin/env python3
"""
Main entry point for the Endorsement Process RAG System
"""

import uvicorn
import os
import sys

# Add src to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)