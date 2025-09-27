from fastapi import FastAPI, Depends
from settings import config
from langchain_openai import OpenAI
from db import get_chroma_client
from markitdown import MarkItDown

import sys
import os

def get_markdown_converter():
    """Initialize and return a Markitdown instance."""
    return MarkItDown(enable_plugins=False)

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
    result = md.convert("./docs/New Leave policy.pdf")
    return {"converted_text": result}