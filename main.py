from fastapi import FastAPI, Depends
from settings import config
# from langchain_openai import OpenAI
from openai import OpenAI
from db import get_chroma_client, get_collection
from markitdown import MarkItDown
from fastapi import Query

app = FastAPI(title="Hubble Chatbot Backend")


def get_markdown_converter():
    """Initialize and return a Markitdown instance."""
    return MarkItDown(enable_plugins=False)


@app.get("/")
def hello_world():
    return {"message": "Hello, World!"}


@app.get("/config")
def get_config():
    return {"open_api_key": config.open_api_key}


# @app.get("/llm-test")
# def llm_test():
#     llm = OpenAI(api_key=config.open_api_key, temperature=0)
#     response = llm.invoke("What is the capital of India?")
#     return {"response": response}


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


@app.get("/list-collections/")
def list_collections(chroma_db=Depends(get_chroma_client)):
    collections = chroma_db.list_collections()
    return {"collections": collections}


@app.get("/query-leave-policy/")
def query_leave_policy(query: str = Query(...), chroma_db=Depends(get_collection)):
    results = chroma_db.query(query_texts=[query], n_results=5)
    print(results)
    client = OpenAI(api_key=config.open_api_key)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that helps employees with their HR related queries based on the company policies.",
        },
        {
            "role": "user",
            "content": f"Answer the question based on the context below. If you don't know the answer, just say that you don't know, don't try to make up an answer. Context: {results['documents']}",
        },
        {"role": "user", "content": query},
    ]
    response = client.responses.create(
        model="gpt-4o-mini", input=messages
    )
    return {"response": response.output_text}
