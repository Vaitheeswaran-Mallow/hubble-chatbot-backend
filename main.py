from fastapi import FastAPI
from settings import config
from langchain_openai import OpenAI

app = FastAPI(title="Hubble Chatbot Backend")


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