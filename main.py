from fastapi import FastAPI
from settings import config

app = FastAPI(title="Hubble Chatbot Backend")


@app.get("/")
def hello_world():
    return {"message": "Hello, World!"}


@app.get("/config")
def get_config():
    return {"open_api_key": config.open_api_key}
