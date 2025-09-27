import os
from pathlib import Path

import chromadb
from markitdown import MarkItDown
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)


def get_chroma_client():
    """Initialize and return a ChromaDB client."""
    return chromadb.PersistentClient(path="./chroma_db")


def get_collection():
    """Get or create a collection in ChromaDB."""
    client = get_chroma_client()
    return client.get_or_create_collection(name="documents")


def convert_to_markdown(file_path: str) -> str:
    """Convert a given file to markdown format using MarkItDown."""
    md = MarkItDown(enable_plugins=False)
    result = md.convert(file_path)
    return result.text_content


def character_splitter(text: str):
    """Split text into smaller chunks using RecursiveCharacterTextSplitter."""
    text_splitter = CharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=100,
        # length_function=len,
        separator="\n\n",
    )
    return text_splitter.split_text(text)


def add_document_to_db():
    collection = get_chroma_client().get_or_create_collection(name="documents")
    docs = os.listdir("./docs")
    for doc in docs:
        text = convert_to_markdown(f"./docs/{doc}")
        chunks = character_splitter(text)
        for chunk in chunks:
            collection.add(
                documents=[chunk], metadatas={"source": doc}, ids=[f"{doc}_{chunk}"]
            )


if __name__ == "__main__":
    add_document_to_db()
