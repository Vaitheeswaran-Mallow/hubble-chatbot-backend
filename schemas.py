from pydantic import BaseModel


class AskQuestion(BaseModel):
    question: str
    n_results: int = 3
