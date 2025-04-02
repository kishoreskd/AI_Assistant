from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    input: str
    top_k: int = 5
