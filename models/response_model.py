from pydantic import BaseModel
from typing import Optional as optional


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: optional[str] = None


class QueryResult(BaseModel):
    match: int
    input: str
    output: str
    similarity_score: float


class QueryResponse(BaseModel):
    status: str
    results: list[QueryResult]
