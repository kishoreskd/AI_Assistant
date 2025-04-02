from fastapi import FastAPI
from chroma_client import ChromaClient
from model import EmbeddingModel
from models import QueryRequest, QueryResponse, ErrorResponse, QueryRequest
from pydantic import BaseModel

app = FastAPI()
chroma_client = ChromaClient()
embedding_model = EmbeddingModel()


@app.get("/", response_model=QueryResponse)
def health_check():
    return {"status": "ok", "response": "Service is running"}


def get_similar_responses(request: QueryRequest):
    try:
        embedding = embedding_model.get_embeddings(request.input)
        results = chroma_client.query_embedding(embedding, top_k=request.top_k)

    except Exception as e:
        raise RuntimeError(f"Error processing request: {e}")
