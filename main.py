from fastapi import FastAPI
from chroma_client import ChromaClient
from model import EmbeddingModel
from models.response_model import QueryResult, QueryResponse, ErrorResponse
from models.request_model import QueryRequest
from pydantic import BaseModel

app = FastAPI()
chroma_client = ChromaClient()
embedding_model = EmbeddingModel()


@app.get("/", response_model=QueryResponse)
def health_check():
    return {"status": "ok", "results": []}


@app.post(
    "/query", response_model=QueryResponse, responses={500: {"model": ErrorResponse}}
)
def get_similar_responses(request: QueryRequest):
    try:
        embedding = embedding_model.get_embeddings(request.input)
        results = chroma_client.query_embedding(embedding, top_k=request.top_k)

        print(results)

        response = [
            QueryResult(
                match=idx + 1,
                input=metadata["input"],
                output=metadata["output"],
                similarity_score=distance,
            )
            for idx, (metadata, distance) in enumerate(
                zip(results["metadatas"][0], results["distances"][0])
            )
        ]

        return QueryResponse(status="ok", results=response)

    except Exception as e:
        raise RuntimeError(f"Error processing request: {e}")
