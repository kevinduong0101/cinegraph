from api.schemas import QueryRequest, QueryResponse 
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from core.generator import generate_movie_answer, generate_movie_answer_stream
import time 


app = FastAPI()

@app.get("/health") 
def health():
    return {
        "status": "healthy",
        "service": "CineGraph Intel"
    }

@app.post("/api/v1/search", response_model=QueryResponse)

async def search(req: QueryRequest):
    start = time.perf_counter()
    try: 
        response = generate_movie_answer(req.query)
    except Exception as e: 
        duration_ms = time.perf_counter() - start
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    
    duration_ms = time.perf_counter() - start

    return {
        "answer":response['answer'],
        "recommended_movies":response['recommended_movies'],
        "graph_context":response['graph_context'],
        "execution_time_seconds":duration_ms
    }

@app.post("/api/v1/search/stream", )
async def search_stream(req: QueryRequest):
    generation = generate_movie_answer_stream(req.query)
    return StreamingResponse(generation)