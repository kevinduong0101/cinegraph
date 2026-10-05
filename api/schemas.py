from pydantic import BaseModel

class QueryRequest(BaseModel): 
   query: str

class QueryResponse(BaseModel): 
   answer: str
   recommended_movies: list[str]
   graph_context: list[dict]
   execution_time_seconds: float 