from sentence_transformers import CrossEncoder 
from core.indexer import movies_data

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')

# Use CrossEncoder Rerank top 3 movies again after hybrid 
def rerank_movies(query: str, candidate_indices: list[int], context: str = "", top_k=3) -> list[tuple[int, float]]:
    scoring_query = f"{query} | Context: {context}" if context else query
    score = model.predict([(scoring_query, movies_data[idx]) for idx in candidate_indices])
    list_score = list(zip(candidate_indices, score))
    sorted_score = sorted(list_score, key=lambda x: x[1], reverse=True)[:top_k]
    return sorted_score



