from core.indexer import bm25_search, vector_search 
from collections import defaultdict
from core.hyde import generate_hypothetical_overview
from core.indexer import movies_data
from core.reranker import rerank_movies

#Calculate RRF function - between bm25 vs vector 
def reciprocal_rank_fusion(bm25_results, vector_results, k=60) -> list[tuple[int, float]]:
    scores = defaultdict(float)

    for rank, (doc_id, _) in enumerate(bm25_results, start=1): 
        scores[doc_id] += 1 / (k + rank)

    for rank, (doc_id, _) in enumerate(vector_results, start=1): 
        scores[doc_id] += 1 / (k + rank) 

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

#find top-10 movies 
def hybrid_search(query: str, top_k=10):
    #check if Gemini work well 
    try: 
        vector_query = generate_hypothetical_overview(query)
    except Exception as e:
        print("HyDE failed; fallback to original query: %s", e)
        vector_query = query
    #Check Vector search
    try: 
        vector_result = vector_search(vector_query, top_k=20)
        result = reciprocal_rank_fusion(bm25_search(query, top_k=20), vector_result)[:top_k]
        return result, vector_query
    except Exception as e: 
        print("Vector search or RRF failed; fallback to BM25 only: %s", e)
        return bm25_search(query, top_k=top_k), vector_query

#this function helps cover list tuble to list [int]
def sorted_index_hybrid(items): 
    return [int(rank) for (rank,_) in items]


if __name__ == "__main__": 
    user = "A sci-fi movie about traveling through a wormhole near Saturn to save humanity"
    search, vector_query = hybrid_search(user)
    sorted_hybrid = sorted_index_hybrid(search)
    rerank_top_3 = rerank_movies(user, sorted_hybrid, vector_query)
    sorted_rerank = sorted_index_hybrid(rerank_top_3)
    print(f"The movie base on describe: {user}")
    for rank,idx in enumerate(sorted_rerank, start=1):
        print(f"Top {rank}: {movies_data[idx]}")
        print("-----------------------------")

