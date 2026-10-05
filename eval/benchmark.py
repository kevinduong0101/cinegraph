from core.indexer import bm25_search, vector_search
from core.hybrid import hybrid_search, sorted_index_hybrid
from core.ingest import load_clean_data
from core.reranker import rerank_movies
import time

df = load_clean_data()
title_lists = df['title'].tolist()


questions_list = [
    {
        "query": "The movie about batman, direct by Christopher Nolan something dark knight",
        "expected": "The Dark Knight"
    },
    {
        "query": "A sci-fi movie about dreams within dreams",
        "expected": "Inception"
    },
    {
        "query": "A young boy accidentally travels to the Land of the Dead",
        "expected": "Coco"
    }]

expected_movies = [movie['expected'] for movie in questions_list]



def evaluate_metrics(predictions: list[list[str]], expected: list[str]):
    total = len(expected)
    hits = 0
    rr_scores = 0.0

    for i, target in enumerate(expected):
        top_3 = predictions[i]
        if target in top_3:
            hits += 1
            rank = top_3.index(target) + 1
            rr_scores += 1.0 / rank

    hit_rate = (hits / total) * 100
    mrr = rr_scores / total
    return hit_rate, mrr


def check_top_movies():

    bm_25_movies = []
    vec_movies = []
    hybrid_movies = []

    for q in questions_list: 

        question = q['query']

        bm25 = sorted_index_hybrid(bm25_search(question, top_k=3))
        try:
            vector = sorted_index_hybrid(vector_search(question,top_k=3))
        except Exception as e:
            vector = []
        time.sleep(20)
        hybrid_candidates, hyde_ctx = hybrid_search(question, top_k=10)
        reranked = rerank_movies(question, sorted_index_hybrid(hybrid_candidates), context=hyde_ctx, top_k=3)
        sorted_hybrid = sorted_index_hybrid(reranked)


        bm_25_movies.append([title_lists[i] for i in bm25])
        vec_movies.append([title_lists[i] for i in vector])
        hybrid_movies.append([title_lists[i] for i in sorted_hybrid])
        time.sleep(20)

    hit_rate_bm25, mrr_bm25 = evaluate_metrics(bm_25_movies, expected_movies)
    hit_rate_vec, mrr_vec = evaluate_metrics(vec_movies, expected_movies)
    hit_rate_hybrid, mrr_hybrid = evaluate_metrics(hybrid_movies, expected_movies)

    return f""" Hit rate top 3:
    BM-25: {hit_rate_bm25:.1f}%
    Vector: {hit_rate_vec:.1f}%
    Hybrid: {hit_rate_hybrid:.1f}%
----------------------------------
    Mean Reciprocal Rank:
    BM-25: {mrr_bm25:.3f}
    Vector: {mrr_vec:.3f}
    Hybrid: {mrr_hybrid:.3f}

"""

print(check_top_movies())

