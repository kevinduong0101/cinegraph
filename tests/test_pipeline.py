from core.ingest import load_clean_data
from core.graph_builder import build_movie_graph
from core.graph_enricher import enrich_movie_context
from core.hybrid import reciprocal_rank_fusion 

def test_ingest_data():
    df = load_clean_data()
    # Kiểm tra có đủ các cột cốt lõi 
    assert "title" in df.columns
    assert "director" in df.columns
    # Kiểm tra index start from 0 
    assert df.index[0] == 0

def test_graph_structure():
    df = load_clean_data()
    G = build_movie_graph(df)
    info = enrich_movie_context(G, "The Dark Knight")
    assert isinstance(info, dict)
    assert "Christopher Nolan" in info["Director"]

def test_rrf_math():
    bm25_mock = [(1,10.0), (2, 5.0)]
    vector_mock = [(1, 0.9), (3, 0.8)]
    fused = reciprocal_rank_fusion(bm25_mock, vector_mock, k=60)

    assert fused[0][0] == 1

test_ingest_data()