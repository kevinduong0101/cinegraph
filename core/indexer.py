from rank_bm25 import BM25Okapi
from core.ingest import load_clean_data
import numpy as np
import os 
import re
import voyageai
from pathlib import Path
import time
from typing import Sequence
from functools import lru_cache

path = Path(__file__).parent.parent
EMBEDDING_PATH = path/"data"/"movie_embeddings.npy"

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def chunk(data: Sequence, batch_size: int = 50): 
    for i in range(0,len(data), batch_size):
        yield data[i:i+batch_size] 

class MovieIndexer:
    def __init__(self):
        self.bm25 = None
        self.embeddings = None
        self.movies_data = None
        self.client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    def build_bm25(self, corpus): 
        self.bm25 = BM25Okapi(corpus)


    def bm25_search(self, query, top_k=20): 
        # tokenized query
        tokenized_query = tokenize(query)

        #get scores for all documents
        if self.bm25 is None:
            raise ValueError("Bm25 is empty")
        
        doc_scores = self.bm25.get_scores(tokenized_query)

        #sorted and extract top_k 
        top_k_indices = np.argsort(doc_scores)[::-1][:top_k]

        result = [(idx, doc_scores[idx]) for idx in top_k_indices]

        return result

    def load_embeddings(self, filepath): 
        #load data 
        try: 
            self.embeddings = np.load(filepath, allow_pickle=False)
        except FileNotFoundError as e: 
            raise FileNotFoundError(
                f"Can't fine embedding at: {Path(filepath).resolve()}"
                ) from e
        except ValueError as e: 
            raise ValueError(
                f"File embeddings incorrect format or can't load"
            ) from e

    def _embed(self, texts: str | list[str]): 
        is_query = isinstance(texts, str)
        texts_list = [texts] if is_query else texts

        response = self.client.embed(
            texts=texts_list,
            model="voyage-3",
            input_type="query" if is_query else "document"
        )

        return response.embeddings


    def vector_search(self, query, top_k=20): 
        if self.embeddings is None:
            raise ValueError("Embedding is empty")
        
        query_embedding = self._embed(query)[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (idx, scores[idx])
            for idx in top_indices
        ] 

    def embedding_documents(self, data, batch_size:int = 50): 
        embeddings = []
        max_retries = 3 

        for idx, batch in enumerate(chunk(data, batch_size), start=1):
            retry_count = 0
            while retry_count < max_retries: 
                try:
                    embeddings.extend(self._embed(batch))
                    print(f"Batch {idx} completed")
                    break
                except Exception as e: 
                    retry_count += 1
                    if retry_count == max_retries: 
                        raise RuntimeError(
                            f"Failed to run batch {idx}"
                        ) from e
                    else: 
                        #delay 20s
                        time.sleep(20)

        self.embeddings = np.array(embeddings)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / norms
        np.save(EMBEDDING_PATH, self.embeddings)
        return self.embeddings


def prepare_documents(df) -> list[str]: 
    movies_data = []
    for row in df.itertuples(index=False):
        genres = ", ".join(row.genres) if row.genres else ""
        director = ", ".join(row.director) if row.director else ""
        cast = ", ".join(row.cast) if row.cast else ""
        movies_data.append(f"Title: {row.title} | Genres: {genres} | Director: {director} | Cast: {cast} | Overview: {row.overview}")
    return movies_data


@lru_cache(maxsize=1)
def get_indexer():
    indexer = MovieIndexer()
    df = load_clean_data()
    docs = prepare_documents(df)
    indexer.movies_data = docs
    indexer.build_bm25([tokenize(d) for d in docs])
    indexer.load_embeddings(EMBEDDING_PATH)
    return indexer

if __name__ == "__main__":
    indexer = get_indexer()

    # Read datas and build BN25

    query = "A dream within a dream heist thriller"
    print("Result: bm25")
    print(indexer.bm25_search(query,top_k=3))
    # Result: bm25
    # [(np.int64(65), np.float64(11.317545678008184)), (np.int64(82), np.float64(8.454782672342002)), (np.int64(68), np.float64(8.356196327371032))]
    print("------------")
    print("Result: vector_search")
    print(indexer.vector_search(query,top_k=3))
    # Result: vector_search
    # [(np.int64(35), np.float64(0.4958261434363369)), (np.int64(55), np.float64(0.43274956137771936)), (np.int64(79), np.float64(0.4188014845726953))]