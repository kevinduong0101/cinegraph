from rank_bm25 import BM25Okapi
from core.ingest import load_clean_data
import numpy as np
import os 
# from google import genai
# from google.genai import types
import voyageai
from pathlib import Path
import time

vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
path = Path(__file__).parent.parent
BATCH_SIZE = 50
EMBEDDING_PATH = path/"data"/"movie_embeddings.npy"

df = load_clean_data()
movies_data = []
for row in df.itertuples(index=False):
    genres = ", ".join(row.genres) if row.genres else ""
    director = ", ".join(row.director) if row.director else ""
    cast = ", ".join(row.cast) if row.cast else ""
    movies_data.append(f"Title: {row.title} | Genres: {genres} | Director: {director} | Cast: {cast} | Overview: {row.overview}")

tokenized_corpus = [data.lower().split(" ") for data in movies_data] 

bm25 = BM25Okapi(tokenized_corpus)

#load data 
try: 
    movie_embeddings = np.load(EMBEDDING_PATH, allow_pickle=False)
except FileNotFoundError as e: 
    raise FileNotFoundError(
        f"Can't fine embedding at: {EMBEDDING_PATH.resolve()}"
        ) from e
except ValueError as e: 
    raise ValueError(
        f"File embeddings incorrect format or can't load"
    ) from e

def bm25_search(query, top_k=20): 
    results = []
    # tokenized query
    tokenized_query = query.lower().split(" ")

    #get scores for all documents
    doc_scores = bm25.get_scores(tokenized_query)

    #sorted and extract top_k 
    top_k_indices = np.argsort(doc_scores)[::-1][:top_k]

    for idx in top_k_indices: 
        top = (idx, doc_scores[idx])
        results.append(top)

    return results 

def chunk(data:list, batch_size: int = BATCH_SIZE): 
    for i in range(0,len(data), batch_size):
        yield data[i:i+batch_size] 

# def gemini_embedding(data): 
#     if type(data) == list: 
#         content_batch = [types.Content(parts=[
#             types.Part.from_text(text=t) 
#         ]) for t in data]
#     else: 
#         content_batch = data

#     response = client.models.embed_content(
#             model="gemini-embedding-001",
#             contents=content_batch
#     )
#     return [r.values for r in response.embeddings]

def voyageai_embedding(datas: str | list[str]): 
    is_query = isinstance(datas, str)
    if is_query: 
        datas = [datas]

    response = vo.embed(
        texts=datas,
        model="voyage-3",
        input_type="query" if is_query else "document"
    )

    return response.embeddings


def embedding_documents(data, batch_size:int = BATCH_SIZE): 
    embeddings = []
    max_retries = 3 

    for idx, batch in enumerate(chunk(data, batch_size), start=1):
        retry_count = 0
        while retry_count < max_retries: 
            try:
                embeddings.extend(voyageai_embedding(batch))
                print(f"Batch {idx} completed")
                break
            except Exception as e: 
                retry_count += 1
                if retry_count == max_retries: 
                    raise RuntimeError(
                        f"Failed to run batch {idx}"
                    ) from e
                else: 
                    time.sleep(21)


    np.save(EMBEDDING_PATH, embeddings)
    return embeddings

def cosine(a,b): 
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0 
    return np.dot(a,b) / (norm_a * norm_b)

def vector_search(query, top_k=20): 
    

    e_query = voyageai_embedding(query)[0]

    scores = []

    for data in movie_embeddings: 
        scores.append(cosine(e_query, data))

    top_vector_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_vector_idx:
        results.append((idx,scores[idx]))


    return results 


if __name__ == "__main__":
    # embedding_documents(movies_data)
    query = "A dream within a dream heist thriller" 
    print("Result: bm25")
    print(bm25_search(query,top_k=3))
    print("------------")
    print("Result: vector_search")
    print(vector_search(query,top_k=3))
    print(np.load(EMBEDDING_PATH).shape)