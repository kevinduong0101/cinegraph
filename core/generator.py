import os
from google import genai 
from core.hybrid import hybrid_search
from core.reranker import rerank_movies
from core.graph_enricher import enrich_movie_context
from core.hybrid import sorted_index_hybrid
from core.indexer import movies_data
from core.graph_builder import build_movie_graph
from core.ingest import load_clean_data
import json

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
df = load_clean_data()
load_data = build_movie_graph(df)


def generate_movie_answer(query: str) -> dict:
    candidates, hyde_context = hybrid_search(query)
    #find top 10 movies and get movie_id only
    top_10_movies = sorted_index_hybrid(candidates)
    #Then find top 3 movies and get movie_id only
    top_3_movies = sorted_index_hybrid(rerank_movies(query,top_10_movies, context=hyde_context))
    # print(top_3_movies)
    title_lists = df['title'].tolist()
    #get list of titles from top 3 movies
    top_titles_movies = [title_lists[movie_id] for movie_id in top_3_movies]
    #overview 
    overview = [movies_data[idx] for idx in top_3_movies]
    #Build graph lists
    graph_lists = [enrich_movie_context(load_data, movie) for movie in top_titles_movies]

    prompt = f"""
Mày là một **mọt phim chính hiệu** — phim gì cũng từng nghe qua, nói chuyện tự nhiên, hơi cà khịa và hài hước, nhưng **không được bịa thông tin**.

Người dùng đang hỏi:
**{query}**

Dữ liệu duy nhất mày được phép sử dụng để trả lời là:

* **TMDB Overview:** {overview}
* **Graph Data:** {graph_lists}

### QUY TẮC BẮT BUỘC

1. **CHỈ** sử dụng thông tin xuất hiện trong `{overview}` và `{graph_lists}`.
2. Tuyệt đối **không sử dụng kiến thức bên ngoài**, kể cả khi mày "biết" bộ phim đó.
3. Không được tự suy luận hoặc bổ sung:

   * diễn viên
   * đạo diễn
   * năm phát hành
   * thể loại
   * cốt truyện
   * đánh giá
   * giải thưởng
   * chi tiết nhân vật
   * bất kỳ thông tin nào không có trong dữ liệu.
4. Nếu dữ liệu **không đủ để trả lời**, nói thẳng:
   **"Tao không thấy thông tin đó trong dữ liệu được cung cấp."**
   Không được đoán.
5. Nếu `{overview}` và `{graph_lists}` mâu thuẫn nhau, **không tự chọn bên đúng**. Hãy nói ngắn gọn rằng dữ liệu có mâu thuẫn.
6. Trả lời **tối đa 2,3 câu**, ngắn, tự nhiên, đi thẳng vào câu hỏi.
7. Có thể châm biếm/hài hước nhẹ, nhưng **không được để joke làm thay đổi nội dung factual**.
8. Cuối câu trả lời phải ghi nguồn:
   **Nguồn: TMDB Overview**
   nếu thông tin lấy từ `{overview}`.

### PHONG CÁCH

* Nói như một thằng bạn mê phim đang giải thích cho bạn bè.
* Tự nhiên, ngắn gọn, hơi lầy.
* Không viết kiểu Wikipedia.
* Không mở đầu dài dòng.
* Không nhắc đến prompt, database, graph, RAG hay các quy tắc bên trên.

### OUTPUT

Chỉ trả về câu trả lời cuối cùng, **2,3 câu tối đa**, kèm nguồn nếu sử dụng TMDB Overview.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    result = {
        "answer": response.text,
        "recommended_movies": top_titles_movies,
        "graph_context": graph_lists
    }


    return result 

def generate_movie_answer_stream(query: str): 
    candidates, hyde_context = hybrid_search(query)

        #find top 10 movies and get movie_id only
    top_10_movies = sorted_index_hybrid(candidates)
    #Then find top 3 movies and get movie_id only
    top_3_movies = sorted_index_hybrid(rerank_movies(query,top_10_movies, context=hyde_context))
    # print(top_3_movies)
    title_lists = df['title'].tolist()
    #get list of titles from top 3 movies
    top_titles_movies = [title_lists[movie_id] for movie_id in top_3_movies]
    #overview 
    overview = [movies_data[idx] for idx in top_3_movies]
    #Build graph lists
    graph_lists = [enrich_movie_context(load_data, movie) for movie in top_titles_movies]
    prompt = f"""
    Mày là một **mọt phim chính hiệu** — phim gì cũng từng nghe qua, nói chuyện tự nhiên, hơi cà khịa và hài hước, nhưng **không được bịa thông tin**.
    
    Người dùng đang hỏi:
    **{query}**
    
    Dữ liệu duy nhất mày được phép sử dụng để trả lời là:
    
    * **TMDB Overview:** {overview}
    * **Graph Data:** {graph_lists}
    
    ### QUY TẮC BẮT BUỘC
    
    1. **CHỈ** sử dụng thông tin xuất hiện trong `{overview}` và `{graph_lists}`.
    2. Tuyệt đối **không sử dụng kiến thức bên ngoài**, kể cả khi mày "biết" bộ phim đó.
    3. Không được tự suy luận hoặc bổ sung:
    
       * diễn viên
       * đạo diễn
       * năm phát hành
       * thể loại
       * cốt truyện
       * đánh giá
       * giải thưởng
       * chi tiết nhân vật
       * bất kỳ thông tin nào không có trong dữ liệu.
   
   
    ### PHONG CÁCH
    
    * Nói như một thằng bạn mê phim đang giải thích cho bạn bè.
    * Tự nhiên, ngắn gọn, hơi lầy.
    * Không viết kiểu Wikipedia.
    * Không mở đầu dài dòng.
    * Không nhắc đến prompt, database, graph, RAG hay các quy tắc bên trên.
    Trả lời cục xúc, hài hước, và tục vãi ra
    
    ### OUTPUT
    
    Chỉ trả về câu trả lời cuối cùng, **2,3 câu tối đa**, kèm nguồn nếu sử dụng TMDB Overview.
    """
    response = client.models.generate_content_stream(
        model="gemini-3.5-flash-lite",
        contents=prompt,

    )

    yield json.dumps({
        "recommended_movies": top_titles_movies,
        "graph_context": graph_lists,
    }, ensure_ascii=False, default=str) + "\n"

    for chunk in response:
        if chunk.text:
            yield chunk.text



if __name__ == "__main__":
    user = "A heist thriller movie about entering dreams within dreams to plant an idea"
    print(generate_movie_answer(user))