

def enrich_movie_context(G, movie_title: str) -> str: 
    directors_list = []
    actors = []
    genres = []
    movies_list = []
    # Find directors, actors from movie 
    for item in G.predecessors(movie_title):
        relation = G[item][movie_title]['relation']
        if relation == "DIRECTED": 
            directors_list.append(item)
        elif relation == "ACTED_IN": 
            actors.append(item)

    # find genres from movie 
    for genre in G.successors(movie_title): 
        genres.append(genre)

    # find another movies from director 
    for director in directors_list: 
        for movies in G.successors(director): 
            if movies != movie_title:
                movies_list.append(movies)
                
    # show_movie = (f"các tác phẩm khác: {', '.join(movies_list)}" if movies_list else "")

    return {
        "Movie": movie_title,
        "Director": directors_list,
        "Actors": actors,
        "Genres": genres
    }
