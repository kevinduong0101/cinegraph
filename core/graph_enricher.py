

def enrich_movie_context(G, movie_title: str) -> dict: 
    directors_list = []
    actors = []
    genres = []

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


    return {
        "Movie": movie_title,
        "Director": directors_list,
        "Actors": actors,
        "Genres": genres
    }
