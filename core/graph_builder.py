# from core.ingest import load_clean_data 
from core.ingest import load_clean_data
from core.graph_enricher import enrich_movie_context
import networkx as nx


def build_movie_graph(df): 
    G = nx.DiGraph()
    #Use itertuples for database pandas 
    for data in df.itertuples(index=False): 
        G.add_node(data.title, type="movie", movie_id=data.movie_id)
        # relationship between movie and the genre 
        for genre in data.genres:
            G.add_edge(data.title, genre, relation="BELONGS_TO")
        # relationship between directors and their movies
        for d in data.director:
            G.add_edge(d, data.title, relation="DIRECTED")
        # relationship between actors and their movies
        for actor in data.cast:
            G.add_edge(actor, data.title, relation="ACTED_IN") 
    return G

def find_actors_by_director(G, director_name: str) -> list: 
    results = []
    # find movies are directed by directors
    for movie_name in G.successors(director_name): 
        relation = G[director_name][movie_name]['relation']
        # find directors and actors that made movies 
        for cast in G.predecessors(movie_name): 
            # Just find movies
            if G[cast][movie_name]['relation'] == 'ACTED_IN': 
                relation_2 = G[cast][movie_name]['relation'] 
                results.append(f"{director_name} ---[{relation}] --> {movie_name} <-- [{relation_2}] -- {cast}")
    return results


if __name__ == "__main__":
    load_data = build_movie_graph(load_clean_data())
    christopher_nolan_list = find_actors_by_director(load_data, "Christopher Nolan")

    movie_title = "Interstellar"
    print(enrich_movie_context(load_data,movie_title))