import csv
import pandas as pd 
from pathlib import Path   
import ast

path = Path(__file__).parent.parent
LINK_DATA = path/'data'/'movie_data.csv'

def load_clean_data(link=LINK_DATA):

    csv.field_size_limit(10_000_000)
    df = pd.read_csv(LINK_DATA, engine="python")

    #Filter to find the real good movies 
    best_movies = df[(df["popularity"] > 30) & (df["vote_count"] > 300)]

    # Top 300 movies
    df = best_movies.nlargest(100, "vote_average")

    #cover str to list 
    list_cols = ["genres","cast", "crew"] 
    df[list_cols] = df[list_cols].map(ast.literal_eval)

    df = df[["movie_id", "title", "overview", "genres", "cast", "crew"]]
    df = df.dropna(subset=["overview", "title"])

    df = df.reset_index(drop=True)

    #cover genres to list of kind of movies 
    df['genres'] = df['genres'].apply(
        lambda genres: [item['name'] for item in genres]
    )
    #cover cast to list of actor 
    df['cast'] = df['cast'].apply(
        lambda actor: [x['name'] for x in actor][:5]
    )

    #Column Director from crew 
    df['director'] = df['crew'].apply(
        lambda crew: [x['name'] for x in crew if x['job'] == 'Director']
    )


    #Delete crew 
    df = df.drop("crew", axis=1)

    return df

if __name__ == "__main__":
    print(load_clean_data(LINK_DATA).director)