from google import genai
from functools import lru_cache
import os 

@lru_cache
def get_client():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    if not client:
        raise ValueError("Need Gemini Key to run")
    return client

def generate_hypothetical_overview(query: str) -> str: 
    client = get_client()
    prompt = f"""You are a film expert.
When the user describes a movie they remember in just a few sentences, your task is to:
Guess the movie title that best matches that description.
Summarize the plot of the movie in 2 to 3 sentences in English.
Here is the movie description: {query} """
    response = client.models.generate_content( 
        model="gemini-3.5-flash-lite",
        contents=prompt
    )
    return response.text 

def generate(prompt: str): 
    client = get_client()
    response = client.models.generate_content( 
        model="gemini-3.5-flash-lite",
        contents=prompt
    )
    return response.text 


def generate_stream(prompt: str): 
    client = get_client()
    response = client.models.generate_content_stream(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )
    for chunk in response: 
        if chunk.text:
            yield chunk.text

