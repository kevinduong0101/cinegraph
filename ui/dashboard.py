import streamlit as st 
from streamlit_agraph import agraph, Node, Edge, Config
import requests
import json

FASTAPI_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Movies search",
    layout="wide"
)

with st.sidebar:
    st.text("""⚡ Dual Indexing: BM25Okapi + Voyage-3 Embeddings
🎯 Re-ranking: Cross-Encoder (ms-marco-MiniLM-L-6-v2)
🕸️ Knowledge Graph: NetworkX 2-hop Traversal
🤖 Generator: Gemini 2.5 Flash Lite (NDJSON Stream)"""
    )
    st.divider()
    if st.button("🗑️ Xóa lịch sử hội thoại"):
        st.session_state.messages = []
        st.session_state.moviedatas = []
        st.rerun()



st.title("🎬 CineGraph Intel — Multi-Stage Hybrid & Graph RAG Engine.")

if "messages" not in st.session_state:
    st.session_state.messages = []

response = None

if "moviedatas" not in st.session_state: 
    st.session_state.moviedatas = []

def text_generator(stream_lines):
    for line in stream_lines:
        if line:
            yield line.decode("utf-8") + "\n"

#create column, first column 60%, second 40%
col1, col2 = st.columns(spec=[0.55, 0.45])

with col1: 
    st.header("Find out your movie 🎥")
    st.caption("🟡 Phim đề xuất | 🟨 Đạo diễn | 🔷 Diễn viên | 🟢 Thể loại")
    # Form input 
    chat_box = st.container(height=500)

    #show previous chat 
    with chat_box: 
        for message in st.session_state.messages: 
            with st.chat_message(message['role']):
                st.write(message['content'])

    #input chat here
    prompt = st.chat_input("Describe your damn movie...")

    #check if user just send message 
    if prompt: 
        #show message in chatbox 
        with chat_box: 
            with st.chat_message("user"): 
                st.write(prompt)
        # save message to history 
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        #Send message to fastAPI by POST 
        try: 
            response = requests.post(
                f"{FASTAPI_URL}/api/v1/search/stream",
                json={
                    "query": prompt,
                },
                stream=True
            )
            # check if response work 
            if response.ok: 
                lines = response.iter_lines()
                first_line = next(lines)
                metadata = json.loads(first_line.decode("utf-8"))
                st.session_state.moviedatas.append(metadata)
                with chat_box:
                    with st.chat_message("assistant"):
                        reply = st.write_stream(text_generator(lines))

                #Save responses to history chat 
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply
                })

            else: 
                st.error("The message processing API has error")

        except requests.exceptions.ConnectionError:
            st.error("Can't connect to fastAPI backend")


def create_graph(movie):
    nodes = []
    edges = []

    node_style = {
        "movie": {"size": 30, "shape": "box", "color": "yellow"},
        "director": {"size": 20, "shape": "box", "color": "yellow"},
        "actor": {"size": 15, "shape": "box", "color": "blue"},
        "genre": {"size": 12, "shape": "ellipse", "color": "green"}
    }
    movie_style = node_style['movie']

    for idx, film in enumerate(movie, start=1): 
        relationships = {
            "director": film["Director"],
            "actor": film["Actors"],
            "genre": film["Genres"]
        }
        movie_id = f"movie_{idx}"
        nodes.append(Node(
            id=movie_id,
            label=film['Movie'],
            size= movie_style['size'],
            shape=movie_style['shape'],
            color= movie_style['color'],
        ))
        for node_type, values in relationships.items():
            style = node_style[node_type]
            for node_idx, value in enumerate(values):
                node_id = f"{idx}_{node_type}_{node_idx}"
                nodes.append(Node(
                    id=node_id,
                    label=value,
                    size=style['size'],
                    shape=style['shape'],
                    color=style['color'],
                ))

                edges.append(Edge(
                    source=movie_id,
                    target=node_id,
                    color=style['color']
                ))


    return nodes, edges

with col2: 
    st.header("Graph")
    if st.session_state.moviedatas: 
        movies = st.session_state.moviedatas[-1]['graph_context']
        nodes, edges = create_graph(movies)
        config = Config(width=700,
                height=900,
                directed=True, 
                physics=True, 
                hierarchical=False,
                )
        return_value = agraph(
            nodes=nodes,
            edges=edges,
            config=config
        )