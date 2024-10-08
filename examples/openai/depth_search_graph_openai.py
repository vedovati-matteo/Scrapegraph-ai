"""
depth_search_graph_opeani example
"""
import os
from dotenv import load_dotenv
from scrapegraphai.graphs import DepthSearchGraph

load_dotenv()

openai_key = os.getenv("OPENAI_APIKEY")

graph_config = {
    "llm": {
        "api_key": openai_key,
        "model": "openai/gpt-4o-mini",
    },
    "embedder_model": {
        "source": "openai",
        "model": "text-embedding-ada-002",
    },
    "verbose": True,
    "headless": False,
    "depth": 2,
    "only_inside_links": False,
}

search_graph = DepthSearchGraph(
    prompt="List me all the projects with their description",
    source="https://perinim.github.io",
    config=graph_config
)

result = search_graph.run()
print(result)
