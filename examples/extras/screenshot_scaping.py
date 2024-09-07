""" 
Basic example of scraping pipeline using SmartScraper
"""

import json
from scrapegraphai.graphs import ScreenshotScraperGraph
from scrapegraphai.utils import prettify_exec_info

# ************************************************
# Create the ScreenshotScraperGraph instance and run it
# ************************************************

smart_scraper_graph = ScreenshotScraperGraph(
    prompt="",
    source="https://perinim.github.io/projects/",
    config={"llm": {
        "model": "ollama/llama3",
        "temperature": 0,
        "format": "json",  # Ollama needs the format to be specified explicitly
        "model_tokens": 4000,
    },}
)

result = smart_scraper_graph.run()
print(json.dumps(result, indent=4))
