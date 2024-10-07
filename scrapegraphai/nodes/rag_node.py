"""
RAGNode Module
"""
from typing import List, Optional
from .base_node import BaseNode
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

class RAGNode(BaseNode):
    """
    A node responsible for compressing the input tokens and storing the document
    in a vector database for retrieval. Relevant chunks are stored in the state.

    It allows scraping of big documents without exceeding the token limit of the language model.

    Attributes:
        llm_model: An instance of a language model client, configured for generating answers.
        verbose (bool): A flag indicating whether to show print statements during execution.

    Args:
        input (str): Boolean expression defining the input keys needed from the state.
        output (List[str]): List of output keys to be updated in the state.
        node_config (dict): Additional configuration for the node.
        node_name (str): The unique identifier name for the node, defaulting to "Parse".
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "RAG",
    ):
        super().__init__(node_name, "node", input, output, 2, node_config)

        self.llm_model = node_config["llm_model"]
        self.embedder_model = node_config.get("embedder_model", None)
        self.verbose = (
            False if node_config is None else node_config.get("verbose", False)
        )

    def execute(self, state: dict) -> dict:
        self.logger.info(f"--- Executing {self.node_name} Node ---")
        
        if self.node_config.get("client_type") in ["memory", None]:
            client = QdrantClient(":memory:")
        elif self.node_config.get("client_type") == "local_db":
            client = QdrantClient(path="path/to/db")
        elif self.node_config.get("client_type") == "image":
            client = QdrantClient(url="http://localhost:6333")
        else:
            raise ValueError("client_type provided not correct")

        docs = state.get("docs")

        if state.get("embeddings"):
            collection_name = "collection"
            vector_size = 1536
            
            client.create_collection(
                collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            
            import openai
            openai_client = openai.Client()

            # Prepare and insert data
            points = []

            for idx, doc in enumerate(docs):
                # Encode summary
                #summary_embedding = encoder.encode(doc["summary"]).tolist()
                summary_embedding = openai_client.embeddings.create(input=doc["summary"],
                                                             model=state.get("embeddings").get("model"))
                
                # Create point
                point = PointStruct(
                    id=idx,
                    vector={"summary_vector": summary_embedding},
                    payload={
                        "summary": doc["summary"],
                        "document": doc["document"],
                        "metadata": doc["metadata"]
                    }
                )
                points.append(point)

            # Insert points into collection
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            state["vectorial_db"] = client
            return state

        for idx, doc in enumerate(docs):
            # Create point
            point = PointStruct(
                id=idx,
                payload={
                    "summary": doc["summary"],
                    "document": doc["document"],
                    "metadata": doc["metadata"]
                }
            )
            points.append(point)

        # Insert points into collection
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        state["vectorial_db"] = client
        return state
