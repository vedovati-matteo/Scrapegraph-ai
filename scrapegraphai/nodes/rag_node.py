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
        
        # Initialize client based on config
        if self.node_config.get("client_type") in ["memory", None]:
            client = QdrantClient(":memory:")
        elif self.node_config.get("client_type") == "local_db":
            client = QdrantClient(path="path/to/db")
        elif self.node_config.get("client_type") == "image":
            client = QdrantClient(url="http://localhost:6333")
        else:
            raise ValueError("client_type provided not correct")

        docs = state.get("docs")
        collection_name = "collection"
        points = []

        # Handle case with embeddings
        if self.embedder_model:
            client.create_collection(
                collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            
            if self.embedder_model.get("source") == "openai":
                import openai
                openai_client = openai.Client()
            
                vector_size = 1536

                for idx, doc in enumerate(docs):
                    # Generate embedding for summary
                    summary_embedding = openai_client.embeddings.create(
                        input=doc["summary"],
                        model=self.embedder_model.get("model")
                    ).data[0].embedding
                    
                    # Create point with vector
                    point = PointStruct(
                        id=idx,
                        vector=summary_embedding,
                        payload={
                            "summary": doc["summary"],
                            "document": doc["document"],
                            "metadata": doc["metadata"]
                        }
                    )
                    points.append(point)
            
            elif self.embedder_model.get("source") == "huggingface":
                from transformers import AutoModel, AutoTokenizer
                import torch
                
                # Load Hugging Face model and tokenizer
                model_name = self.embedder_model.get("model")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                
                vector_size = model.config.hidden_size

                for idx, doc in enumerate(docs):
                    # Tokenize and generate embedding for summary
                    inputs = tokenizer(doc["summary"], return_tensors="pt", padding=True, truncation=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    # Take the mean of token embeddings to create a single vector for the document
                    summary_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().tolist()

                    # Create point with vector
                    point = PointStruct(
                        id=idx,
                        vector=summary_embedding,
                        payload={
                            "summary": doc["summary"],
                            "document": doc["document"],
                            "metadata": doc["metadata"]
                        }
                    )
                    points.append(point)
            
            else:
                raise ValueError("Embedder model source not supported")
            
        # Handle case without embeddings
        else:
            # Create collection without vector configuration
            client.create_collection(
                collection_name,
                vectors_config=None
            )
            
            for idx, doc in enumerate(docs):
                # Create point without vector
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
