"""
GenerateAnswerNodeKLevel Module
"""
from typing import List, Optional
from langchain.prompts import PromptTemplate
from tqdm import tqdm
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_aws import ChatBedrock
from qdrant_client import models
from ..utils.output_parser import get_structured_output_parser, get_pydantic_output_parser
from .base_node import BaseNode
from ..prompts import (
    TEMPLATE_CHUNKS, TEMPLATE_NO_CHUNKS, TEMPLATE_MERGE,
    TEMPLATE_CHUNKS_MD, TEMPLATE_NO_CHUNKS_MD, TEMPLATE_MERGE_MD
)

class GenerateAnswerNodeKLevel(BaseNode):
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
        node_name: str = "GANLK",
    ):
        super().__init__(node_name, "node", input, output, 2, node_config)

        self.llm_model = node_config["llm_model"]
        self.embedder_model = node_config.get("embedder_model", None)
        self.verbose = node_config.get("verbose", False)
        self.force = node_config.get("force", False)
        self.script_creator = node_config.get("script_creator", False)
        self.is_md_scraper = node_config.get("is_md_scraper", False)
        self.additional_info = node_config.get("additional_info")

    def execute(self, state: dict) -> dict:
        self.logger.info(f"--- Executing {self.node_name} Node ---")

        user_prompt = state.get("user_prompt")

        if self.node_config.get("schema", None) is not None:
            if isinstance(self.llm_model, (ChatOpenAI, ChatMistralAI)):
                self.llm_model = self.llm_model.with_structured_output(
                    schema=self.node_config["schema"]
                )
                output_parser = get_structured_output_parser(self.node_config["schema"])
                format_instructions = "NA"
            else:
                if not isinstance(self.llm_model, ChatBedrock):
                    output_parser = get_pydantic_output_parser(self.node_config["schema"])
                    format_instructions = output_parser.get_format_instructions()
                else:
                    output_parser = None
                    format_instructions = ""
        else:
            if not isinstance(self.llm_model, ChatBedrock):
                output_parser = JsonOutputParser()
                format_instructions = output_parser.get_format_instructions()
            else:
                output_parser = None
                format_instructions = ""

        if isinstance(self.llm_model, (ChatOpenAI, AzureChatOpenAI)) \
            and not self.script_creator \
            or self.force \
            and not self.script_creator or self.is_md_scraper:
            template_no_chunks_prompt = TEMPLATE_NO_CHUNKS_MD
            template_chunks_prompt = TEMPLATE_CHUNKS_MD
            template_merge_prompt = TEMPLATE_MERGE_MD
        else:
            template_no_chunks_prompt = TEMPLATE_NO_CHUNKS
            template_chunks_prompt = TEMPLATE_CHUNKS
            template_merge_prompt = TEMPLATE_MERGE

        if self.additional_info is not None:
            template_no_chunks_prompt = self.additional_info + template_no_chunks_prompt
            template_chunks_prompt = self.additional_info + template_chunks_prompt
            template_merge_prompt = self.additional_info + template_merge_prompt

        client = state["vectorial_db"]

        search_results = self.hybrid_search(
            query=user_prompt,
            client=client,
            collection_name="vectorial_collection",
            threshold=0.5,
            summary_weight=0.7,
            document_weight=0.3
        )

        if not search_results:
            state["answer"] = "No results found."
            return state
        
        chains_dict = {}
        
        # Process each search result in parallel
        for i, result in enumerate(tqdm(search_results, 
                                   desc="Setting up chunk processing", 
                                   disable=not self.verbose)):
            # Create a prompt for each search result
            prompt = PromptTemplate(
                template=template_chunks_prompt,
                input_variables=["format_instructions"],
                partial_variables={
                    "context": result['payload']['document'],
                    "chunk_id": i + 1
                }
            )
            
            chain_name = f"chunk{i+1}"
            chains_dict[chain_name] = prompt | self.llm_model
        
        # Execute all chains in parallel
        async_runner = RunnableParallel(**chains_dict)
        batch_results = async_runner.invoke({"format_instructions": user_prompt})

        # Create and execute merge prompt
        merge_prompt = PromptTemplate(
            template=template_merge_prompt,
            input_variables=["context", "question"],
            partial_variables={"format_instructions": format_instructions}
        )

        merge_chain = merge_prompt | self.llm_model
        if output_parser:
            merge_chain = merge_chain | output_parser
        answer = merge_chain.invoke({
            "context": batch_results,
            "question": user_prompt
        })

        return answer

    def hybrid_search(
        self,
        query: str,
        client,
        collection_name,
        threshold=0.7,  # Adjust this threshold based on your needs
        summary_weight=0.7,  # Weight for summary relevance (0.0 to 1.0)
        document_weight=0.3  # Weight for document relevance (0.0 to 1.0)
    ):
        """
        Perform a hybrid search using a single query string with threshold and weighted scoring.
        If embeddings are available, performs both vector and text search; otherwise, only text search.
        
        Parameters:
        - query: str, the user's search query
        - client: the database client to perform the search
        - collection_name: str, the collection name to search in
        - threshold: float, minimum score for results (0.0 to 1.0)
        - summary_weight: float, weight for summary relevance (0.0 to 1.0)
        - document_weight: float, weight for document relevance (0.0 to 1.0)
        """
        combined_results = {}
        
        # 1. Vector search on summaries (if embeddings are available)
        if self.embedder_model is not None:
            if self.embedder_model.get("source") == "openai":
                import openai
                openai_client = openai.Client()

                query_vector=openai_client.embeddings.create(
                    input=query,
                    model=self.embedder_model.get("model"),
                )
            
            elif self.embedder_model.get("source") == "huggingface":
                from transformers import AutoModel, AutoTokenizer
                import torch
                
                # Load Hugging Face model and tokenizer
                model_name = self.embedder_model.get("model")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                
                # Tokenize and generate embedding for summary
                inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Take the mean of token embeddings to create a single vector for the document
                query_vector = torch.mean(outputs.last_hidden_state, dim=1).squeeze().tolist()
            
            else:
                raise ValueError("Invalid embedder source")
            
            vector_results = client.search(
                collection_name=collection_name,
                query_vector={"summary_vector": query_vector},
                search_params=models.SearchParams(
                    hnsw_ef=128,
                    exact=False
                ),
                limit=100,  # High limit to get all potential matches
                score_threshold=threshold * 0.8,  # Slightly lower threshold for initial fetch
                with_payload=True,
            )

            # Process vector results
            for result in vector_results:
                combined_results[result.id] = {
                    'point_id': result.id,
                    'payload': result.payload,
                    'summary_score': result.score * summary_weight,
                    'document_score': 0
                }
            
        # 2. Text search on documents
        text_results = client.search(
            collection_name=collection_name,
            query_filter=models.Filter(
                should=[
                    models.FieldCondition(
                        key="document",
                        match=models.MatchText(text=query)
                    ),
                    models.FieldCondition(
                        key="summary",
                        match=models.MatchText(text=query)
                    )
                ]
            ),
            limit=100,  # High limit to get all potential matches
            with_payload=True,
        )
        
        # Process text search results and merge with vector results if available
        for result in text_results:
            if result.id in combined_results:
                # Merge document score with existing summary score
                combined_results[result.id]['document_score'] = result.score * document_weight
            else:
                # Add new result for cases without vector result
                combined_results[result.id] = {
                    'point_id': result.id,
                    'payload': result.payload,
                    'summary_score': 0,  # No vector score
                    'document_score': result.score * document_weight
                }
        
        # Adjust weights if no embeddings
        if self.embedder_model is None:
            # Use full weight for document score when no embeddings
            for result_data in combined_results.values():
                result_data['document_score'] /= document_weight  # Remove original weight
                result_data['document_score'] *= 1.0  # Apply full weight
        
        # 3. Calculate final scores and filter by threshold
        final_results = []
        for result_data in combined_results.values():
            final_score = result_data['summary_score'] + result_data['document_score']
            if final_score >= threshold:
                result_entry = {
                    'id': result_data['point_id'],
                    'score': final_score,
                    'payload': result_data['payload'],
                }
                
                # Add separate scores only if embeddings were used
                if self.embedder_model is not None:
                    result_entry.update({
                        'summary_score': result_data['summary_score'] / summary_weight,
                        'document_score': result_data['document_score'] / document_weight
                    })
                
                final_results.append(result_entry)
        
        # Sort by final score
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        return final_results