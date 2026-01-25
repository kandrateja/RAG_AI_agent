"""
Azure OpenAI Embedding Generator
"""
import os
from openai import AzureOpenAI
from typing import List, Union, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using Azure OpenAI"""
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployment_name: str = "text-embedding-3-small"
    ):
        """
        Initialize Azure OpenAI client for embeddings
        
        Args:
            endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            api_version: API version
            deployment_name: Deployment name for embedding model (default: text-embedding-3-large)
        """
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.deployment_name = deployment_name
    
    def generate_embedding(self, text: str, dimensions: Optional[int] = None) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            dimensions: Optional number of dimensions (for text-embedding-3 models)
            
        Returns:
            List of embedding values
        """
        try:
            kwargs = {"model": self.deployment_name, "input": text}
            if dimensions is not None:
                kwargs["dimensions"] = dimensions
            response = self.client.embeddings.create(**kwargs)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        dimensions: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts to embed
            dimensions: Optional number of dimensions (for text-embedding-3 models)
            
        Returns:
            List of embedding vectors
        """
        try:
            kwargs = {"model": self.deployment_name, "input": texts}
            if dimensions is not None:
                kwargs["dimensions"] = dimensions
            response = self.client.embeddings.create(**kwargs)
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for the current model
        
        Returns:
            Embedding dimension size
        """
        # text-embedding-3-large: 3072 dimensions
        # text-embedding-3-small: 1536 dimensions
        if "large" in self.deployment_name.lower():
            return 3072
        else:
            return 1536
