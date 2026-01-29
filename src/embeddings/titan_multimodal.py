"""
Amazon Bedrock Titan Multimodal Embeddings client.
Supports embedding both text and images into the same vector space.
"""
import json
import base64
import logging
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    BOTO_AVAILABLE = True
except ImportError:
    BOTO_AVAILABLE = False
    boto3 = None  # type: ignore


class TitanMultimodalEmbeddings:
    """
    Bedrock Titan Multimodal Embeddings G1.
    Embeds text and images into the same 1024-dimensional vector space.
    
    This enables:
    - Text-to-text similarity search
    - Image-to-image similarity search
    - Text-to-image cross-modal search (find images by text query)
    """
    
    MODEL_ID = "amazon.titan-embed-image-v1"
    EMBEDDING_DIM = 1024
    
    def __init__(
        self,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        if not BOTO_AVAILABLE:
            raise RuntimeError("boto3 is not installed. Install with: pip install boto3")
        
        self.region_name = region_name
        kwargs = {"service_name": "bedrock-runtime", "region_name": region_name}
        if aws_access_key_id and aws_secret_access_key:
            kwargs["aws_access_key_id"] = aws_access_key_id
            kwargs["aws_secret_access_key"] = aws_secret_access_key
        
        self._client = boto3.client(**kwargs)
    
    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension (1024 for Titan Multimodal)."""
        return self.EMBEDDING_DIM
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            1024-dimensional embedding vector
        """
        body = json.dumps({"inputText": text})
        
        try:
            response = self._client.invoke_model(
                modelId=self.MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            result = json.loads(response["body"].read())
            return result["embedding"]
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Titan text embedding error: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple text strings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of 1024-dimensional embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_image(
        self,
        image_data: Union[bytes, str],
        text_description: Optional[str] = None,
    ) -> List[float]:
        """
        Embed an image (optionally with text description).
        
        Args:
            image_data: Image bytes or base64-encoded string
            text_description: Optional text to embed alongside the image
            
        Returns:
            1024-dimensional embedding vector
        """
        # Convert bytes to base64 if needed
        if isinstance(image_data, bytes):
            image_b64 = base64.b64encode(image_data).decode("utf-8")
        else:
            image_b64 = image_data
        
        body_dict = {"inputImage": image_b64}
        if text_description:
            body_dict["inputText"] = text_description
        
        body = json.dumps(body_dict)
        
        try:
            response = self._client.invoke_model(
                modelId=self.MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            result = json.loads(response["body"].read())
            return result["embedding"]
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Titan image embedding error: {e}")
            raise
    
    def embed_text_and_image(
        self,
        text: str,
        image_data: Union[bytes, str],
    ) -> List[float]:
        """
        Embed text and image together (joint embedding).
        
        Args:
            text: Text content
            image_data: Image bytes or base64-encoded string
            
        Returns:
            1024-dimensional joint embedding vector
        """
        return self.embed_image(image_data, text_description=text)
