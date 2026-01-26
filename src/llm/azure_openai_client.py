"""
Azure OpenAI LLM Client
"""
from openai import AzureOpenAI
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AzureOpenAIClient:
    """Azure OpenAI LLM client for chat completions"""
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployment_name: str,
        max_tokens: int = 4096
    ):
        """
        Initialize Azure OpenAI client
        
        Args:
            endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            api_version: API version
            deployment_name: Deployment name for the model
            max_tokens: Maximum tokens to generate (default: 4096)
        """
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.deployment_name = deployment_name
        self.max_tokens = max_tokens
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_completion_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate chat completion
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            **kwargs: Additional parameters
            
        Returns:
            Generated response text
        """
        try:
            req = {
                "model": self.deployment_name,
                "messages": messages,
                "max_completion_tokens": max_completion_tokens or self.max_tokens,
                **kwargs,
            }
            response = self.client.chat.completions.create(**req)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise

    def chat_completion_raw(
        self,
        messages: List[Dict[str, str]],
        max_completion_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Generate chat completion and return the full response object.
        """
        try:
            req = {
                "model": self.deployment_name,
                "messages": messages,
                "max_completion_tokens": max_completion_tokens or self.max_tokens,
                **kwargs,
            }
            return self.client.chat.completions.create(**req)
        except Exception as e:
            logger.error(f"Error in raw chat completion: {str(e)}")
            raise
    
    def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_completion_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Generate streaming chat completion
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            **kwargs: Additional parameters
            
        Yields:
            Response chunks
        """
        try:
            req = {
                "model": self.deployment_name,
                "messages": messages,
                "max_completion_tokens": max_completion_tokens or self.max_tokens,
                "stream": True,
                **kwargs,
            }
            stream = self.client.chat.completions.create(**req)
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error in streaming chat completion: {str(e)}")
            raise

    def chat_completion_with_image(
        self,
        prompt: str,
        image_base64: str,
        max_completion_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate chat completion with an image input (vision-capable models).
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        },
                    ],
                }
            ]
            req = {
                "model": self.deployment_name,
                "messages": messages,
                "max_completion_tokens": max_completion_tokens or self.max_tokens,
                **kwargs,
            }
            response = self.client.chat.completions.create(**req)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in image chat completion: {str(e)}")
            raise
