"""
Entity and Relationship Extraction using Azure OpenAI
"""
import logging
import json
from typing import List, Dict, Optional
from openai import AzureOpenAI

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extract entities and relationships from text using Azure OpenAI"""
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployment_name: str
    ):
        """
        Initialize entity extractor
        
        Args:
            endpoint: Azure OpenAI endpoint
            api_key: Azure OpenAI API key
            api_version: API version
            deployment_name: Deployment name for the model
        """
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.deployment_name = deployment_name
    
    def extract_entities_and_relationships(
        self,
        text: str,
        max_entities: int = 50
    ) -> Dict:
        """
        Extract entities and relationships from text
        
        Args:
            text: Input text to analyze
            max_entities: Maximum number of entities to extract
            
        Returns:
            Dictionary with entities and relationships
        """
        try:
            prompt = f"""Extract entities and relationships from the following text.

Text:
{text[:4000]}  # Limit text to avoid token limits

Extract:
1. Named entities (Person, Organization, Location, Concept, Date, etc.)
2. Relationships between entities

Return a JSON object with this structure:
{{
    "entities": [
        {{"name": "entity_name", "type": "EntityType", "description": "brief description"}}
    ],
    "relationships": [
        {{"from": "entity1", "to": "entity2", "type": "RELATIONSHIP_TYPE", "description": "brief description"}}
    ]
}}

Limit to {max_entities} entities maximum. Focus on the most important entities and relationships.
Return only valid JSON, no additional text."""

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at extracting named entities and relationships from text. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Some Azure OpenAI models (e.g., GPT-5.x) only allow default temperature.
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_completion_tokens=2000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON from response
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            
            return {
                "entities": result.get("entities", []),
                "relationships": result.get("relationships", [])
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse entity extraction JSON: {e}")
            return {"entities": [], "relationships": []}
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return {"entities": [], "relationships": []}
