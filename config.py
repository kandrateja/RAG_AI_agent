"""
Configuration settings for RAG AI Agent
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Azure Document Intelligence
    azure_document_intelligence_endpoint: str
    azure_document_intelligence_key: str
    
    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_deployment_name: str
    azure_openai_embedding_deployment_name: str = "text-embedding-3-small"
    
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str
    neo4j_database: str = "neo4j"

    # Postgres (pgvector)
    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/rag"
    
    # Web Search (Surf-like API)
    surf_api_endpoint: Optional[str] = None
    surf_api_key: Optional[str] = None
    surf_max_results: int = 5
    
    # Application Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens: int = 4096
    temperature: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
