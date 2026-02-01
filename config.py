"""
Configuration settings for RAG AI Agent.
Loaded from environment variables and .env.
"""
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Ingestion pipeline hint
    ingestion_pipeline_hint: Optional[str] = None  # auto | standard | arabic | handwritten

    # Azure OpenAI (required for embeddings; used for LLM only if use_bedrock_llm is False)
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_deployment_name: str = ""
    azure_openai_embedding_deployment_name: str = "text-embedding-3-small"

    # Amazon Bedrock (Claude for LLM + vision)
    use_bedrock_llm: bool = True
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    # Titan Multimodal Embeddings
    use_titan_embeddings: bool = True
    titan_embedding_model_id: str = "amazon.titan-embed-image-v1"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    # Postgres (pgvector)
    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/rag"

    # Web Search
    surf_api_endpoint: Optional[str] = None
    surf_api_key: Optional[str] = None
    surf_max_results: int = 5

    # Application
    chunk_size: int = 1500
    chunk_overlap: int = 300
    max_tokens: int = 4096
    temperature: float = 0.7
    ingestion_vision_fallback_min_chars: int = 50

    # Arabic / cross-lingual: translate Arabic to English for embedding (Titan is English-only)
    # So English queries can retrieve Arabic content. Original Arabic is kept for display.
    translate_arabic_for_embedding: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
