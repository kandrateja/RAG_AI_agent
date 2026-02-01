"""
Amazon Bedrock Titan Embeddings client.
Supports:
- Titan Multimodal V1 (amazon.titan-embed-image-v1): Text + Image, English only, 1024 dims
- Titan Text V2 (amazon.titan-embed-text-v2:0): Text only, Multilingual (100+ languages), 1024 dims

Hybrid approach: Use V2 for multilingual text (Arabic, etc.), V1 for images.
"""
import json
import base64
import hashlib
import logging
from typing import List, Optional, Union, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    BOTO_AVAILABLE = True
except ImportError:
    BOTO_AVAILABLE = False
    boto3 = None  # type: ignore


class TranslationCache:
    """
    Simple file-based cache for Arabic-to-English translations.
    Avoids re-translating the same chunks.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "rag_translations"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, str] = {}
    
    def _get_key(self, text: str) -> str:
        """Generate cache key from text hash."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    
    def get(self, text: str) -> Optional[str]:
        """Get cached translation if exists."""
        key = self._get_key(text)
        
        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Check file cache
        cache_file = self.cache_dir / f"{key}.txt"
        if cache_file.exists():
            translation = cache_file.read_text(encoding="utf-8")
            self._memory_cache[key] = translation
            return translation
        
        return None
    
    def set(self, text: str, translation: str) -> None:
        """Cache a translation."""
        key = self._get_key(text)
        self._memory_cache[key] = translation
        
        # Write to file cache
        cache_file = self.cache_dir / f"{key}.txt"
        cache_file.write_text(translation, encoding="utf-8")
    
    def clear(self) -> None:
        """Clear all cached translations."""
        self._memory_cache.clear()
        for f in self.cache_dir.glob("*.txt"):
            f.unlink()


class TitanMultimodalEmbeddings:
    """
    Bedrock Titan Embeddings with Hybrid Model Support.
    
    - V1 (amazon.titan-embed-image-v1): Multimodal (text + image), English only
    - V2 (amazon.titan-embed-text-v2:0): Multilingual text (100+ languages including Arabic)
    
    This enables:
    - Native Arabic text embeddings (no translation needed) with V2
    - Image embeddings with V1
    - Cross-modal text-to-image search with V1
    """
    
    # Model IDs
    MODEL_ID_V1 = "amazon.titan-embed-image-v1"      # Multimodal, English only
    MODEL_ID_V2 = "amazon.titan-embed-text-v2:0"    # Text only, Multilingual
    
    EMBEDDING_DIM = 1024  # Both models use 1024 dimensions
    
    def __init__(
        self,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        use_v2_for_text: bool = False,  # If True, use V2 for all text (multilingual)
        translation_cache_dir: Optional[str] = None,
    ):
        if not BOTO_AVAILABLE:
            raise RuntimeError("boto3 is not installed. Install with: pip install boto3")
        
        self.region_name = region_name
        self.use_v2_for_text = use_v2_for_text
        
        kwargs = {"service_name": "bedrock-runtime", "region_name": region_name}
        if aws_access_key_id and aws_secret_access_key:
            kwargs["aws_access_key_id"] = aws_access_key_id
            kwargs["aws_secret_access_key"] = aws_secret_access_key
        
        self._client = boto3.client(**kwargs)
        
        # Translation cache (for fallback when V2 is not used)
        self.translation_cache = TranslationCache(translation_cache_dir)
        
        if use_v2_for_text:
            logger.info("Using Titan V2 for text embeddings (multilingual support)")
        else:
            logger.info("Using Titan V1 for text embeddings (English only, with translation for Arabic)")
    
    # Legacy property for backward compatibility
    @property
    def MODEL_ID(self) -> str:
        return self.MODEL_ID_V2 if self.use_v2_for_text else self.MODEL_ID_V1
    
    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension (1024 for both Titan V1 and V2)."""
        return self.EMBEDDING_DIM
    
    def _is_arabic(self, text: str, threshold: float = 0.2) -> bool:
        """Check if text contains significant Arabic content."""
        if not text or len(text.strip()) < 10:
            return False
        clean = text.replace(" ", "").replace("\n", "")
        arabic_count = sum(1 for c in clean if "\u0600" <= c <= "\u06FF")
        return (arabic_count / len(clean)) >= threshold if clean else False
    
    def embed_text(self, text: str, force_v2: bool = False) -> List[float]:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            force_v2: If True, force use of V2 model (multilingual)
            
        Returns:
            1024-dimensional embedding vector
        """
        # Determine which model to use
        use_v2 = force_v2 or self.use_v2_for_text
        model_id = self.MODEL_ID_V2 if use_v2 else self.MODEL_ID_V1
        
        # V2 has different request format
        if use_v2:
            body = json.dumps({
                "inputText": text,
                "dimensions": self.EMBEDDING_DIM,
                "normalize": True
            })
        else:
            body = json.dumps({"inputText": text})
        
        try:
            response = self._client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            result = json.loads(response["body"].read())
            return result["embedding"]
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Titan text embedding error (model={model_id}): {e}")
            raise
    
    def embed_texts(self, texts: List[str], force_v2: bool = False) -> List[List[float]]:
        """
        Embed multiple text strings.
        
        Args:
            texts: List of texts to embed
            force_v2: If True, force use of V2 model (multilingual)
            
        Returns:
            List of 1024-dimensional embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text, force_v2=force_v2)
            embeddings.append(embedding)
        return embeddings
    
    def embed_text_smart(self, text: str, translate_func=None) -> List[float]:
        """
        Smart text embedding that automatically handles Arabic.
        
        - If V2 is enabled: embed directly (native multilingual)
        - If V2 is disabled but text is Arabic: translate first (with caching), then embed with V1
        - Otherwise: embed directly with V1
        
        Args:
            text: Text to embed
            translate_func: Optional function to translate Arabic to English (signature: str -> str)
            
        Returns:
            1024-dimensional embedding vector
        """
        if self.use_v2_for_text:
            # V2 supports Arabic natively - no translation needed
            return self.embed_text(text, force_v2=True)
        
        # V1 path: check if Arabic and needs translation
        if self._is_arabic(text):
            # Check cache first
            cached = self.translation_cache.get(text)
            if cached:
                logger.debug("Using cached translation for Arabic text")
                return self.embed_text(cached, force_v2=False)
            
            # Translate if function provided
            if translate_func:
                translated = translate_func(text)
                if translated:
                    self.translation_cache.set(text, translated)
                    logger.debug("Cached new Arabic translation")
                    return self.embed_text(translated, force_v2=False)
            
            # Fallback: embed Arabic directly (may have lower quality)
            logger.warning("Embedding Arabic text directly with V1 (no translation)")
        
        return self.embed_text(text, force_v2=False)
    
    def embed_texts_smart(self, texts: List[str], translate_func=None) -> List[List[float]]:
        """
        Smart batch text embedding with automatic Arabic handling.
        
        Args:
            texts: List of texts to embed
            translate_func: Optional function to translate Arabic to English
            
        Returns:
            List of 1024-dimensional embedding vectors
        """
        return [self.embed_text_smart(text, translate_func) for text in texts]
    
    def embed_image(
        self,
        image_data: Union[bytes, str],
        text_description: Optional[str] = None,
    ) -> List[float]:
        """
        Embed an image (optionally with text description).
        
        NOTE: Always uses V1 (amazon.titan-embed-image-v1) since V2 doesn't support images.
        
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
            # Always use V1 for images (V2 is text-only)
            response = self._client.invoke_model(
                modelId=self.MODEL_ID_V1,
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
