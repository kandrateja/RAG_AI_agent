"""
Main RAG AI Agent Orchestrator
"""
import logging
import re
import json
import hashlib
import base64
from typing import List, Dict, Optional, Union
import uuid

from config import settings
from src.ingestion.docling_processor import DoclingProcessor
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.titan_multimodal import TitanMultimodalEmbeddings
from src.llm.azure_openai_client import AzureOpenAIClient
from src.llm.bedrock_client import BedrockClient
from src.graphrag.neo4j_client import Neo4jClient
from src.graphrag.graph_confidence import GraphConfidenceScorer
from src.graphrag.text_chunker import TextChunker
from src.websearch.surf_client import WebSearchClient
from src.ner.entity_extractor import EntityExtractor
from src.vectorstore.postgres_pgvector import PostgresVectorStore, VectorHit

logger = logging.getLogger(__name__)


class RAGAgent:
    """Main RAG AI Agent that orchestrates OCR, embeddings, GraphRAG, and web search"""
    
    def __init__(self):
        """Initialize RAG Agent with all required components"""
        # Document extraction (Docling)
        self.doc_processor = DoclingProcessor()
        
        # Initialize embedding generator
        # Use Titan Multimodal (text+image) or Azure OpenAI (text-only)
        self.use_titan_embeddings = getattr(settings, "use_titan_embeddings", False)
        
        if self.use_titan_embeddings:
            logger.info("Using Titan Multimodal Embeddings (1024 dims, text+image)")
            self.embedding_generator = TitanMultimodalEmbeddings(
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id or None,
                aws_secret_access_key=settings.aws_secret_access_key or None,
            )
        else:
            logger.info("Using Azure OpenAI Embeddings (1536 dims, text-only)")
            self.embedding_generator = EmbeddingGenerator(
                endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                deployment_name=settings.azure_openai_embedding_deployment_name
            )

        # Initialize Postgres pgvector store (primary vector DB)
        self.vector_store = PostgresVectorStore(
            dsn=settings.postgres_dsn,
            embedding_dim=self.embedding_generator.get_embedding_dimension(),
        )
        self.vector_store.ensure_schema()
        
        # LLM: Bedrock Claude (default) or Azure OpenAI
        if getattr(settings, "use_bedrock_llm", True):
            self.llm_client = BedrockClient(
                region_name=settings.aws_region,
                model_id=settings.bedrock_model_id,
                max_tokens=settings.max_tokens,
                aws_access_key_id=settings.aws_access_key_id or None,
                aws_secret_access_key=settings.aws_secret_access_key or None,
            )
        else:
            self.llm_client = AzureOpenAIClient(
                endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                deployment_name=settings.azure_openai_deployment_name,
                max_tokens=settings.max_tokens
            )
        
        # Initialize Neo4j client (graph DB)
        self.neo4j_client = Neo4jClient(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database
        )
        self.graph_scorer = GraphConfidenceScorer(self.neo4j_client)
        
        # Initialize text chunker
        self.text_chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

        # Initialize web search client (Surf-like API), if configured
        self.web_search_client: Optional[WebSearchClient] = None
        if settings.surf_api_endpoint and settings.surf_api_key:
            self.web_search_client = WebSearchClient(
                endpoint=settings.surf_api_endpoint,
                api_key=settings.surf_api_key,
                default_max_results=settings.surf_max_results,
            )
        
        # Entity extractor: use same LLM as chat (Bedrock or Azure)
        bedrock_for_ner = getattr(settings, "use_bedrock_llm", True)
        self.entity_extractor = EntityExtractor(
            endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            deployment_name=settings.azure_openai_deployment_name,
            llm_client=self.llm_client if bedrock_for_ner else None,
        )
    
    def _compute_document_hash(self, document_path: str) -> str:
        """Compute hash of document for deduplication"""
        try:
            with open(document_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.warning(f"Could not compute document hash: {e}")
            return ""
    
    def ingest_document(
        self,
        document_path: str,
        doc_id: Optional[str] = None,
        source_name: Optional[str] = None,
        pipeline_hint: Optional[str] = None,
    ) -> Dict:
        """
        Ingest a document: Extract (Docling or Azure) -> optional vision fallback -> Chunk -> Embed -> Store.
        pipeline_hint: "auto" | "standard" | "arabic" | "handwritten" (used when ingestion_backend=docling).
        """
        try:
            doc_hash = self._compute_document_hash(document_path)
            if not doc_id:
                doc_id = str(uuid.uuid4())
            
            if self.vector_store.document_exists(doc_id):
                logger.info(f"[INGEST] Document {doc_id} already exists in vector DB. Skipping ingestion.")
                return {
                    "doc_id": doc_id,
                    "status": "skipped",
                    "message": "Document already exists in knowledge base",
                    "chunks_created": 0
                }

            existing_doc_id = self.vector_store.document_exists_by_hash(doc_hash)
            if existing_doc_id:
                logger.info(f"[INGEST] Document already ingested (hash match). Existing doc_id={existing_doc_id}.")
                return {
                    "doc_id": existing_doc_id,
                    "status": "skipped",
                    "message": "Document already exists in knowledge base (hash match)",
                    "chunks_created": 0
                }
            
            # Resolve pipeline hint (param > settings > auto)
            hint = (pipeline_hint or getattr(settings, "ingestion_pipeline_hint", None) or "auto").strip().lower()
            
            # Step 1: Extract text (Docling or Azure) with page info
            logger.info(f"[INGEST] Extracting text from {document_path} (pipeline_hint={hint})")
            try:
                if isinstance(self.doc_processor, DoclingProcessor):
                    ocr_result = self.doc_processor.extract_text_with_pages(document_path, pipeline_hint=hint)
                else:
                    ocr_result = self.doc_processor.extract_text_with_pages(document_path)
                full_text = ocr_result.get("text", "")
                pages = ocr_result.get("pages", [])
                
                if not full_text and not pages:
                    raise ValueError("No text extracted from document - extraction may have failed")
                if not full_text and pages:
                    full_text = "\n\n".join(p.get("text", "") for p in pages)
                
                # Remove Docling HTML placeholder comments (e.g. <!-- image -->) from text
                def _strip_image_placeholders(s: str) -> str:
                    if not s:
                        return s
                    return re.sub(r"\s*<!--\s*image\s*-->\s*", "\n", s, flags=re.IGNORECASE).strip()
                
                full_text = _strip_image_placeholders(full_text)
                for p in pages:
                    if p.get("text"):
                        p["text"] = _strip_image_placeholders(p["text"])
                
                logger.info(f"[INGEST] Extracted {len(pages)} pages of text")
                # Per-page char counts so you can verify every page has content (e.g. Arabic on all 10 pages)
                if pages:
                    per_page = ", ".join(
                        f"p{p.get('page_number', i+1)}: {len((p.get('text') or '').strip())}"
                        for i, p in enumerate(pages)
                    )
                    logger.info(f"[INGEST] Per-page character counts: {per_page}")
            except Exception as e:
                logger.error(f"[INGEST] Extraction failed: {str(e)}")
                return {
                    "doc_id": doc_id,
                    "status": "error",
                    "error": f"Extraction failed: {str(e)}",
                    "error_type": "ocr_failure"
                }
            
            # Auto-detect document type from extraction result (no user params needed)
            pipeline_used = ocr_result.get("pipeline_used") or hint
            total_extracted_text = sum(len((p.get("text") or "").strip()) for p in pages)
            
            # Detect if this is a handwritten/scanned document (low OCR text)
            is_handwritten = hint == "handwritten" or pipeline_used == "auto-ocr" or (total_extracted_text < 200 and len(pages) > 0)
            
            if "ocr" in (pipeline_used or "").lower() or pipeline_used == "handwritten":
                logger.info("[INGEST] Handwritten/scanned pipeline was used (auto-detected low text)")
            if self._is_predominantly_arabic(full_text) and not is_handwritten:
                logger.info("[INGEST] Document detected as predominantly Arabic - Arabic chunks will be translated to English for embedding (so English queries can retrieve them)")
            
            # For handwritten docs: defer entity extraction until AFTER vision extraction
            # For other docs: extract entities now from OCR text
            entities = []
            relationships = []
            if not is_handwritten:
                # Step 2: Extract entities and relationships (for non-handwritten docs)
                print("[INGEST] Step 2: Extracting entities and relationships")
                logger.info("[INGEST] Extracting entities and relationships")
                # Ensure full_text is built from pages if it's empty or too short
                if not full_text or len(full_text.strip()) < 100:
                    if pages:
                        full_text = "\n\n".join(p.get("text", "") for p in pages if p.get("text"))
                        print(f"[INGEST] Rebuilt full_text from pages: {len(full_text)} chars")
                        logger.info(f"[INGEST] Rebuilt full_text from pages: {len(full_text)} chars")
                print(f"[INGEST] Entity extraction input: {len(full_text)} chars")
                print(f"[INGEST] First 200 chars: {full_text[:200]!r}")
                logger.info(f"[INGEST] Entity extraction input: {len(full_text)} chars (first 200: {full_text[:200]!r})")
                try:
                    entity_result = self.entity_extractor.extract_entities_and_relationships(full_text)
                    entities = entity_result.get("entities", [])
                    relationships = entity_result.get("relationships", [])
                    print(f"[INGEST] Extracted {len(entities)} entities and {len(relationships)} relationships")
                    logger.info(f"[INGEST] Extracted {len(entities)} entities and {len(relationships)} relationships")
                except Exception as e:
                    print(f"[INGEST] Entity extraction failed: {str(e)}")
                    logger.warning(f"[INGEST] Entity extraction failed: {str(e)}")
            else:
                print(f"[INGEST] Handwritten document detected ({total_extracted_text} chars from OCR). Will extract entities after vision processing.")

            # Step 3: Vision for handwritten/scanned documents. Text-only docs: skip images.
            page_captions: Dict[int, str] = {}
            min_chars = getattr(settings, "ingestion_vision_fallback_min_chars", 50)
            
            # Detect if this is an Arabic text document - skip ALL vision/image processing for it
            is_arabic_doc = self._is_predominantly_arabic(full_text)
            
            # Check if this is a handwritten document (explicit hint or auto-detected from low OCR text)
            is_handwritten = hint == "handwritten" or pipeline_used == "auto-ocr"
            total_extracted_text = sum(len((p.get("text") or "").strip()) for p in pages)
            
            # For handwritten: if OCR extracted very little text, use vision as primary extraction method
            if is_handwritten or (total_extracted_text < 200 and len(pages) > 0):
                logger.info(f"[INGEST] Handwritten/scanned document detected (extracted {total_extracted_text} chars). Using vision model for text extraction...")
                is_handwritten = True
            
            use_vision_fallback = is_handwritten or any(
                len((p.get("text") or "").strip()) < min_chars for p in pages
            )
            # Threshold: page is "text-heavy" if it has this many chars (don't treat as diagram)
            min_text_for_text_page = 150
            all_pages_text_heavy = all(
                len((p.get("text") or "").strip()) >= min_text_for_text_page for p in pages
            )
            # Also check average: if avg > 200 chars per page, treat as text doc even if some pages are short
            avg_text_per_page = sum(len((p.get("text") or "").strip()) for p in pages) / len(pages) if pages else 0
            
            # COMPLETELY SKIP vision for Arabic documents - they are text documents, not image documents
            # But NOT for handwritten documents (they need vision even if Arabic)
            if is_arabic_doc and not is_handwritten:
                print("[INGEST] Arabic document detected - skipping ALL vision/image processing")
                logger.info("[INGEST] Arabic document detected - skipping ALL vision/image processing (text-only extraction)")
                use_vision_fallback = False
                all_pages_text_heavy = True  # Force skip
            
            # Skip Step 3 entirely for text-only docs (e.g. 10-page Arabic): extract text only, no page-as-image
            if use_vision_fallback or not all_pages_text_heavy:
                try:
                    logger.info("[INGEST] Extracting page images for vision captions / fallback")
                    page_images = self.doc_processor.extract_page_images(document_path)
                    logger.info(f"[INGEST] Found {len(page_images)} page images")
                    for page in page_images:
                        page_number = page.get("page_number")
                        image_bytes = page.get("image_bytes")
                        if not image_bytes:
                            continue
                        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                        media_type = "image/png"
                        if image_bytes[:2] == b'\xff\xd8':
                            media_type = "image/jpeg"
                        elif image_bytes[:4] == b'\x89PNG':
                            media_type = "image/png"
                        page_idx = int(page_number or 0)
                        page_text = next((p.get("text", "") for p in pages if p.get("page_number") == page_idx), "")
                        page_text_len = len((page_text or "").strip())
                        
                        # Check if Docling's extraction might be incomplete (e.g., tables with only 1 column)
                        # Signs of incomplete table extraction: has [Table] but few | characters
                        docling_table_incomplete = (
                            "[Table]" in page_text and 
                            page_text.count("|") < 10  # Should have many | for multi-column tables
                        )
                        
                        # Vision fallback: extract from image when:
                        # 1. Page has very little text, OR
                        # 2. This is a handwritten/form document (always use vision to capture all content), OR
                        # 3. Docling's table extraction seems incomplete
                        should_use_vision = (
                            (use_vision_fallback and page_text_len < min_chars) or
                            (is_handwritten and page_text_len < 2000) or  # For handwritten, always try vision unless page is very text-heavy
                            docling_table_incomplete
                        )
                        
                        if should_use_vision:
                            try:
                                # Use different prompts for handwritten vs scanned printed text
                                if is_handwritten:
                                    # Form-aware extraction prompt for handwritten documents
                                    extraction_prompt = (
                                        "This is a scanned form with handwritten entries. Extract ALL information in a STRUCTURED format.\n\n"
                                        "CRITICAL INSTRUCTIONS:\n\n"
                                        "1. TABLES - THIS IS VERY IMPORTANT:\n"
                                        "   - Extract EVERY column and EVERY row of data\n"
                                        "   - Use markdown table format with | separators\n"
                                        "   - Include ALL column headers\n"
                                        "   - Do NOT skip any columns - if a table has 4 columns, output all 4\n"
                                        "   - Example of a 4-column table:\n"
                                        "     | Local Authority | Team | Telephone | Email |\n"
                                        "     |-----------------|------|-----------|-------|\n"
                                        "     | Hartlepool | Adults Team | 01234567 | email@gov.uk |\n"
                                        "     | Durham | Access Team | 01onal234568 | other@gov.uk |\n\n"
                                        "2. FORM FIELDS: For each field, output as 'Field Label: Value'\n\n"
                                        "3. CHECKBOXES: Look carefully for checkbox fields (Yes/No, tick boxes).\n"
                                        "   - If a checkbox is TICKED (✓ or filled): 'Field: Yes [TICKED]' or 'Field: No [TICKED]'\n"
                                        "   - If a checkbox is EMPTY (unfilled): 'Field: Yes [EMPTY]' or 'Field: No [EMPTY]'\n\n"
                                        "4. HANDWRITTEN TEXT: Transcribe exactly what is written, even if messy\n\n"
                                        "5. SECTIONS: Preserve section numbers and headings\n\n"
                                        "6. EMPTY FIELDS: Write 'Field: [BLANK]' for empty fields\n\n"
                                        "EXAMPLE OUTPUT:\n"
                                        "---\n"
                                        "Form - SG01\n"
                                        "Inter-Agency Safeguarding Adults Concern Form\n\n"
                                        "| Local Authority | Team | Telephone Number | Email Address |\n"
                                        "|-----------------|------|------------------|---------------|\n"
                                        "| Hartlepool | Early Intervention Adults Team | 01429 523309 | ISA@hartlepool.gov.uk |\n"
                                        "| Middlesbrough | Adult Access Team | 01642 065070 | adultaccess@middlesbrough.gov.uk |\n\n"
                                        "SECTION 1: DETAILS OF ADULT AT RISK\n"
                                        "Name: Peter Jones\n"
                                        "DOB: 01/01/01\n"
                                        "Gender: Male\n"
                                        "Home Address: 1 The Front, Hartlepool\n"
                                        "Post Code: TS25 2SQ\n"
                                        "Interpreter needed?: No [TICKED]\n"
                                        "---\n\n"
                                        "Now extract ALL content from this page, including ALL table columns:"
                                    )
                                else:
                                    extraction_prompt = (
                                        "Extract all text visible on this page as plain text. "
                                        "Preserve structure (paragraphs, lists, tables) where possible. "
                                        "Include handwritten or printed text in any language (e.g. Arabic, English)."
                                    )
                                
                                vision_text = self.llm_client.chat_completion_with_image(
                                    prompt=extraction_prompt,
                                    image_base64=image_b64,
                                    media_type=media_type,
                                    max_completion_tokens=4096 if is_handwritten else 1024,  # Increased for detailed form extraction
                                )
                                if vision_text and vision_text.strip():
                                    # For handwritten docs, replace OCR text entirely with vision text (vision is more accurate)
                                    if is_handwritten:
                                        # Log extraction stats for debugging
                                        ticked_count = vision_text.lower().count("[ticked]")
                                        empty_count = vision_text.lower().count("[empty]")
                                        blank_count = vision_text.lower().count("[blank]")
                                        table_count = vision_text.count("|")  # Count pipe characters for table detection
                                        
                                        print(f"[INGEST] Page {page_idx}: Form extraction stats:")
                                        print(f"  - Checkboxes: {ticked_count} ticked, {empty_count} empty")
                                        print(f"  - Blank fields: {blank_count}")
                                        print(f"  - Table columns (| chars): {table_count}")
                                        print(f"  - Total chars: {len(vision_text)}")
                                        
                                        # Log first 500 chars of extraction for debugging
                                        preview = vision_text[:500].replace('\n', '\\n')
                                        print(f"[INGEST] Page {page_idx}: Preview: {preview}...")
                                        
                                        logger.info(f"[INGEST] Page {page_idx}: Form extraction - {ticked_count} ticked, {empty_count} empty, {blank_count} blank, {table_count} table pipes")
                                        
                                        page_captions[page_idx] = f"[Form data - Vision extracted]\n{vision_text.strip()}"
                                        for p in pages:
                                            if p.get("page_number") == page_idx:
                                                p["text"] = vision_text.strip()  # Replace, don't append
                                                break
                                        print(f"[INGEST] Page {page_idx}: Vision extracted {len(vision_text)} chars of form data")
                                    else:
                                        page_captions[page_idx] = f"[Vision-extracted text]\n{vision_text.strip()}"
                                        for p in pages:
                                            if p.get("page_number") == page_idx:
                                                p["text"] = (p.get("text", "") + "\n\n" + vision_text.strip()).strip()
                                                break
                            except Exception as ve:
                                logger.warning(f"[INGEST] Vision fallback failed for page {page_idx}: {ve}")
                        # Only ask "describe diagram" for pages with very little text (likely a figure page)
                        elif page_text_len < min_text_for_text_page:
                            caption = self.llm_client.chat_completion_with_image(
                                prompt=(
                                    "Describe the diagram or visual content on this page. "
                                    "If there is no meaningful diagram, respond with 'No diagram detected.'"
                                ),
                                image_base64=image_b64,
                                media_type=media_type,
                                max_completion_tokens=256,
                            )
                            if caption and "no diagram detected" not in caption.lower():
                                page_captions[page_idx] = caption.strip()
                    if page_captions:
                        logger.info(f"[INGEST] Generated {len(page_captions)} image captions")
                except Exception as e:
                    logger.warning(f"[INGEST] Vision captioning skipped: {str(e)}")
            else:
                logger.info("[INGEST] Text-only document (all pages have substantial text); skipping page-as-image and vision")

            # For handwritten docs: NOW extract entities from the vision-extracted text
            if is_handwritten:
                # Rebuild full_text from pages (now containing vision-extracted text)
                full_text = "\n\n".join(p.get("text", "") for p in pages if p.get("text"))
                print(f"[INGEST] Handwritten: Rebuilt full_text after vision: {len(full_text)} chars")
                logger.info(f"[INGEST] Handwritten: Rebuilt full_text after vision: {len(full_text)} chars")
                
                if full_text and len(full_text.strip()) > 50:
                    print("[INGEST] Step 2 (deferred): Extracting entities from handwritten text")
                    logger.info("[INGEST] Extracting entities and relationships from handwritten text")
                    print(f"[INGEST] Entity extraction input: {len(full_text)} chars")
                    print(f"[INGEST] First 200 chars: {full_text[:200]!r}")
                    try:
                        entity_result = self.entity_extractor.extract_entities_and_relationships(full_text)
                        entities = entity_result.get("entities", [])
                        relationships = entity_result.get("relationships", [])
                        print(f"[INGEST] Extracted {len(entities)} entities and {len(relationships)} relationships")
                        logger.info(f"[INGEST] Extracted {len(entities)} entities and {len(relationships)} relationships")
                    except Exception as e:
                        print(f"[INGEST] Entity extraction failed: {str(e)}")
                        logger.warning(f"[INGEST] Entity extraction failed: {str(e)}")
                else:
                    print(f"[INGEST] Not enough text for entity extraction: {len(full_text)} chars")

            # Step 3b: Extract embedded images for multimodal embedding (if using Titan)
            # SKIP entirely for text-only documents (e.g. Arabic PDF): no vision, no image detection/extraction
            # Also skip for handwritten (we already have page images as captions)
            embedded_images: List[Dict] = []
            avg_text_per_page = sum(len((p.get("text") or "").strip()) for p in pages) / len(pages) if pages else 0
            is_text_document = all_pages_text_heavy or is_arabic_doc or (avg_text_per_page > 200 and len(pages) > 3) or is_handwritten
            
            if self.use_titan_embeddings and not is_text_document:
                try:
                    logger.info("[INGEST] Extracting embedded images (diagrams, figures)")
                    embedded_images = self.doc_processor.extract_embedded_images(document_path)
                    logger.info(f"[INGEST] Found {len(embedded_images)} embedded images")
                    
                    # FALLBACK: Only when no embedded images AND some pages have very little text (likely diagram pages).
                    # Do NOT treat text-only pages as images (e.g. 10-page Arabic doc: extract text only).
                    if not embedded_images:
                        # Only consider a page a "figure" page if it has very little text (likely a diagram, not body text)
                        min_text_to_be_diagram_page = 200
                        figure_pages = {
                            p.get("page_number")
                            for p in pages
                            if len((p.get("text") or "").strip()) < min_text_to_be_diagram_page
                        }
                        if figure_pages:
                            logger.info(f"[INGEST] Low-text pages treated as potential figures: {sorted(figure_pages)}")
                            page_images = self.doc_processor.extract_page_images(
                                document_path,
                                page_numbers=sorted(figure_pages),
                                dpi=150,
                            )
                            embedded_images = page_images
                            logger.info(f"[INGEST] Rendered {len(embedded_images)} page images for figure fallback")
                        else:
                            logger.info("[INGEST] No embedded images and no low-text pages; document treated as text-only (no page images stored)")
                        
                except Exception as e:
                    logger.warning(f"[INGEST] Image extraction failed: {e}")
            elif self.use_titan_embeddings and is_text_document:
                if is_handwritten:
                    reason = "handwritten (pages already processed via vision)"
                elif is_arabic_doc:
                    reason = "Arabic document"
                elif all_pages_text_heavy:
                    reason = "all pages text-heavy"
                else:
                    reason = f"avg {avg_text_per_page:.0f} chars/page"
                logger.info(f"[INGEST] Text-only document ({reason}): skipping embedded image extraction (no vision, no image detection)")

            # Step 4: Chunk the text with page information
            logger.info("[INGEST] Chunking document text")
            
            # Detect if this is a form document (has structured fields like checkboxes)
            combined_text = "\n".join(p.get("text", "") for p in pages) if pages else full_text
            is_form_document = bool(
                "[ticked]" in combined_text.lower() or 
                "[empty]" in combined_text.lower() or
                "[blank]" in combined_text.lower() or
                (combined_text and "section" in combined_text.lower()[:500])  # Forms typically have sections
            )
            
            if is_form_document:
                logger.info("[INGEST] Form document detected - using larger chunks to preserve form structure")
                print("[INGEST] Form document detected - using larger chunks (2000 chars) to keep form fields together")
            
            if pages:
                # Merge page-level captions into page text (skip placeholder-only captions like "Image 1")
                if page_captions:
                    for page in pages:
                        page_number = int(page.get("page_number", 0) or 0)
                        caption = page_captions.get(page_number)
                        if caption:
                            caption_clean = (caption or "").strip()
                            # Do not replace/append if caption is just placeholder (e.g. "Image 1", "Image one")
                            if caption_clean and not re.match(
                                r"^(image\s*(one|two|three|\d+)|figure\s*\d+|page\s*\d*|no diagram).*$",
                                caption_clean, re.IGNORECASE
                            ):
                                page_text = page.get("text", "")
                                page["text"] = f"{page_text}\n\nImage caption: {caption_clean}".strip()
                
                # For form documents, use larger chunks to keep form sections together
                if is_form_document:
                    # Create a temporary chunker with larger size for forms
                    form_chunker = TextChunker(chunk_size=2000, chunk_overlap=300)
                    chunks = form_chunker.chunk_text_by_pages(
                        pages,
                        metadata={"doc_id": doc_id, "file_path": document_path, "doc_hash": doc_hash, "is_form": True}
                    )
                else:
                    chunks = self.text_chunker.chunk_text_by_pages(
                        pages,
                        metadata={"doc_id": doc_id, "file_path": document_path, "doc_hash": doc_hash}
                    )
            else:
                chunks = self.text_chunker.chunk_text(
                    full_text,
                    metadata={"doc_id": doc_id, "file_path": document_path, "doc_hash": doc_hash}
                )
            
            if not chunks:
                raise ValueError("No chunks created from document")
            
            total_chunk_chars = sum(len((c.get("content") or "")) for c in chunks)
            avg_chunk = total_chunk_chars // len(chunks) if chunks else 0
            logger.info(f"[INGEST] Created {len(chunks)} chunks (avg {avg_chunk} chars/chunk, total {total_chunk_chars} chars)")
            
            # Step 5: Generate embeddings for chunks
            logger.info(f"[INGEST] Generating embeddings for {len(chunks)} chunks")
            try:
                # For Titan (English-only): translate Arabic chunks to English for embedding
                # so English queries retrieve them. Original text is kept for display.
                translate_arabic = getattr(settings, "translate_arabic_for_embedding", False) and self.use_titan_embeddings
                chunk_texts_for_embedding = []
                for chunk in chunks:
                    content = chunk.get("content") or ""
                    if translate_arabic and self._is_predominantly_arabic(content):
                        translated = self._translate_arabic_to_english(content)
                        chunk_texts_for_embedding.append(translated if translated else content)
                        logger.debug("[INGEST] Using English translation for embedding (Arabic chunk)")
                    else:
                        chunk_texts_for_embedding.append(content)
                if self.use_titan_embeddings:
                    embeddings = self.embedding_generator.embed_texts(chunk_texts_for_embedding)
                else:
                    embeddings = self.embedding_generator.generate_embeddings_batch(chunk_texts_for_embedding)
                logger.info(f"[INGEST] Generated {len(embeddings)} text embeddings")
            except Exception as e:
                logger.error(f"[INGEST] Embedding generation failed: {str(e)}")
                return {
                    "doc_id": doc_id,
                    "status": "error",
                    "error": f"Embedding generation failed: {str(e)}",
                    "error_type": "embedding_failure"
                }
            
            # Step 5b: Generate embeddings for images (if using Titan Multimodal)
            image_embeddings: List[Dict] = []
            if self.use_titan_embeddings and embedded_images:
                logger.info(f"[INGEST] Generating embeddings for {len(embedded_images)} images")
                try:
                    for img in embedded_images:
                        img_embedding = self.embedding_generator.embed_image(img["image_bytes"])
                        image_embeddings.append({
                            "page_number": img["page_number"],
                            "embedding": img_embedding,
                            "image_b64": base64.b64encode(img["image_bytes"]).decode("utf-8"),
                            "format": img.get("format", "jpeg"),
                        })
                    logger.info(f"[INGEST] Generated {len(image_embeddings)} image embeddings")
                except Exception as e:
                    logger.warning(f"[INGEST] Image embedding failed: {e}")
            
            # Step 5: Store document + chunks in Postgres pgvector (primary KB)
            #         + create ChunkRef nodes in Neo4j to enable graph-time expansion
            logger.info("[INGEST] Storing document/chunks in Postgres (pgvector) + chunk refs in Neo4j")
            try:
                source_path = source_name or document_path
                self.vector_store.upsert_document(doc_id=doc_id, doc_hash=doc_hash, source_path=source_path)
                
                stored_chunks = []
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    page_number = chunk.get("page_number")

                    self.vector_store.upsert_chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        page_number=page_number,
                        chunk_index=i,
                        text=chunk["content"],
                        embedding=embedding,
                        content_type="text",
                    )
                    # Neo4j lightweight chunk ref
                    self.neo4j_client.upsert_chunk_ref(chunk_id=chunk_id, doc_id=doc_id, page_number=page_number)
                    stored_chunks.append({
                        "chunk_id": chunk_id,
                        "page_number": page_number,
                        "content_type": "text",
                    })
                
                logger.info(f"[INGEST] Stored {len(stored_chunks)} text chunks in Postgres")
                
                # Store image embeddings (if using Titan Multimodal)
                if image_embeddings:
                    for j, img_data in enumerate(image_embeddings):
                        img_chunk_id = f"{doc_id}_img_{j}"
                        img_page = img_data["page_number"]
                        img_text = f"[Image on page {img_page}]"  # Placeholder text for display
                        
                        self.vector_store.upsert_chunk(
                            chunk_id=img_chunk_id,
                            doc_id=doc_id,
                            page_number=img_page,
                            chunk_index=len(chunks) + j,  # After text chunks
                            text=img_text,
                            embedding=img_data["embedding"],
                            image_b64=img_data["image_b64"],
                            content_type="image",
                        )
                        stored_chunks.append({
                            "chunk_id": img_chunk_id,
                            "page_number": img_page,
                            "content_type": "image",
                        })
                    
                    logger.info(f"[INGEST] Stored {len(image_embeddings)} image chunks in Postgres")
                
                # Step 6: Store entities and relationships in graph
                if entities:
                    logger.info("[INGEST] Storing entities and relationships in graph")
                    entity_id_map = {}  # Maps entity name to entity_id
                    entity_nodes = {}  # Maps entity name to node_id
                    
                    for entity in entities:
                        entity_name = entity.get("name", "")
                        if not entity_name:
                            continue
                        
                        entity_id = f"{doc_id}_entity_{abs(hash(entity_name))}"
                        entity_type = entity.get("type", "Entity")
                        
                        # Check if entity already exists (search by name across all Entity labels)
                        with self.neo4j_client.driver.session(database=self.neo4j_client.database) as session:
                            check_query = """
                            MATCH (e)
                            WHERE e.name = $name AND 'Entity' IN labels(e)
                            RETURN e, id(e) as id
                            LIMIT 1
                            """
                            result = session.run(check_query, name=entity_name)
                            existing = result.single()
                        
                        if existing:
                            node_id = existing["id"]
                            entity_id = existing["e"].get("entity_id", entity_id)
                        else:
                            node_id = self.neo4j_client.create_entity_node(
                                entity_id=entity_id,
                                entity_type=entity_type,
                                name=entity_name,
                                properties={"description": entity.get("description", "")}
                            )
                        
                        entity_id_map[entity_name] = entity_id
                        entity_nodes[entity_name] = node_id
                    
                    # Create relationships
                    for rel in relationships:
                        from_entity_name = rel.get("from")
                        to_entity_name = rel.get("to")
                        rel_type = rel.get("type", "RELATED_TO")
                        
                        if from_entity_name in entity_id_map and to_entity_name in entity_id_map:
                            self.neo4j_client.create_entity_relationship(
                                from_entity_id=entity_id_map[from_entity_name],
                                to_entity_id=entity_id_map[to_entity_name],
                                rel_type=rel_type,
                                properties={"description": rel.get("description", "")}
                            )

                    # Link chunks -> mentioned entities (simple: link all entities to all chunks for this doc)
                    # If you want higher precision, we can extract per-chunk entities later.
                    for ch in stored_chunks:
                        for entity_name, entity_id in entity_id_map.items():
                            self.neo4j_client.link_chunk_mentions_entity(ch["chunk_id"], entity_id)
                    
                    logger.info(f"[INGEST] Stored {len(entity_nodes)} entities in Neo4j graph")
                
            except Exception as e:
                logger.error(f"[INGEST] Storage failed: {str(e)}")
                return {
                    "doc_id": doc_id,
                    "status": "error",
                    "error": f"Storage failed: {str(e)}",
                    "error_type": "storage_failure"
                }
            
            # Build result with form-specific information if applicable
            result = {
                "doc_id": doc_id,
                "chunks_created": len(stored_chunks),
                "chunks": stored_chunks,
                "entities_extracted": len(entities),
                "relationships_extracted": len(relationships),
                "status": "success"
            }
            
            # Add form extraction summary if this was a form document
            if is_form_document:
                ticked_total = combined_text.lower().count("[ticked]")
                empty_total = combined_text.lower().count("[empty]")
                blank_total = combined_text.lower().count("[blank]")
                result["form_extraction"] = {
                    "is_form": True,
                    "checkboxes_ticked": ticked_total,
                    "checkboxes_empty": empty_total,
                    "fields_blank": blank_total,
                    "note": "Form fields extracted with checkbox states. Query using field names to find specific values."
                }
                print(f"[INGEST] Form summary: {ticked_total} checkboxes ticked, {empty_total} empty, {blank_total} blank fields")
                logger.info(f"[INGEST] Form extraction complete: {ticked_total} ticked, {empty_total} empty, {blank_total} blank")
            
            return result
            
        except Exception as e:
            logger.error(f"[INGEST] Error ingesting document: {str(e)}", exc_info=True)
            return {
                "doc_id": doc_id,
                "status": "error",
                "error": str(e),
                "error_type": "unknown_error"
            }
    
    def _route_tools(self, question: str) -> Dict[str, bool]:
        """
        Decide which tools to use for a given question.

        Tools:
        - vector: Postgres/pgvector semantic search over internal chunks (PRIMARY internal KB)
        - graph: Neo4j graph expansion over entities/relationships linked to retrieved chunks
        - web: Surf-like web search (ONLY if internal KB insufficient)

        Returns:
            Dict with boolean flags: {use_graph, use_vector, use_web}
        """
        logger.info(f"[ROUTING] Analyzing question: {question[:100]}...")
        
        # Default: always try internal first (vector + optional graph expansion).
        routing = {
            "use_graph": True,
            "use_vector": True,
            "use_web": False,
        }

        lower_q = question.lower()

        # Heuristic cues for web search (but still internal-first; web is *fallback*)
        if any(
            kw in lower_q
            for kw in [
                "latest",
                "current",
                "news",
                "today",
                "this year",
                "price of",
                "stock",
                "weather",
                "internet",
                "website",
            ]
        ):
            routing["use_web"] = True

        # General knowledge cues (biography/definitions) should allow web search.
        if any(
            kw in lower_q
            for kw in [
                "who is",
                "who's",
                "what is",
                "define",
                "biography",
                "about",
            ]
        ):
            routing["use_web"] = True

        # Internal cue detection (references to KB / documents).
        internal_cues = [
            "document",
            "documents",
            "pdf",
            "file",
            "knowledge base",
            "kb",
            "internal",
            "ingest",
            "ingestion",
            "chunk",
            "citation",
        ]
        has_internal_cue = any(kw in lower_q for kw in internal_cues)

        # If Neo4j is unavailable, disable graph expansion (still keep vector)
        if self.neo4j_client is None:
            routing["use_graph"] = False

        # If web search not configured, disable it
        if self.web_search_client is None:
            routing["use_web"] = False

        # Try to refine with an LLM-based router (optional).
        # Important: web remains a fallback; we will only execute web if internal is insufficient.
        try:
            logger.info("[ROUTING] Using LLM-based router")
            routing_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a tool router. Decide which tools to use for a user question.\n\n"
                        "Tools:\n"
                        "- vector: Postgres/pgvector semantic search over internal chunks (primary internal knowledge base).\n"
                        "- graph: Neo4j graph expansion (entities/relationships) linked to retrieved chunks.\n"
                        "- web: online search (Surf API) used only if internal knowledge is insufficient.\n\n"
                        "Rules:\n"
                        "- If the question references 'documents', 'knowledge base', or previously ingested files, "
                        "prefer graph and/or vector.\n"
                        "- If the question is about current events, news, prices, live data, or general world knowledge, "
                        "you may set web=true, but it will still only run if internal evidence is insufficient.\n"
                        "- You may combine tools (e.g., graph + web) if both internal docs and web context matter.\n\n"
                        "Respond with a strict JSON object, no explanation, in the form:\n"
                        '{"use_graph": true/false, "use_vector": true/false, "use_web": true/false}'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}",
                },
            ]
            raw = self.llm_client.chat_completion(routing_prompt)
            parsed = json.loads(raw)
            for key in ["use_graph", "use_vector", "use_web"]:
                if key in parsed and isinstance(parsed[key], bool):
                    routing[key] = parsed[key]
            logger.info(f"[ROUTING] LLM routing result: {routing}")
        except Exception as e:
            logger.debug(f"[ROUTING] Tool routing via LLM failed, using heuristic routing: {e}")

        # External-only questions: skip internal tools unless question references KB/documents.
        external_only = routing["use_web"] and not has_internal_cue
        if external_only:
            routing["use_vector"] = False
            routing["use_graph"] = False

        # Final safety: don't enable tools that are not available
        if self.web_search_client is None:
            routing["use_web"] = False

        logger.info(f"[ROUTING] Final routing decision: {routing}")
        return routing

    def _format_citations(self, chunks: List[Dict]) -> List[Dict]:
        """Format citations for retrieved chunks with content type for provenance"""
        citations = []
        figure_num = 0
        
        for chunk in chunks:
            content_type = chunk.get("content_type", "text")
            
            # Assign figure number for image chunks
            source_label = None
            if content_type == "image":
                figure_num += 1
                source_label = f"Figure {figure_num}"
            
            citation = {
                "chunk_id": chunk.get("chunk_id", "unknown"),
                "doc_id": chunk.get("doc_id", "unknown"),
                "doc_name": chunk.get("doc_name"),
                "page_number": chunk.get("page_number"),
                "similarity": chunk.get("similarity"),
                "semantic_score": chunk.get("semantic_score"),
                "keyword_score": chunk.get("keyword_score"),
                "content_type": content_type,  # "text" or "image"
                "source_label": source_label,  # "Figure 1", "Figure 2", etc. for images
            }
            citations.append(citation)
        return citations
    
    @staticmethod
    def _is_predominantly_arabic(text: str, ratio_threshold: float = 0.2) -> bool:
        """Return True if a meaningful fraction of the text is Arabic script (for translation)."""
        if not text or not text.strip():
            return False
        clean = text.replace(" ", "").replace("\n", "")
        if len(clean) < 10:
            return False
        arabic_count = sum(1 for c in clean if "\u0600" <= c <= "\u06FF")
        return (arabic_count / len(clean)) >= ratio_threshold

    def _translate_arabic_to_english(self, text: str, max_chars: int = 3000) -> Optional[str]:
        """Translate Arabic (or mixed) text to English using the LLM for embedding alignment."""
        if not text or not text.strip():
            return text
        trunc = text[:max_chars] + ("..." if len(text) > max_chars else "")
        try:
            prompt = (
                "Translate the following text to English. "
                "Keep the same meaning and tone. Output ONLY the translation, no preamble.\n\n"
                f"{trunc}"
            )
            out = self.llm_client.chat_completion([{"role": "user", "content": prompt}])
            return (out or "").strip()
        except Exception as e:
            logger.warning(f"[INGEST] Arabic translation failed: {e}")
            return None

    def _is_visual_query(self, question: str) -> bool:
        """
        Detect if a query would benefit from visual content (diagrams, figures, images).
        Uses LLM for smart detection (domain-agnostic) with keyword fallback.
        """
        question_lower = question.lower()
        
        # Quick check: Direct visual keywords - definitely want images (fast path)
        explicit_visual = [
            "diagram", "figure", "image", "picture", "photo", "illustration",
            "chart", "graph", "table", "show me", "visual", "drawing"
        ]
        if any(kw in question_lower for kw in explicit_visual):
            logger.info(f"[VISUAL] Explicit visual keyword detected")
            return True
        
        # Use LLM to intelligently determine if visuals would help
        try:
            classification_prompt = f"""Analyze this question and determine if answering it would benefit from visual content like diagrams, figures, charts, or images.

Question: "{question}"

Consider:
- Does it ask about physical structure, layout, or arrangement?
- Does it ask WHERE something is located spatially?
- Does it ask HOW something looks or is organized?
- Would a diagram or image help explain the answer?
- Is it about something that is typically illustrated (anatomy, architecture, processes, etc.)?

Respond with ONLY one word: "visual" if images/diagrams would help, or "text" if text alone is sufficient."""

            messages = [{"role": "user", "content": classification_prompt}]
            response = self.llm_client.chat_completion(messages)
            
            is_visual = "visual" in response.lower()
            logger.info(f"[VISUAL] LLM classification: {'visual' if is_visual else 'text'} for: {question[:50]}...")
            return is_visual
            
        except Exception as e:
            logger.warning(f"[VISUAL] LLM classification failed: {e}, using fallback")
            
            # Fallback to keyword heuristics if LLM fails
            fallback_keywords = [
                "where", "located", "structure", "anatomy", "shape", "layers",
                "position", "arrangement", "components", "parts"
            ]
            return any(kw in question_lower for kw in fallback_keywords)

    def _compress_image_b64(self, image_b64: str, max_size_bytes: int = 500_000, max_dimension: int = 1024) -> str:
        """
        Compress a base64 image to fit within size limits.
        Returns compressed base64 string.
        """
        try:
            from PIL import Image
            import io
            
            # Decode base64
            img_bytes = base64.b64decode(image_b64)
            
            # If already small enough, return as-is
            if len(img_bytes) <= max_size_bytes:
                return image_b64
            
            # Open and resize
            img = Image.open(io.BytesIO(img_bytes))
            
            # Convert RGBA to RGB for JPEG
            if img.mode == "RGBA":
                img = img.convert("RGB")
            
            # Resize if too large
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            # Compress with decreasing quality until small enough
            for quality in [85, 70, 50, 35, 20]:
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality, optimize=True)
                compressed = buffer.getvalue()
                if len(compressed) <= max_size_bytes:
                    return base64.b64encode(compressed).decode("utf-8")
            
            # Last resort: return heavily compressed
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=15, optimize=True)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
            
        except Exception as e:
            logger.warning(f"Image compression failed: {e}, using original")
            return image_b64

    def _build_multimodal_context(self, chunks: List[Dict], question: str) -> List[Dict]:
        """
        Build multimodal message content from chunks that may include images.
        Returns a list of content blocks for Claude's multimodal API.
        """
        content_blocks = []
        
        # Add context header with grounding instruction
        content_blocks.append({
            "type": "text",
            "text": "=== RETRIEVED CONTEXT (answer ONLY from these sources) ===\n\n"
        })
        
        image_count = 0
        text_count = 0
        max_images = 2  # Limit images to avoid token limit
        
        for i, chunk in enumerate(chunks):
            content_type = chunk.get("content_type", "text")
            doc_name = chunk.get("doc_name", "unknown")
            page_num = chunk.get("page_number", "?")
            
            if content_type == "image" and chunk.get("image_b64"):
                # Limit number of images to avoid token overflow
                if image_count >= max_images:
                    logger.info(f"[MULTIMODAL] Skipping image {image_count + 1}, max {max_images} reached")
                    continue
                    
                image_count += 1
                # Compress image to fit within limits
                image_b64 = self._compress_image_b64(chunk["image_b64"])
                
                # Always use JPEG after compression
                media_type = "image/jpeg"
                
                content_blocks.append({
                    "type": "text",
                    "text": f"\n--- IMAGE SOURCE ---\n[Figure {image_count} from {doc_name}, Page {page_num}]\n"
                })
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64,
                    }
                })
                content_blocks.append({
                    "type": "text",
                    "text": f"[End of Figure {image_count}]\n"
                })
            else:
                text_count += 1
                # Add text block with clear source marker
                chunk_text = chunk.get("content", "")
                # Truncate very long text chunks
                if len(chunk_text) > 2000:
                    chunk_text = chunk_text[:2000] + "... [truncated]"
                content_blocks.append({
                    "type": "text",
                    "text": f"\n--- TEXT SOURCE {text_count} ---\n[Source: {doc_name}, Page {page_num}]\n{chunk_text}\n[End of Source {text_count}]\n"
                })
        
        # Add the question with explicit grounding instruction
        content_blocks.append({
            "type": "text",
            "text": f"""
=== END OF RETRIEVED CONTEXT ===

Sources provided: {image_count} image(s), {text_count} text chunk(s)

Question: {question}

INSTRUCTIONS:
- Answer using ONLY the sources above
- If answering from a diagram/figure, reference it explicitly (e.g., "Figure 1 on page 1 shows...")
- List items as they appear in the source, without adding functional descriptions unless explicitly stated
- End with a brief "Sources:" line citing which figures/pages you used"""
        })
        
        return content_blocks

    def query(self, question: str, top_k: int = 10, use_graph_context: bool = True, history: Optional[List[Dict]] = None) -> Dict:
        """
        Query the RAG system with dynamic tool selection.
        
        Args:
            question: User question
            top_k: Number of relevant chunks to retrieve
            use_graph_context: Backward-compat flag; if False, forces vector-only
            
        Returns:
            Dictionary with answer, retrieved context, citations, and provenance
        """
        logger.info(f"[QUERY] Processing question: {question}")
        
        try:
            vector_high_threshold = 0.7
            vector_low_threshold = 0.3
            question_lower = question.strip().lower()
            question_norm = re.sub(r"[^a-z0-9\s]", " ", question_lower)
            question_norm = re.sub(r"\s+", " ", question_norm).strip()
            small_talk = question_lower in {
                "hi", "hello", "hey", "hey!", "hi!", "hello!", "hola", "good morning", "good evening"
            }
            history_intent_phrases = [
                "what are the questions i asked",
                "questions i asked so far",
                "what did i ask",
                "what have i asked",
                "conversation history",
                "chat history",
                "previous questions",
                "questions so far",
            ]
            history_intent = any(p in question_norm for p in history_intent_phrases)
            if not history_intent:
                history_intent = (
                    "question" in question_norm
                    and "ask" in question_norm
                    and ("so far" in question_norm or "history" in question_norm)
                )
            vector_tool = {
                "type": "function",
                "function": {
                    "name": "vector_search",
                    "description": "Search internal knowledge base documents by semantic similarity.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                            "initial_k": {"type": "integer", "minimum": 1, "maximum": 50},
                        },
                        "required": ["query"],
                    },
                },
            }
            graph_tool = {
                "type": "function",
                "function": {
                    "name": "graph_expand",
                    "description": "Expand graph context using chunk_ids for richer relationships.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chunk_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["chunk_ids"],
                    },
                },
            }
            web_tool = {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for up-to-date or general knowledge.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "required": ["query"],
                    },
                },
            }

            chunks: List[Dict] = []
            entities: List[Dict] = []
            relationships: List[Dict] = []
            web_results: List[Dict] = []
            web_citations: List[Dict] = []
            citations: List[Dict] = []
            tools_used = {"vector": False, "graph": False, "web": False, "direct": False}

            def _tool_call_to_dict(call) -> Dict:
                if hasattr(call, "model_dump"):
                    return call.model_dump()
                return {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }

            internal_sufficient = False
            vector_sufficient = False
            graph_sufficient = False
            vector_best_score: Optional[float] = None
            graph_signal: int = 0
            graph_threshold: int = 4
            graph_confidence_threshold = 0.6
            graph_confidence: Dict[str, float] = {"coverage": 0.0, "path_score": 0.0, "relation_score": 0.0, "confidence": 0.0}
            decision_trace: Dict[str, object] = {}

            def _handle_vector_search(args: Dict) -> Dict:
                nonlocal chunks, vector_sufficient, vector_best_score
                query_text = args.get("query") or question
                k = int(args.get("top_k") or top_k)
                initial_k = int(args.get("initial_k") or max(k, k * 2))
                logger.info("[QUERY] Tool: vector_search")
                
                # Detect if this is a visual query (asking about diagrams/figures/images)
                is_visual = self._is_visual_query(query_text)
                if is_visual:
                    logger.info("[QUERY] Visual query detected - will prioritize image chunks")
                
                # Generate query embedding using Titan or Azure OpenAI
                if self.use_titan_embeddings:
                    query_embedding = self.embedding_generator.embed_text(query_text)
                else:
                    query_embedding = self.embedding_generator.generate_embedding(query_text)
                
                hits: List[VectorHit] = self.vector_store.similarity_search(
                    query_embedding, top_k=initial_k, query_text=query_text
                )
                results = [
                    {
                        "chunk_id": h.chunk_id,
                        "doc_id": h.doc_id,
                        "doc_name": h.doc_name,
                        "page_number": h.page_number,
                        "content": h.text,
                        "similarity": h.score,
                        "semantic_score": h.semantic_score,
                        "keyword_score": h.keyword_score,
                        "source_path": h.source_path,
                        "rerank_score": None,
                        "content_type": h.content_type,
                        "image_b64": h.image_b64,
                    }
                    for h in hits
                ]

                # Graph boost: mark chunks linked to entities in Neo4j
                graph_chunk_ids = set()
                if self.neo4j_client is not None and results:
                    try:
                        seed_chunk_ids = [c["chunk_id"] for c in results]
                        graph_chunk_ids = set(self.neo4j_client.chunk_ids_with_entities(seed_chunk_ids))
                    except Exception:
                        graph_chunk_ids = set()

                # Re-ranking: combine hybrid similarity, keyword overlap, graph boost, and image boost
                query_tokens = set(re.findall(r"[a-z0-9]+", query_text.lower()))
                for item in results:
                    chunk_tokens = set(re.findall(r"[a-z0-9]+", (item.get("content") or "").lower()))
                    overlap = 0.0
                    if query_tokens:
                        overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
                    similarity = item.get("similarity") or 0.0
                    graph_boost = 1.0 if item.get("chunk_id") in graph_chunk_ids else 0.0
                    
                    # Image boost: prioritize image chunks for visual queries
                    image_boost = 0.0
                    if is_visual and item.get("content_type") == "image" and item.get("image_b64"):
                        image_boost = 0.3  # Significant boost for image chunks on visual queries
                    
                    item["rerank_score"] = (0.5 * similarity) + (0.2 * overlap) + (0.1 * graph_boost) + (0.2 * image_boost)

                results.sort(key=lambda x: x.get("rerank_score") or 0.0, reverse=True)
                
                # For visual queries, ensure RELEVANT image chunks are included
                if is_visual:
                    image_chunks = [r for r in results if r.get("content_type") == "image" and r.get("image_b64")]
                    
                    # If no image chunks in similarity results, fetch images from RELEVANT document/pages only
                    if not image_chunks:
                        logger.info("[QUERY] No image chunks from similarity search, fetching from relevant document")
                        
                        # Get (doc_id, page_number) pairs from relevant text chunks
                        relevant_doc_pages = set()  # (doc_id, page_number) tuples
                        relevant_docs = set()
                        for r in results[:k]:
                            doc_id = r.get("doc_id")
                            page_num = r.get("page_number")
                            if doc_id:
                                relevant_docs.add(doc_id)
                                if page_num:
                                    relevant_doc_pages.add((doc_id, page_num))
                        
                        logger.info(f"[QUERY] Relevant docs: {relevant_docs}, doc-pages: {relevant_doc_pages}")
                        
                        try:
                            db_image_hits = self.vector_store.get_image_chunks(limit=20)
                            
                            # STRICT filtering: images MUST be from the relevant document
                            for img_hit in db_image_hits:
                                # Check if image is from a relevant document
                                if img_hit.doc_id not in relevant_docs:
                                    continue  # Skip images from other documents!
                                
                                # Check if image is from same page as relevant text
                                is_same_page = (img_hit.doc_id, img_hit.page_number) in relevant_doc_pages
                                
                                # Include if: same document AND (same page OR we need at least one image)
                                if is_same_page or len(image_chunks) < 1:
                                    image_chunks.append({
                                        "chunk_id": img_hit.chunk_id,
                                        "doc_id": img_hit.doc_id,
                                        "doc_name": img_hit.doc_name,
                                        "page_number": img_hit.page_number,
                                        "content": img_hit.text,
                                        "similarity": 0.7 if is_same_page else 0.5,
                                        "semantic_score": 0.7 if is_same_page else 0.5,
                                        "keyword_score": 0.0,
                                        "source_path": img_hit.source_path,
                                        "rerank_score": 0.7 if is_same_page else 0.5,
                                        "content_type": img_hit.content_type,
                                        "image_b64": img_hit.image_b64,
                                    })
                                    logger.info(f"[QUERY] Found image from {img_hit.doc_name} page {img_hit.page_number} (same_page={is_same_page})")
                                    
                                    # Limit to 2 images from the relevant document
                                    if len(image_chunks) >= 2:
                                        break
                            
                            logger.info(f"[QUERY] Selected {len(image_chunks)} relevant image chunks")
                        except Exception as e:
                            logger.warning(f"[QUERY] Failed to fetch image chunks: {e}")
                    
                    if image_chunks:
                        # Take top k results but ensure relevant image chunks are included
                        top_results = results[:k]
                        existing_image_ids = {r.get("chunk_id") for r in top_results if r.get("content_type") == "image"}
                        
                        # Add only relevant image chunks (max 1-2)
                        images_added = 0
                        for img_chunk in image_chunks:
                            if images_added >= 2:
                                break
                            if img_chunk.get("chunk_id") not in existing_image_ids:
                                # Replace lowest scoring text chunk with image chunk
                                text_indices = [i for i, r in enumerate(top_results) if r.get("content_type") != "image"]
                                if text_indices:
                                    replace_idx = text_indices[-1]
                                    top_results[replace_idx] = img_chunk
                                    existing_image_ids.add(img_chunk.get("chunk_id"))
                                    images_added += 1
                                    logger.info(f"[QUERY] Injected image chunk from page {img_chunk.get('page_number')}")
                        results = top_results
                    else:
                        results = results[:k]
                        logger.warning("[QUERY] Visual query but no relevant image chunks found")
                else:
                    results = results[:k]

                vector_best_score = max([c["similarity"] for c in results if c.get("similarity") is not None], default=None)
                chunks = results  # Keep full data including image_b64 for multimodal context
                vector_sufficient = vector_best_score is not None and vector_best_score >= vector_high_threshold
                
                # Strip image_b64 from tool result to avoid bloating the message (images are huge)
                # The full data is preserved in `chunks` for the multimodal answer generation
                tool_result_chunks = [
                    {k: v for k, v in c.items() if k != "image_b64"}
                    for c in results
                ]
                return {"chunks": tool_result_chunks, "vector_sufficient": vector_sufficient, "best_score": vector_best_score}

            def _handle_graph_expand(args: Dict) -> Dict:
                nonlocal entities, relationships, graph_sufficient, graph_signal
                logger.info("[QUERY] Tool: graph_expand")
                chunk_ids = args.get("chunk_ids") or []
                if not chunk_ids and chunks:
                    chunk_ids = [c["chunk_id"] for c in chunks]
                if not chunk_ids:
                    return {"error": "chunk_ids required"}
                graph_ctx = self.neo4j_client.expand_graph_context(chunk_ids)
                entities = graph_ctx.get("entities", [])
                relationships = graph_ctx.get("relationships", [])
                graph_signal = len(entities) + len(relationships)
                graph_sufficient = graph_signal >= graph_threshold
                return {"entities": entities, "relationships": relationships, "graph_sufficient": graph_sufficient}

            def _handle_web_search(args: Dict) -> Dict:
                logger.info("[QUERY] Tool: web_search")
                if self.web_search_client is None:
                    return {"error": "web_search_client not configured"}
                request_id = str(uuid.uuid4())
                query_text = args.get("query") or question
                max_results = int(args.get("max_results") or settings.surf_max_results)
                results = self.web_search_client.search(query=query_text, max_results=max_results)
                citations = []
                for i, item in enumerate(results, start=1):
                    citations.append(
                        {
                            "web_id": f"web_{i}",
                            "request_id": request_id,
                            "title": item.get("title"),
                            "url": item.get("url"),
                            "snippet": item.get("snippet"),
                        }
                    )
                return {"results": results, "web_citations": citations, "request_id": request_id}

            system_prompt = (
                "You are a tool-using assistant. You must call vector_search first to retrieve internal knowledge. "
                "Only call graph_expand if vector_search results are insufficient. "
                "Only call web_search if internal knowledge (vector + graph) is insufficient. "
                "After tools return, answer the question using the tool outputs. "
                "Respond in the same language as the user's question (e.g. if they ask in English, answer in English; if in Arabic, answer in Arabic). "
                "Do NOT include inline citations, doc_ids, chunk_ids, page numbers, or source notes in the answer body. "
                "The UI will display citations separately. Focus only on the answer. "
                "If vector_search returns no relevant chunks, do not claim internal knowledge. "
                "If insufficient info, say so explicitly without guessing. "
                "Format the response as: "
                "First a single paragraph (no label), "
                "then bullet points (no label),"
                "then a short summary (label)"
            )

            messages: List[Dict] = [{"role": "system", "content": system_prompt}]
            if history:
                for msg in history:
                    role = msg.get("role")
                    content = msg.get("content")
                    if role in {"user", "assistant"} and content:
                        messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": question})

            # History or small-talk shortcut: direct response, no tools.
            if history_intent:
                tools_used["direct"] = True
                direct_messages: List[Dict] = [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. Answer using only the conversation history provided. "
                            "Do not use tools or external sources."
                        ),
                    }
                ]
                if history:
                    for msg in history:
                        role = msg.get("role")
                        content = msg.get("content")
                        if role in {"user", "assistant"} and content:
                            direct_messages.append({"role": role, "content": content})
                direct_messages.append({"role": "user", "content": question})
                answer = self.llm_client.chat_completion(direct_messages)
                return {
                    "question": question,
                    "answer": answer,
                    "retrieved_chunks": [],
                    "citations": [],
                    "entities": [],
                    "relationships": [],
                    "web_results": [],
                    "web_citations": [],
                    "context_used": "",
                    "provenance": "none",
                    "tools_used": tools_used,
                    "tools_satisfied": {"vector": False, "graph": False, "web": False, "direct": True},
                    "sources_used": {"vector": False, "graph": False, "web": False, "direct": True},
                    "has_internal_knowledge": False,
                    "internal_sufficient": False,
                    "decision_trace": {"reason": "conversation_history"},
                }
            if small_talk:
                tools_used["direct"] = True
                answer = self.llm_client.chat_completion(messages)
                return {
                    "question": question,
                    "answer": answer,
                    "retrieved_chunks": [],
                    "citations": [],
                    "entities": [],
                    "relationships": [],
                    "web_results": [],
                    "web_citations": [],
                    "context_used": "",
                    "provenance": "none",
                    "tools_used": tools_used,
                    "tools_satisfied": {"vector": False, "graph": False, "web": False, "direct": True},
                    "sources_used": {"vector": False, "graph": False, "web": False, "direct": True},
                    "has_internal_knowledge": False,
                    "internal_sufficient": False,
                }

            # Step 1: Force vector search
            response = self.llm_client.chat_completion_raw(
                messages=messages, tools=[vector_tool], tool_choice={"type": "function", "function": {"name": "vector_search"}}
            )
            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                tool_calls = [
                    type("obj", (), {"id": "manual_vector_call", "function": type("f", (), {"name": "vector_search", "arguments": json.dumps({"query": question, "top_k": top_k})})})()
                ]
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tool_call_to_dict(tc) for tc in tool_calls],
                }
            )
            for call in tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    logger.warning("[QUERY] Tool arguments were not valid JSON; using empty args")
                    args = {}
                tool_result = _handle_vector_search(args)
                tools_used["vector"] = True
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(tool_result),
                    }
                )

            # Step 2: Graph expansion ONLY when vector score is in [0.4, 0.7)
            if (
                (vector_best_score is not None)
                and (vector_low_threshold <= vector_best_score < vector_high_threshold)
                and chunks
                and use_graph_context
                and self.neo4j_client is not None
            ):
                # Compute graph confidence based on entities linked to retrieved chunks.
                entity_names = []
                try:
                    seed_chunk_ids = [c["chunk_id"] for c in chunks]
                    graph_ctx = self.neo4j_client.expand_graph_context(seed_chunk_ids)
                    entity_names = [e.get("name") for e in graph_ctx.get("entities", []) if e.get("name")]
                except Exception:
                    entity_names = []
                if entity_names:
                    graph_confidence = self.graph_scorer.graph_confidence(entity_names)

                response = self.llm_client.chat_completion_raw(
                    messages=messages, tools=[graph_tool], tool_choice={"type": "function", "function": {"name": "graph_expand"}}
                )
                msg = response.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None) or []
                if not tool_calls:
                    tool_calls = [
                        type("obj", (), {"id": "manual_graph_call", "function": type("f", (), {"name": "graph_expand", "arguments": json.dumps({"chunk_ids": [c["chunk_id"] for c in chunks]})})})()
                    ]
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [_tool_call_to_dict(tc) for tc in tool_calls],
                    }
                )
                for call in tool_calls:
                    try:
                        args = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        logger.warning("[QUERY] Tool arguments were not valid JSON; using empty args")
                        args = {}
                    tool_result = _handle_graph_expand(args)
                    tools_used["graph"] = True
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(tool_result),
                        }
                    )

            internal_sufficient = vector_sufficient or graph_sufficient

            # Step 3: Web search ONLY when vector score < 0.4
            if (
                vector_best_score is not None
                and (
                    vector_best_score < vector_low_threshold
                    or (vector_low_threshold <= vector_best_score < vector_high_threshold and graph_confidence.get("confidence", 0.0) < graph_confidence_threshold)
                )
                and self.web_search_client is not None
            ):
                response = self.llm_client.chat_completion_raw(
                    messages=messages, tools=[web_tool], tool_choice={"type": "function", "function": {"name": "web_search"}}
                )
                msg = response.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None) or []
                if not tool_calls:
                    tool_calls = [
                        type("obj", (), {"id": "manual_web_call", "function": type("f", (), {"name": "web_search", "arguments": json.dumps({"query": question, "max_results": settings.surf_max_results})})})()
                    ]
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [_tool_call_to_dict(tc) for tc in tool_calls],
                    }
                )
                for call in tool_calls:
                    try:
                        args = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        logger.warning("[QUERY] Tool arguments were not valid JSON; using empty args")
                        args = {}
                    tool_result = _handle_web_search(args)
                    web_results = tool_result.get("results", [])
                    web_citations = tool_result.get("web_citations", [])
                    tools_used["web"] = True
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(tool_result),
                        }
                    )

            decision_trace = {
                "vector_high_threshold": vector_high_threshold,
                "vector_low_threshold": vector_low_threshold,
                "vector_best_score": vector_best_score,
                "vector_initial_k": int(max(top_k, top_k * 2)),
                "vector_final_k": top_k,
                "vector_sufficient": vector_sufficient,
                "graph_sufficient": graph_sufficient,
                "graph_confidence": graph_confidence,
                "graph_confidence_threshold": graph_confidence_threshold,
                "internal_sufficient": internal_sufficient,
                "graph_triggered": tools_used["graph"],
                "web_triggered": tools_used["web"],
                "web_trigger_reason": "internal_insufficient" if (tools_used["web"] and not internal_sufficient) else None,
            }

            # If vector score is below low threshold, ignore internal chunks/citations.
            if vector_best_score is not None and vector_best_score < vector_low_threshold:
                chunks = []
                citations = []

            internal_available = len(chunks) > 0
            web_available = len(web_results) > 0

            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Internal citations available: {internal_available}. "
                        f"Web results available: {web_available}. "
                        "Only reference internal knowledge if internal citations are available. "
                        "Only reference web sources if web results are available."
                    ),
                }
            )

            answer = ""
            if internal_available or web_available or entities or relationships:
                # Check if any chunks have images for multimodal context
                has_image_chunks = any(c.get("content_type") == "image" and c.get("image_b64") for c in chunks)
                
                if has_image_chunks and self.use_titan_embeddings:
                    # Build multimodal message with images
                    logger.info("[QUERY] Building multimodal context with images")
                    multimodal_content = self._build_multimodal_context(chunks, question)
                    
                    # Grounded multimodal system prompt
                    grounded_system_prompt = """You are a helpful assistant that answers questions ONLY using the provided context (text and images).

CRITICAL RULES:
1. Answer ONLY using information explicitly shown or stated in the retrieved content
2. Respond in the same language as the user's question (e.g. if they ask in English, answer in English; if in Arabic, answer in Arabic)
3. For diagram/figure questions, explicitly reference "the diagram" or "Figure X" when describing what is shown
4. Do NOT add explanatory details (like functions, mechanisms, processes) unless they are explicitly labeled or described in the source
5. If information comes from an image, say "As shown in the diagram..." or "The figure shows..."
6. Keep answers factual and directly tied to visible labels/text in the sources
7. Include page numbers when citing (e.g., "Figure 1, page 1")

BAD: "Meissner's corpuscles detect light touch" (adding function not shown in diagram)
GOOD: "The diagram shows Meissner's corpuscles in the upper dermis"

Answer the question based strictly on the provided context."""
                    
                    multimodal_messages = [
                        {"role": "system", "content": grounded_system_prompt},
                        {"role": "user", "content": multimodal_content}
                    ]
                    answer = self.llm_client.chat_completion(multimodal_messages)
                else:
                    answer = self.llm_client.chat_completion(messages)
            elif not internal_sufficient and self.web_search_client is None:
                answer = (
                    "I could not find sufficient information in the internal knowledge base, "
                    "and web search is not available."
                )

            if not answer:
                answer = (
                    "I could not find sufficient information to answer your question. "
                    "Please try a different question or provide more context."
                )

            if chunks:
                citations = self._format_citations(chunks)

            has_internal_knowledge = len(chunks) > 0
            if has_internal_knowledge and len(web_results) > 0:
                provenance = "both"
            elif has_internal_knowledge:
                provenance = "internal"
            elif len(web_results) > 0:
                provenance = "online"
            else:
                provenance = "none"

            sources_used = {
                "vector": len(chunks) > 0,
                "graph": len(entities) > 0 or len(relationships) > 0,
                "web": len(web_results) > 0,
                "direct": False,
            }

            return {
                "question": question,
                "answer": answer,
                "retrieved_chunks": chunks,
                "citations": citations,
                "entities": entities,
                "relationships": relationships,
                "web_results": web_results,
                "web_citations": web_citations,
                "decision_trace": decision_trace,
                "context_used": "",
                "provenance": provenance,
                "tools_used": tools_used,
                "tools_satisfied": {
                    "vector": vector_sufficient,
                    "graph": graph_sufficient,
                    "web": len(web_results) > 0,
                    "direct": False,
                },
                "sources_used": sources_used,
                "has_internal_knowledge": has_internal_knowledge,
                "internal_sufficient": internal_sufficient,
            }
            
        except Exception as e:
            logger.error(f"[QUERY] Error querying RAG system: {str(e)}", exc_info=True)
            return {
                "question": question,
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "error": str(e),
                "provenance": "error",
                "tools_used": {},
                "retrieved_chunks": [],
                "citations": [],
                "web_results": []
            }
    
    def ingest_batch(self, document_paths: List[str]) -> List[Dict]:
        """
        Ingest multiple documents
        
        Args:
            document_paths: List of document file paths
            
        Returns:
            List of ingestion results
        """
        results = []
        for doc_path in document_paths:
            result = self.ingest_document(doc_path)
            results.append(result)
        return results
    
    def close(self):
        """Close connections"""
        self.neo4j_client.close()
