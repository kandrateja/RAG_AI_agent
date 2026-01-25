"""
Main RAG AI Agent Orchestrator
"""
import logging
import json
import hashlib
from typing import List, Dict, Optional
import uuid

from config import settings
from src.ocr.document_processor import DocumentProcessor
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.llm.azure_openai_client import AzureOpenAIClient
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
        # Initialize OCR processor
        self.ocr_processor = DocumentProcessor(
            endpoint=settings.azure_document_intelligence_endpoint,
            key=settings.azure_document_intelligence_key
        )
        
        # Initialize embedding generator
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
        
        # Initialize LLM client
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
        
        # Initialize entity extractor
        self.entity_extractor = EntityExtractor(
            endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            deployment_name=settings.azure_openai_deployment_name
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
    
    def ingest_document(self, document_path: str, doc_id: Optional[str] = None) -> Dict:
        """
        Ingest a document: OCR -> Chunk -> Embed -> Store in Neo4j
        
        Args:
            document_path: Path to the document file
            doc_id: Optional document ID (generated if not provided)
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            # Check for deduplication
            doc_hash = self._compute_document_hash(document_path)
            
            if not doc_id:
                doc_id = str(uuid.uuid4())
            
            # Deduplication: if doc_id exists in Postgres, skip.
            if self.vector_store.document_exists(doc_id):
                logger.info(f"[INGEST] Document {doc_id} already exists in vector DB. Skipping ingestion.")
                return {
                    "doc_id": doc_id,
                    "status": "skipped",
                    "message": "Document already exists in knowledge base",
                    "chunks_created": 0
                }

            # Deduplication by content hash (if same PDF re-uploaded with different doc_id)
            existing_doc_id = self.vector_store.document_exists_by_hash(doc_hash)
            if existing_doc_id:
                logger.info(f"[INGEST] Document already ingested (hash match). Existing doc_id={existing_doc_id}.")
                return {
                    "doc_id": existing_doc_id,
                    "status": "skipped",
                    "message": "Document already exists in knowledge base (hash match)",
                    "chunks_created": 0
                }
            
            # Step 1: Extract text using Azure Document Intelligence with page info
            logger.info(f"[INGEST] Extracting text from {document_path}")
            try:
                ocr_result = self.ocr_processor.extract_text_with_pages(document_path)
                full_text = ocr_result.get("text", "")
                pages = ocr_result.get("pages", [])
                
                if not full_text:
                    raise ValueError("No text extracted from document - OCR may have failed")
                
                logger.info(f"[INGEST] Extracted {len(pages)} pages of text")
            except Exception as e:
                logger.error(f"[INGEST] OCR extraction failed: {str(e)}")
                return {
                    "doc_id": doc_id,
                    "status": "error",
                    "error": f"OCR extraction failed: {str(e)}",
                    "error_type": "ocr_failure"
                }
            
            # Step 2: Extract entities and relationships
            logger.info("[INGEST] Extracting entities and relationships")
            try:
                entity_result = self.entity_extractor.extract_entities_and_relationships(full_text)
                entities = entity_result.get("entities", [])
                relationships = entity_result.get("relationships", [])
                logger.info(f"[INGEST] Extracted {len(entities)} entities and {len(relationships)} relationships")
            except Exception as e:
                logger.warning(f"[INGEST] Entity extraction failed: {str(e)}")
                entities = []
                relationships = []
            
            # Step 3: Chunk the text with page information
            logger.info("[INGEST] Chunking document text")
            if pages:
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
            
            logger.info(f"[INGEST] Created {len(chunks)} chunks")
            
            # Step 4: Generate embeddings for chunks
            logger.info(f"[INGEST] Generating embeddings for {len(chunks)} chunks")
            try:
                chunk_texts = [chunk["content"] for chunk in chunks]
                embeddings = self.embedding_generator.generate_embeddings_batch(chunk_texts)
                logger.info(f"[INGEST] Generated {len(embeddings)} embeddings")
            except Exception as e:
                logger.error(f"[INGEST] Embedding generation failed: {str(e)}")
                return {
                    "doc_id": doc_id,
                    "status": "error",
                    "error": f"Embedding generation failed: {str(e)}",
                    "error_type": "embedding_failure"
                }
            
            # Step 5: Store document + chunks in Postgres pgvector (primary KB)
            #         + create ChunkRef nodes in Neo4j to enable graph-time expansion
            logger.info("[INGEST] Storing document/chunks in Postgres (pgvector) + chunk refs in Neo4j")
            try:
                self.vector_store.upsert_document(doc_id=doc_id, doc_hash=doc_hash, source_path=document_path)
                
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
                    )
                    # Neo4j lightweight chunk ref
                    self.neo4j_client.upsert_chunk_ref(chunk_id=chunk_id, doc_id=doc_id, page_number=page_number)
                    stored_chunks.append({
                        "chunk_id": chunk_id,
                        "page_number": page_number
                    })
                
                logger.info(f"[INGEST] Stored {len(stored_chunks)} chunks in Postgres")
                
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
            
            return {
                "doc_id": doc_id,
                "chunks_created": len(stored_chunks),
                "chunks": stored_chunks,
                "entities_extracted": len(entities),
                "relationships_extracted": len(relationships),
                "status": "success"
            }
            
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
        """Format citations for retrieved chunks"""
        citations = []
        for chunk in chunks:
            citation = {
                "chunk_id": chunk.get("chunk_id", "unknown"),
                "doc_id": chunk.get("doc_id", "unknown"),
                "doc_name": chunk.get("doc_name"),
                "page_number": chunk.get("page_number"),
                "similarity": chunk.get("similarity")
            }
            citations.append(citation)
        return citations

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
            small_talk = question.strip().lower() in {
                "hi", "hello", "hey", "hey!", "hi!", "hello!", "hola", "good morning", "good evening"
            }
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
                logger.info("[QUERY] Tool: vector_search")
                query_embedding = self.embedding_generator.generate_embedding(query_text)
                hits: List[VectorHit] = self.vector_store.similarity_search(query_embedding, top_k=k)
                results = [
                    {
                        "chunk_id": h.chunk_id,
                        "doc_id": h.doc_id,
                        "doc_name": h.doc_name,
                        "page_number": h.page_number,
                        "content": h.text,
                        "similarity": h.score,
                        "source_path": h.source_path,
                    }
                    for h in hits
                ]
                vector_best_score = max([c["similarity"] for c in results if c.get("similarity") is not None], default=None)
                chunks = results
                vector_sufficient = vector_best_score is not None and vector_best_score >= vector_high_threshold
                return {"chunks": results, "vector_sufficient": vector_sufficient, "best_score": vector_best_score}

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
                "Always include citations for internal KB content using doc_id, page_number, and chunk_id. "
                "State whether the answer used internal knowledge, web, or both. "
                "If vector_search returns no relevant chunks, do not claim internal knowledge. "
                "If insufficient info, say so explicitly without guessing."
            )

            messages: List[Dict] = [{"role": "system", "content": system_prompt}]
            if history:
                for msg in history:
                    role = msg.get("role")
                    content = msg.get("content")
                    if role in {"user", "assistant"} and content:
                        messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": question})

            # Small-talk shortcut: direct response, no tools.
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
