"""
Neo4j Client for GraphRAG
"""
from neo4j import GraphDatabase
from typing import List, Dict, Optional, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j database client for GraphRAG"""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j client
        
        Args:
            uri: Neo4j connection URI (e.g., bolt://localhost:7687)
            username: Neo4j username
            password: Neo4j password
            database: Database name (default: neo4j)
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self._verify_connectivity()
    
    def _verify_connectivity(self):
        """Verify connection to Neo4j"""
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
    
    def create_node(
        self,
        label: str,
        properties: Dict[str, Any],
        node_id: Optional[str] = None
    ) -> int:
        """
        Create a node in Neo4j
        
        Args:
            label: Node label
            properties: Node properties dictionary
            node_id: Optional custom node ID (stored as property, not Neo4j internal ID)
            
        Returns:
            Created node internal ID (integer)
        """
        with self.driver.session(database=self.database) as session:
            if node_id:
                properties["id"] = node_id
            
            query = f"""
            CREATE (n:{label} $properties)
            RETURN id(n) as node_id
            """
            result = session.run(query, properties=properties)
            return result.single()["node_id"]
    
    def create_relationship(
        self,
        from_node_id: int,
        to_node_id: int,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """
        Create a relationship between nodes
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            rel_type: Relationship type
            properties: Optional relationship properties
        """
        with self.driver.session(database=self.database) as session:
            if properties:
                query = """
                MATCH (a), (b)
                WHERE id(a) = $from_id AND id(b) = $to_id
                CREATE (a)-[r:%s $properties]->(b)
                RETURN r
                """ % rel_type
                session.run(query, from_id=from_node_id, to_id=to_node_id, properties=properties)
            else:
                query = """
                MATCH (a), (b)
                WHERE id(a) = $from_id AND id(b) = $to_id
                CREATE (a)-[r:%s]->(b)
                RETURN r
                """ % rel_type
                session.run(query, from_id=from_node_id, to_id=to_node_id)
    
    def create_document_node(self, doc_id: str, content: str, metadata: Optional[Dict] = None) -> int:
        """
        Create a document node
        
        Args:
            doc_id: Document identifier
            content: Document content
            metadata: Optional metadata dictionary
            
        Returns:
            Created node internal ID (integer)
        """
        properties = {
            "doc_id": doc_id,
            "content": content,
            **(metadata or {})
        }
        return self.create_node("Document", properties, node_id=doc_id)

    def upsert_chunk_ref(self, chunk_id: str, doc_id: str, page_number: Optional[int] = None) -> int:
        """
        Create a lightweight ChunkRef node used for graph-time joins back to the vector DB.

        We do NOT store embeddings here (those live in Postgres/pgvector).
        """
        properties: Dict[str, Any] = {"chunk_id": chunk_id, "doc_id": doc_id}
        if page_number is not None:
            properties["page_number"] = page_number
        return self.create_node("ChunkRef", properties, node_id=chunk_id)

    def link_chunk_mentions_entity(self, chunk_id: str, entity_id: str) -> None:
        """Create (:ChunkRef {chunk_id})-[:MENTIONS]->(:Entity {entity_id})"""
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (c:ChunkRef {chunk_id: $chunk_id})
                MATCH (e)
                WHERE e.entity_id = $entity_id AND 'Entity' IN labels(e)
                MERGE (c)-[:MENTIONS]->(e)
                """,
                chunk_id=chunk_id,
                entity_id=entity_id,
            )

    def expand_graph_context(self, chunk_ids: List[str], max_entities: int = 20) -> Dict[str, Any]:
        """
        Given seed chunk_ids (from pgvector retrieval), return entities mentioned and their relationships.
        """
        entities: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        if not chunk_ids:
            return {"entities": entities, "relationships": relationships}

        with self.driver.session(database=self.database) as session:
            ent_res = session.run(
                """
                MATCH (c:ChunkRef)-[:MENTIONS]->(e)
                WHERE c.chunk_id IN $chunk_ids AND 'Entity' IN labels(e)
                RETURN DISTINCT e.entity_id AS entity_id, e.name AS name, e.type AS type
                LIMIT $limit
                """,
                chunk_ids=chunk_ids,
                limit=max_entities,
            )
            for r in ent_res:
                entities.append({"entity_id": r["entity_id"], "name": r["name"], "type": r["type"]})

            rel_res = session.run(
                """
                MATCH (c:ChunkRef)-[:MENTIONS]->(e1)-[r]->(e2)
                WHERE c.chunk_id IN $chunk_ids AND 'Entity' IN labels(e1) AND 'Entity' IN labels(e2)
                RETURN DISTINCT e1.name AS from_entity, type(r) AS rel_type, e2.name AS to_entity
                LIMIT 50
                """,
                chunk_ids=chunk_ids,
            )
            for r in rel_res:
                relationships.append({"from": r["from_entity"], "type": r["rel_type"], "to": r["to_entity"]})

        return {"entities": entities, "relationships": relationships}
    
    def create_chunk_node(
        self,
        chunk_id: str,
        content: str,
        embedding: List[float],
        doc_id: str,
        chunk_index: int,
        page_number: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Create a chunk node with embedding
        
        Args:
            chunk_id: Chunk identifier
            content: Chunk content
            embedding: Chunk embedding vector
            doc_id: Parent document ID
            chunk_index: Index of chunk in document
            page_number: Optional page number
            metadata: Optional metadata dictionary
        """
        properties = {
            "chunk_id": chunk_id,
            "content": content,
            "embedding": embedding,
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            **(metadata or {})
        }
        if page_number is not None:
            properties["page_number"] = page_number
        
        node_id = self.create_node("Chunk", properties, node_id=chunk_id)
        
        # Create relationship to parent document
        doc_node = self.get_node_by_property("Document", "doc_id", doc_id)
        if doc_node:
            self.create_relationship(node_id, doc_node["id"], "BELONGS_TO")
        
        return node_id
    
    def get_node_by_property(self, label: str, property_key: str, property_value: Any) -> Optional[Dict]:
        """
        Get a node by property
        
        Args:
            label: Node label
            property_key: Property key to search
            property_value: Property value to match
            
        Returns:
            Node dictionary or None
        """
        with self.driver.session(database=self.database) as session:
            query = f"""
            MATCH (n:{label})
            WHERE n.{property_key} = $value
            RETURN n, id(n) as id
            LIMIT 1
            """
            result = session.run(query, value=property_value)
            record = result.single()
            if record:
                return {"id": record["id"], **dict(record["n"])}
            return None
    
    def document_exists(self, doc_id: str) -> bool:
        """
        Check if a document with the given doc_id already exists
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if document exists, False otherwise
        """
        doc_node = self.get_node_by_property("Document", "doc_id", doc_id)
        return doc_node is not None
    
    def get_chunk_metadata(self, chunk_id: str) -> Optional[Dict]:
        """
        Get full metadata for a chunk including page_number
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk metadata dictionary or None
        """
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (c:Chunk {chunk_id: $chunk_id})
            RETURN c
            LIMIT 1
            """
            result = session.run(query, chunk_id=chunk_id)
            record = result.single()
            if record:
                return dict(record["c"])
            return None
    
    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict]:
        """
        Perform vector similarity search using cosine similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Similarity threshold
            
        Returns:
            List of matching chunks with similarity scores
        """
        with self.driver.session(database=self.database) as session:
            # Create index if it doesn't exist
            self._create_vector_index(session)
            
            query = """
            MATCH (c:Chunk)
            WHERE c.embedding IS NOT NULL
            WITH c, 
                 gds.similarity.cosine(c.embedding, $query_embedding) AS similarity
            WHERE similarity >= $threshold
            RETURN c.content AS content, 
                   c.chunk_id AS chunk_id,
                   c.doc_id AS doc_id,
                   c.page_number AS page_number,
                   similarity
            ORDER BY similarity DESC
            LIMIT $top_k
            """
            
            try:
                result = session.run(
                    query,
                    query_embedding=query_embedding,
                    threshold=threshold,
                    top_k=top_k
                )
                return [
                    {
                        "content": record["content"],
                        "chunk_id": record["chunk_id"],
                        "doc_id": record["doc_id"],
                        "page_number": record.get("page_number"),
                        "similarity": record["similarity"]
                    }
                    for record in result
                ]
            except Exception as e:
                # Fallback to manual cosine similarity if GDS is not available
                logger.warning(f"GDS similarity not available, using manual calculation: {str(e)}")
                return self._manual_vector_search(session, query_embedding, top_k, threshold)
    
    def _manual_vector_search(
        self,
        session,
        query_embedding: List[float],
        top_k: int,
        threshold: float
    ) -> List[Dict]:
        """Manual vector search using Python cosine similarity"""
        
        query = """
        MATCH (c:Chunk)
        WHERE c.embedding IS NOT NULL
        RETURN c.content AS content,
               c.chunk_id AS chunk_id,
               c.doc_id AS doc_id,
               c.page_number AS page_number,
               c.embedding AS embedding
        """
        
        results = session.run(query)
        similarities = []
        
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        
        for record in results:
            chunk_embedding = np.array(record["embedding"])
            chunk_norm = np.linalg.norm(chunk_embedding)
            
            if query_norm > 0 and chunk_norm > 0:
                similarity = np.dot(query_vec, chunk_embedding) / (query_norm * chunk_norm)
                if similarity >= threshold:
                    similarities.append({
                        "content": record["content"],
                        "chunk_id": record["chunk_id"],
                        "doc_id": record["doc_id"],
                        "page_number": record.get("page_number"),
                        "similarity": float(similarity)
                    })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    def _create_vector_index(self, session):
        """Create vector index for embeddings if it doesn't exist"""
        try:
            # Try to create index using Neo4j Vector Index (if available)
            index_query = """
            CREATE INDEX chunk_embedding_index IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            """
            session.run(index_query)
        except Exception as e:
            logger.debug(f"Vector index creation skipped: {str(e)}")
    
    def create_entity_node(self, entity_id: str, entity_type: str, name: str, properties: Optional[Dict] = None):
        """
        Create an entity node for GraphRAG
        
        Args:
            entity_id: Entity identifier
            entity_type: Type of entity (e.g., Person, Organization, Concept)
            name: Entity name
            properties: Optional additional properties
        """
        props = {
            "entity_id": entity_id,
            "name": name,
            "type": entity_type,
            **(properties or {})
        }
        # Create node with Entity label and type stored as property
        # Neo4j supports multiple labels, so we use Entity as base label
        with self.driver.session(database=self.database) as session:
            if entity_id:
                props["id"] = entity_id
            
            # Create with Entity label (can add specific type as second label if needed)
            query = f"""
            CREATE (n:Entity:{entity_type} $properties)
            RETURN id(n) as node_id
            """
            result = session.run(query, properties=props)
            return result.single()["node_id"]
    
    def create_entity_relationship(
        self,
        from_entity_id: str,
        to_entity_id: str,
        rel_type: str,
        properties: Optional[Dict] = None
    ):
        """
        Create relationship between entities
        
        Args:
            from_entity_id: Source entity ID
            to_entity_id: Target entity ID
            rel_type: Relationship type
            properties: Optional relationship properties
        """
        # Search for entities by entity_id (works with any Entity label)
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (e)
            WHERE e.entity_id = $entity_id AND 'Entity' IN labels(e)
            RETURN e, id(e) as id
            LIMIT 1
            """
            from_result = session.run(query, entity_id=from_entity_id)
            to_result = session.run(query, entity_id=to_entity_id)
            
            from_record = from_result.single()
            to_record = to_result.single()
            
            if from_record and to_record:
                self.create_relationship(from_record["id"], to_record["id"], rel_type, properties)
    
    def graph_rag_query(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
        include_relationships: bool = True
    ) -> Dict:
        """
        Perform GraphRAG query combining vector search with graph traversal
        
        Args:
            query: Natural language query
            query_embedding: Query embedding vector
            top_k: Number of initial chunks to retrieve
            include_relationships: Whether to include related entities
            
        Returns:
            Dictionary with relevant chunks and graph context
        """
        # Initial vector search
        initial_results = self.vector_search(query_embedding, top_k=top_k)
        
        if not include_relationships:
            return {"chunks": initial_results, "entities": [], "relationships": []}
        
        # Extract document IDs from results
        doc_ids = list(set([r["doc_id"] for r in initial_results]))
        
        # Get related entities and relationships
        entities = []
        relationships = []
        
        with self.driver.session(database=self.database) as session:
            # Find entities mentioned in the retrieved chunks
            for doc_id in doc_ids:
                entity_query = """
                MATCH (d:Document {doc_id: $doc_id})<-[:BELONGS_TO]-(c:Chunk)
                MATCH (c)-[:MENTIONS]->(e:Entity)
                RETURN DISTINCT e.entity_id AS entity_id,
                               e.name AS name,
                               e.type AS type
                LIMIT 10
                """
                result = session.run(entity_query, doc_id=doc_id)
                for record in result:
                    entities.append({
                        "entity_id": record["entity_id"],
                        "name": record["name"],
                        "type": record["type"]
                    })
                
                # Get relationships between entities
                rel_query = """
                MATCH (d:Document {doc_id: $doc_id})<-[:BELONGS_TO]-(c:Chunk)
                MATCH (c)-[:MENTIONS]->(e1:Entity)-[r]->(e2:Entity)
                RETURN DISTINCT e1.name AS from_entity,
                               type(r) AS rel_type,
                               e2.name AS to_entity
                LIMIT 20
                """
                result = session.run(rel_query, doc_id=doc_id)
                for record in result:
                    relationships.append({
                        "from": record["from_entity"],
                        "type": record["rel_type"],
                        "to": record["to_entity"]
                    })
        
        return {
            "chunks": initial_results,
            "entities": entities,
            "relationships": relationships
        }
    
    def close(self):
        """Close the Neo4j driver connection"""
        self.driver.close()
