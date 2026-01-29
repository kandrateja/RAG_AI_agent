"""
Neo4j Client for GraphRAG
"""
from neo4j import GraphDatabase
from typing import List, Dict, Optional, Any
import logging
import re
import json

logger = logging.getLogger(__name__)


def _sanitize_label(label: str) -> str:
    """
    Sanitize a string to be used as a Neo4j label or relationship type.
    Neo4j labels cannot have spaces or special characters (except underscore).
    """
    if not label:
        return "Unknown"
    # Replace spaces and hyphens with underscores
    sanitized = re.sub(r'[\s\-]+', '_', label.strip())
    # Remove any other non-alphanumeric characters (except underscore)
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', sanitized)
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = '_' + sanitized
    return sanitized or "Unknown"


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
            logger.info("Neo4j connection verified")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
    
    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def create_node(
        self,
        label: str,
        properties: Dict[str, Any],
        node_id: Optional[str] = None
    ) -> str:
        """
        Create a node in Neo4j using MERGE to avoid duplicates.
        Returns the elementId of the node.
        """
        safe_label = _sanitize_label(label)
        with self.driver.session(database=self.database) as session:
            if node_id:
                properties["id"] = node_id
            
            # Use MERGE to avoid duplicates when we have an id
            if node_id:
                query = f"""
                MERGE (n:{safe_label} {{id: $node_id}})
                SET n += $properties
                RETURN elementId(n) as node_id
                """
                result = session.run(query, node_id=node_id, properties=properties)
            else:
                query = f"""
                CREATE (n:{safe_label} $properties)
                RETURN elementId(n) as node_id
                """
                result = session.run(query, properties=properties)
            return result.single()["node_id"]
    
    def create_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """
        Create a relationship between nodes using elementId.
        """
        safe_rel_type = _sanitize_label(rel_type)
        
        with self.driver.session(database=self.database) as session:
            if properties:
                query = f"""
                MATCH (a), (b)
                WHERE elementId(a) = $from_id AND elementId(b) = $to_id
                MERGE (a)-[r:{safe_rel_type}]->(b)
                SET r += $props
                RETURN r
                """
                session.run(query, from_id=from_node_id, to_id=to_node_id, props=properties)
            else:
                query = f"""
                MATCH (a), (b)
                WHERE elementId(a) = $from_id AND elementId(b) = $to_id
                MERGE (a)-[r:{safe_rel_type}]->(b)
                RETURN r
                """
                session.run(query, from_id=from_node_id, to_id=to_node_id)

    def upsert_document_node(self, doc_id: str, source_path: Optional[str] = None) -> None:
        """
        Create or update a Document node in Neo4j.
        """
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MERGE (d:Document {doc_id: $doc_id})
                ON CREATE SET d.source_path = $source_path
                ON MATCH SET d.source_path = $source_path
                """,
                doc_id=doc_id,
                source_path=source_path or "",
            )

    def link_chunk_ref_to_document(self, chunk_id: str, doc_id: str) -> None:
        """Create (ChunkRef)-[:BELONGS_TO]->(Document)"""
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (c:ChunkRef {chunk_id: $chunk_id})
                MATCH (d:Document {doc_id: $doc_id})
                MERGE (c)-[:BELONGS_TO]->(d)
                """,
                chunk_id=chunk_id,
                doc_id=doc_id,
            )

    def upsert_chunk_ref(self, chunk_id: str, doc_id: str, page_number: Optional[int] = None) -> str:
        """
        Create a lightweight ChunkRef node used for graph-time joins back to the vector DB.
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

    def _ensure_mentions_type_exists(self) -> None:
        """
        Ensure the MENTIONS relationship type exists in the DB so Cypher queries
        don't raise UnknownRelationshipTypeWarning when no entities were linked yet.
        """
        with self.driver.session(database=self.database) as session:
            r = session.run("MATCH ()-[r:MENTIONS]->() RETURN count(r) AS c")
            row = r.single()
            if row and row["c"] and row["c"] > 0:
                return
            session.run(
                """
                MERGE (c:ChunkRef {chunk_id: '__sentinel_mentions__', doc_id: '__sentinel__'})
                MERGE (e:Entity {entity_id: '__sentinel_mentions__', name: '', type: 'Unknown'})
                MERGE (c)-[:MENTIONS]->(e)
                """
            )
            logger.debug("Created sentinel MENTIONS relationship")

    def expand_graph_context(self, chunk_ids: List[str], max_entities: int = 20) -> Dict[str, Any]:
        """
        Given seed chunk_ids (from pgvector retrieval), return entities mentioned and their relationships.
        """
        entities: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        if not chunk_ids:
            return {"entities": entities, "relationships": relationships}

        self._ensure_mentions_type_exists()

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
                if r["entity_id"] == "__sentinel_mentions__":
                    continue
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

    def chunk_ids_with_entities(self, chunk_ids: List[str]) -> List[str]:
        """
        Return chunk_ids that have at least one linked Entity in the graph.
        """
        if not chunk_ids:
            return []
        self._ensure_mentions_type_exists()
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (c:ChunkRef)-[:MENTIONS]->(e)
                WHERE c.chunk_id IN $chunk_ids AND 'Entity' IN labels(e)
                RETURN DISTINCT c.chunk_id AS chunk_id
                """,
                chunk_ids=chunk_ids,
            )
            return [r["chunk_id"] for r in result if r["chunk_id"] != "__sentinel_mentions__"]

    def create_entity_node(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create an Entity node with a specific type label.
        """
        safe_type = _sanitize_label(entity_type)
        props = {
            "entity_id": entity_id,
            "name": name,
            "type": entity_type,
            **(properties or {})
        }
        
        with self.driver.session(database=self.database) as session:
            query = f"""
            MERGE (e:Entity:{safe_type} {{entity_id: $entity_id}})
            SET e += $props
            RETURN elementId(e) as node_id
            """
            result = session.run(query, entity_id=entity_id, props=props)
            return result.single()["node_id"]

    def create_entity_relationship(
        self,
        from_entity_id: str,
        to_entity_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create a relationship between two Entity nodes by entity_id.
        """
        safe_rel_type = _sanitize_label(rel_type)
        
        with self.driver.session(database=self.database) as session:
            if properties:
                query = f"""
                MATCH (a), (b)
                WHERE a.entity_id = $from_id AND 'Entity' IN labels(a)
                  AND b.entity_id = $to_id AND 'Entity' IN labels(b)
                MERGE (a)-[r:{safe_rel_type}]->(b)
                SET r += $props
                """
                session.run(query, from_id=from_entity_id, to_id=to_entity_id, props=properties)
            else:
                query = f"""
                MATCH (a), (b)
                WHERE a.entity_id = $from_id AND 'Entity' IN labels(a)
                  AND b.entity_id = $to_id AND 'Entity' IN labels(b)
                MERGE (a)-[r:{safe_rel_type}]->(b)
                """
                session.run(query, from_id=from_entity_id, to_id=to_entity_id)

    def get_node_by_property(self, label: str, prop_name: str, prop_value: Any) -> Optional[Dict]:
        """
        Get a node by a property value.
        """
        safe_label = _sanitize_label(label)
        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"""
                MATCH (n:{safe_label} {{{prop_name}: $value}})
                RETURN n, elementId(n) as id
                LIMIT 1
                """,
                value=prop_value,
            )
            record = result.single()
            if record:
                node = dict(record["n"])
                node["id"] = record["id"]
                return node
            return None
