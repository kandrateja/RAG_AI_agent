"""
Graph confidence scoring utilities for Neo4j.
"""
from typing import List, Dict, Optional


class GraphConfidenceScorer:
    def __init__(self, neo4j_client):
        self._client = neo4j_client

    def _session(self):
        return self._client.driver.session(database=self._client.database)

    def entity_coverage(self, entities: List[str]) -> float:
        if not entities:
            return 0.0
        with self._session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.name IN $entities
                RETURN count(e) AS found
                """,
                entities=entities,
            )
            found = result.single()["found"]
        return found / len(entities)

    def path_score(self, entities: List[str]) -> float:
        if len(entities) < 2:
            return 0.0

        score = 0.0
        checked = 0
        with self._session() as session:
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    result = session.run(
                        """
                        MATCH p = shortestPath(
                            (a:Entity {name:$e1})-[*..4]-(b:Entity {name:$e2})
                        )
                        RETURN length(p) AS hops
                        """,
                        e1=entities[i],
                        e2=entities[j],
                    )
                    record = result.single()
                    checked += 1
                    if not record:
                        continue
                    hops = record["hops"]
                    if hops == 1:
                        score += 1.0
                    elif hops == 2:
                        score += 0.7
                    elif hops == 3:
                        score += 0.4
        if checked == 0:
            return 0.0
        return score / checked

    def relationship_strength(self, entities: List[str]) -> float:
        with self._session() as session:
            result = session.run(
                """
                MATCH (a:Entity)-[r]->(b:Entity)
                WHERE a.name IN $entities AND b.name IN $entities
                RETURN type(r) AS rel, count(r) AS cnt
                """,
                entities=entities,
            )
            score = 0.0
            total = 0
            for record in result:
                total += record["cnt"]
                if record["rel"] in ["PART_OF", "CAUSES"]:
                    score += 1.0 * record["cnt"]
                elif record["rel"] in ["RELATED_TO"]:
                    score += 0.7 * record["cnt"]
                elif record["rel"] in ["MENTIONED_IN"]:
                    score += 0.4 * record["cnt"]
        if total == 0:
            return 0.0
        return score / total

    def graph_confidence(self, entities: List[str]) -> Dict[str, Optional[float]]:
        coverage = self.entity_coverage(entities)
        path = self.path_score(entities)
        relation = self.relationship_strength(entities)
        confidence = 0.4 * coverage + 0.35 * path + 0.25 * relation
        return {
            "coverage": round(coverage, 2),
            "path_score": round(path, 2),
            "relation_score": round(relation, 2),
            "confidence": round(confidence, 2),
        }
