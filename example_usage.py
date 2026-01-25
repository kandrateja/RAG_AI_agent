"""
Example usage of RAG AI Agent
"""
import logging
from src.rag_agent import RAGAgent

logging.basicConfig(level=logging.INFO)


def example_ingest():
    """Example: Ingest a document"""
    agent = RAGAgent()
    
    # Ingest a document
    document_path = "path/to/your/document.pdf"
    result = agent.ingest_document(document_path)
    
    print(f"Document ingested: {result['doc_id']}")
    print(f"Chunks created: {result['chunks_created']}")
    
    agent.close()


def example_query():
    """Example: Query the system"""
    agent = RAGAgent()
    
    # Ask a question
    question = "What are the main topics discussed in the documents?"
    result = agent.query(question, top_k=5)
    
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"\nRetrieved {len(result['retrieved_chunks'])} relevant chunks")
    print(f"Found {len(result['entities'])} entities")
    print(f"Found {len(result['relationships'])} relationships")
    
    agent.close()


def example_batch_ingest():
    """Example: Ingest multiple documents"""
    agent = RAGAgent()
    
    document_paths = [
        "documents/doc1.pdf",
        "documents/doc2.pdf",
        "documents/doc3.pdf"
    ]
    
    results = agent.ingest_batch(document_paths)
    
    for result in results:
        if result['status'] == 'success':
            print(f"✓ Ingested: {result['doc_id']} ({result['chunks_created']} chunks)")
        else:
            print(f"✗ Failed: {result['doc_id']} - {result.get('error', 'Unknown error')}")
    
    agent.close()


if __name__ == "__main__":
    # Uncomment the example you want to run:
    # example_ingest()
    # example_query()
    # example_batch_ingest()
    pass
