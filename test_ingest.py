"""
Test script for document ingestion
"""
from src.rag_agent import RAGAgent
import logging
import sys

# Enable logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # Initialize agent
    print("Initializing RAG Agent...")
    try:
        agent = RAGAgent()
    except Exception as e:
        print(f"ERROR: Failed to initialize agent: {e}")
        print("\nPlease check:")
        print("1. All databases are running (Postgres, Neo4j)")
        print("2. .env file is configured correctly")
        print("3. All credentials are valid")
        sys.exit(1)

    # Get document path from command line or use default
    if len(sys.argv) > 1:
        document_path = sys.argv[1]
    else:
        document_path = input("Enter path to PDF document: ").strip()
        if not document_path:
            print("ERROR: No document path provided")
            sys.exit(1)

    print(f"\nIngesting document: {document_path}")
    print("="*50)

    # Ingest document
    result = agent.ingest_document(document_path)

    # Print results
    print("\n" + "="*50)
    print("INGESTION RESULTS:")
    print("="*50)
    print(f"Status: {result['status']}")
    print(f"Document ID: {result['doc_id']}")
    
    if result['status'] == 'success':
        print(f"Chunks created: {result.get('chunks_created', 0)}")
        print(f"Entities extracted: {result.get('entities_extracted', 0)}")
        print(f"Relationships extracted: {result.get('relationships_extracted', 0)}")
        print("\n✅ Document ingested successfully!")
    elif result['status'] == 'skipped':
        print(f"Message: {result.get('message', 'Document already exists')}")
        print("\n⚠️  Document was skipped (already exists)")
    else:
        print(f"Error: {result.get('error')}")
        print(f"Error type: {result.get('error_type')}")
        print("\n❌ Ingestion failed!")

    # Close connections
    agent.close()
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
