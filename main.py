"""
Main entry point for RAG AI Agent
"""
import logging
import sys
from pathlib import Path
from src.rag_agent import RAGAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main function"""
    try:
        # Initialize RAG Agent
        logger.info("Initializing RAG Agent...")
        agent = RAGAgent()
        
        # Example usage
        if len(sys.argv) > 1:
            command = sys.argv[1]
            
            if command == "ingest" and len(sys.argv) > 2:
                # Ingest a document
                doc_path = sys.argv[2]
                logger.info(f"Ingesting document: {doc_path}")
                result = agent.ingest_document(doc_path)
                logger.info(f"Ingestion result: {result}")
            
            elif command == "query" and len(sys.argv) > 2:
                # Query the system
                question = " ".join(sys.argv[2:])
                logger.info(f"Querying: {question}")
                result = agent.query(question)
                logger.info(f"Answer: {result['answer']}")
                logger.info(f"Retrieved {len(result['retrieved_chunks'])} chunks")
            
            else:
                print("Usage:")
                print("  python main.py ingest <document_path>")
                print("  python main.py query <question>")
        else:
            print("RAG AI Agent")
            print("\nUsage:")
            print("  python main.py ingest <document_path>  - Ingest a document")
            print("  python main.py query <question>       - Query the system")
            print("\nExample:")
            print("  python main.py ingest document.pdf")
            print("  python main.py query 'What is the main topic?'")
        
        # Close connections
        agent.close()
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
