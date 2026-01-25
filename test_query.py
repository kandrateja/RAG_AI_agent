"""
Test script for querying the RAG system
"""
from src.rag_agent import RAGAgent
import logging
import sys

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

    # Get question from command line or prompt
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("\nEnter your question: ").strip()
        if not question:
            print("ERROR: No question provided")
            sys.exit(1)

    print(f"\nQuestion: {question}")
    print("="*50)

    # Query the system
    try:
        result = agent.query(question, top_k=5)
    except Exception as e:
        print(f"\n❌ Query failed: {e}")
        agent.close()
        sys.exit(1)

    # Print results
    print("\n" + "="*50)
    print("QUERY RESULTS:")
    print("="*50)
    print(f"\nAnswer:\n{result['answer']}\n")
    print("-"*50)
    print(f"Provenance: {result['provenance']}")
    print(f"Tools used: {result['tools_used']}")
    print(f"Retrieved chunks: {len(result['retrieved_chunks'])}")
    print(f"Web results: {len(result.get('web_results', []))}")
    print(f"Entities found: {len(result.get('entities', []))}")
    print(f"Relationships found: {len(result.get('relationships', []))}")
    print(f"Internal knowledge sufficient: {result.get('internal_sufficient', False)}")

    # Print citations
    if result.get('citations'):
        print("\nCitations:")
        for i, cit in enumerate(result['citations'], 1):
            page = cit.get('page_number', 'N/A')
            sim = cit.get('similarity', 0)
            print(f"  {i}. Doc: {cit['doc_id'][:20]}..., "
                  f"Page: {page}, "
                  f"Chunk: {cit['chunk_id'][:30]}..., "
                  f"Similarity: {sim:.3f}")

    # Print entities if found
    if result.get('entities'):
        print("\nRelated Entities:")
        for i, ent in enumerate(result['entities'][:10], 1):
            print(f"  {i}. {ent.get('name')} ({ent.get('type')})")

    # Print relationships if found
    if result.get('relationships'):
        print("\nRelated Relationships:")
        for i, rel in enumerate(result['relationships'][:10], 1):
            print(f"  {i}. {rel.get('from')} --[{rel.get('type')}]--> {rel.get('to')}")

    if result.get('error'):
        print(f"\n⚠️  Warning: {result['error']}")

    agent.close()
    print("\n" + "="*50)
    print("Query complete!")

if __name__ == "__main__":
    main()
