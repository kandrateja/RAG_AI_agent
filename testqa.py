"""
Test script for Document Q&A API.
Run the backend first: uvicorn backend.api:app --reload --port 8000
Then run this: python test_document_qa.py
"""

import requests
import sys

BASE_URL = "http://localhost:8000"


def upload_document(file_path: str):
    """Upload a PDF or Word document."""
    print(f"\n--- Uploading: {file_path} ---")
    with open(file_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/api/documents/upload",
            files={"file": (file_path.split("/")[-1].split("\\")[-1], f)},
        )
    if resp.status_code == 200:
        data = resp.json()
        print(f"Uploaded successfully!")
        print(f"  Doc ID:    {data['doc_id']}")
        print(f"  Filename:  {data['filename']}")
        print(f"  Size:      {data['file_size']} bytes")
        print(f"  Chars:     {data['char_count']}")
        return data["doc_id"]
    else:
        print(f"Upload failed: {resp.status_code} - {resp.text}")
        return None


def list_documents():
    """List all uploaded documents."""
    print("\n--- Uploaded Documents ---")
    resp = requests.get(f"{BASE_URL}/api/documents")
    docs = resp.json().get("documents", [])
    if not docs:
        print("  No documents uploaded yet.")
        return []
    for doc in docs:
        print(f"  [{doc['doc_id']}] {doc['filename']} ({doc['char_count']} chars)")
    return docs


def ask_question(doc_id: str, question: str, history=None):
    """Ask a question about a document."""
    print(f"\nYou: {question}")
    payload = {
        "doc_id": doc_id,
        "message": question,
    }
    if history:
        payload["conversation_history"] = history

    resp = requests.post(f"{BASE_URL}/api/documents/chat", json=payload, timeout=120)
    if resp.status_code == 200:
        answer = resp.json()["response"]
        print(f"\nAssistant: {answer}")
        return answer
    else:
        print(f"\nError: {resp.status_code} - {resp.text}")
        return None


def delete_document(doc_id: str):
    """Delete a document."""
    resp = requests.delete(f"{BASE_URL}/api/documents/{doc_id}")
    if resp.status_code == 200:
        print(f"\nDocument {doc_id} deleted.")
    else:
        print(f"\nDelete failed: {resp.status_code} - {resp.text}")


def interactive_chat(doc_id: str):
    """Interactive chat loop with a document."""
    print("\n=== Interactive Document Chat ===")
    print("Type your questions below. Type 'quit' to exit.\n")

    history = []
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        answer = ask_question(doc_id, question, history)
        if answer:
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    # Step 1: Upload a document (pass file path as argument, or it will prompt)
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter path to PDF or Word file: ").strip()

    if not file_path:
        print("No file provided. Listing existing documents instead...")
        docs = list_documents()
        if docs:
            doc_id = docs[0]["doc_id"]
            print(f"\nUsing first document: {doc_id}")
            interactive_chat(doc_id)
        sys.exit(0)

    doc_id = upload_document(file_path)
    if not doc_id:
        print("Upload failed. Exiting.")
        sys.exit(1)

    # Step 2: Interactive Q&A
    interactive_chat(doc_id)
