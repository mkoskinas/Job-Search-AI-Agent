# test_vectorizer.py
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.vectorizer import VectorizeTool


def test_vectorizer():
    print("\n=== Testing VectorizeTool ===")

    # Initialize the tool
    vectorizer = VectorizeTool()

    # Test 1: Store jobs from a list
    print("\n1. Testing direct job storage:")
    test_jobs = [
        {
            "job_id": "1",
            "title": "Software Engineer",
            "company": "TechCorp",
            "location": "Berlin",
            "description": "Python developer role...",
            "link": "http://example.com/job1",
        },
        {
            "job_id": "2",
            "title": "Data Scientist",
            "company": "DataCo",
            "location": "London",
            "description": "ML engineer position...",
            "link": "http://example.com/job2",
        },
    ]

    response = vectorizer._run("store_jobs", jobs=test_jobs)
    print(response)

    # Test 2: Store jobs from CSV
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_path, "data", "jobs.csv")
    print(f"\n2. Testing CSV job storage from {csv_path}:")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Found {len(df)} jobs in CSV file:")
        print(df[["Title", "Company", "Location"]].to_string())

        response = vectorizer._run("store_jobs", csv_path=csv_path)
        print(response)
    else:
        print(f"Skipping CSV test - file not found: {csv_path}")

    # Test 3: Test vector search
    print("\n3. Testing vector search:")
    if vectorizer.vectorstore:
        total_docs = len(vectorizer.vectorstore.docstore._dict)
        print(f"\n=== Debug: Vector Store Stats ===")
        print(f"Total documents in store: {total_docs}")

        results = vectorizer.vectorstore.similarity_search(
            "Product Manager in Berlin", k=total_docs
        )
        print(f"Found {len(results)} matching jobs")
        print("\nFirst result:", results[0].page_content if results else "No results")

    # Test 4: Clear store
    print("\n4. Testing clear store:")
    response = vectorizer._run("clear_store")
    print(response)

    # Test 5: Invalid action
    print("\n5. Testing invalid action:")
    response = vectorizer._run("invalid_action")
    print(response)


if __name__ == "__main__":
    test_vectorizer()
