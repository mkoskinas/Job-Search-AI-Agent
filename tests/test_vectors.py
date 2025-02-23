# In your Python console or notebook
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.vectorizer import VectorizeTool

# Create instance
tool = VectorizeTool()

# Load some jobs (if you haven't already)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(project_root, "data", "jobs.csv")
tool._run("store_jobs", csv_path=csv_path)

# Now inspect the vectors
if tool.vectorstore:
    # Get the raw FAISS index
    index = tool.vectorstore.index

    # Print some vector information
    print(f"Number of vectors: {index.ntotal}")

    # Get first few vectors
    vectors = index.reconstruct_n(0, 1)  # Get first vector
    print("\nFirst vector (first 10 dimensions):")
    print(vectors[0][:10])
