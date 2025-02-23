"""
Vectorizer Tool Module

This module provides functionality for vectorizing and storing job listings
using FAISS and OpenAI embeddings.
"""

# Standard library imports
from typing import Optional, Dict, Any, List

# Third party imports
import pandas as pd

# Langchain imports
from langchain.tools import BaseTool
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from pydantic import Field


class VectorizeTool(BaseTool):
    """
    A tool for storing and managing job descriptions using vector embeddings.

    This tool provides functionality to:
    - Store job descriptions in a FAISS vector store
    - Add single job descriptions
    - Clear the vector store
    - Manage job metadata
    - Process both CSV and structured job data

    Attributes:
        name (str): Name identifier for the tool
        description (str): Description of the tool's functionality
        vectorstore (Optional[FAISS]): FAISS vector store for job embeddings
        embeddings (OpenAIEmbeddings): OpenAI embeddings model instance
    """

    name: str = "job_storage"
    description: str = """
    Useful for storing job descriptions using embeddings.
    Input should be structured job data from the job_scraper tool.
    """

    vectorstore: Optional[FAISS] = Field(default=None)
    embeddings: OpenAIEmbeddings = Field(default_factory=lambda: OpenAIEmbeddings())

    def __init__(self, **data):
        """
        Initialize the VectorizeTool.

        Args:
            **data: Additional initialization parameters
        """
        super().__init__(**data)
        self.vectorstore = None
        self.embeddings = OpenAIEmbeddings()

    def _truncate_text(self, text: str, max_length: int = 200) -> str:
        """Helper to truncate long text for debug output.

        Args:
            text (str): Text to truncate
            max_length (int, optional): Maximum length. Defaults to 200.

        Returns:
            str: Truncated text with ellipsis if needed
        """
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    def get_vectorstore(self):
        """
        Get the current vector store instance.

        Returns:
            Optional[FAISS]: Current vector store or None if not initialized
        """
        return self.vectorstore

    def clear_store(self):
        """Clear the existing vector store."""
        self.vectorstore = None
        print("\n=== Debug: Store cleared ===")

    def add_single_job(
        self, job_description: str, metadata: Optional[Dict] = None
    ) -> str:
        """
        Add a single job description to the vector store.

        Creates a fresh vector store instance with just this job description.

        Args:
            job_description (str): The job description text
            metadata (Optional[Dict]): Additional metadata for the job

        Returns:
            str: Success message or error description
        """
        try:
            # Always start fresh
            self.clear_store()
            self.vectorstore = FAISS.from_documents([], self.embeddings)

            # Create document from job description
            doc = Document(page_content=job_description, metadata=metadata or {})

            # Add to vectorstore
            self.vectorstore.add_documents([doc])
            print("\n=== Debug: Successfully added single job description ===")
            return "Successfully added job description"

        except Exception as e:
            print("\n=== Debug: Error adding job description ===\n{str(e)}")
            return f"Error adding job description: {str(e)}"

    def _run(self, action_input: Dict[str, Any], **kwargs) -> str:
        """
        Run the vectorizer tool with specified action.

        Args:
            action_input (Dict[str, Any]): Action parameters including:
                - action: The action to perform ('add_single_job', 'store_jobs', 'clear_store')
                - jobs: Job data for storage
                - job_description: Single job description text
                - metadata: Additional job metadata
            **kwargs: Additional arguments including:
                - csv_path: Path to CSV file containing job data

        Returns:
            str: Result message indicating success or failure
        """
        print("\n=== Debug: Vectorizer State BEFORE action ===")
        if self.vectorstore is not None:
            print(f"Total documents in index: {self.vectorstore.index.ntotal}")
            # Add more details about the docs
            print("Document IDs:", list(self.vectorstore.docstore._dict.keys()))
        else:
            print("Vectorstore is None")

        action = action_input.get("action")
        jobs_data = action_input.get("jobs")

        print("\n=== Debug: Vectorizer Run ===")
        print(f"Action: {action}")

        if action == "add_single_job":
            job_description = action_input.get("job_description")
            metadata = action_input.get("metadata", {})
            if not job_description:
                return "Error: No job description provided"
            return self.add_single_job(job_description, metadata)

        elif action == "store_jobs":
            print("\n=== Debug: Store state before clearing ===")
            if self.vectorstore is not None:
                print(f"Documents before clear: {self.vectorstore.index.ntotal}")
                print(
                    "Document IDs before clear:",
                    list(self.vectorstore.docstore._dict.keys()),
                )

            self.clear_store()

            print("\n=== Debug: Store state after clearing ===")
            if self.vectorstore is not None:
                print(f"Documents after clear: {self.vectorstore.index.ntotal}")

            # Initialize docs list
            docs = []

            # Handle CSV input for initial setup
            if "csv_path" in kwargs:
                try:
                    print("\n=== Debug: CSV Processing ===")
                    df = pd.read_csv(kwargs["csv_path"])
                    print(f"Found {len(df)} rows in CSV")

                    for _, row in df.iterrows():
                        text = (
                            f"Title: {row.get('Title', '')}\n"
                            f"Company: {row.get('Company', '')}\n"
                            f"Location: {row.get('Location', '')}\n"
                            f"Description: {row.get('Description', '')}\n"
                            f"Link: {row.get('Link', '')}"
                        )
                        doc = Document(
                            page_content=text,
                            metadata={
                                "link": row.get("Link", ""),
                                "job_id": row.get("Job_ID", ""),
                            },
                        )
                        docs.append(doc)

                except Exception as e:
                    return f"Error reading CSV: {str(e)}"

            # Handle structured data from scraper
            elif isinstance(jobs_data, dict) and "jobs" in jobs_data:
                structured_jobs = jobs_data["jobs"]
                print(f"\n=== Debug: Processing {len(structured_jobs)} jobs ===")

                for job in structured_jobs:
                    text = (
                        f"Title: {job.get('title', '')}\n"
                        f"Company: {job.get('company', '')}\n"
                        f"Location: {job.get('location', '')}\n"
                        f"Description: {job.get('description', '')}\n"
                        f"Link: {job.get('link', '')}"
                    )
                    doc = Document(
                        page_content=text,
                        metadata={
                            "link": job.get("link", ""),
                            "job_id": job.get("job_id", ""),
                            "description": job.get("description", ""),
                        },
                    )
                    docs.append(doc)

            try:
                self.vectorstore = FAISS.from_documents(docs, self.embeddings)
                print(f"\n=== Debug: Successfully stored {len(docs)} jobs ===")
                return f"Successfully stored {len(docs)} jobs"
            except Exception as e:
                print(f"\n=== Debug: Error ===\n{str(e)}")
                return f"Error creating embeddings: {str(e)}"

        elif action == "clear_store":
            self.clear_store()
            return "Vector store cleared successfully"

        else:
            return f"Unknown action: {action}"

    def run(self, action: str, **kwargs) -> str:
        """Public method to run the tool."""
        return self._run(action, **kwargs)

    def arun(self, action: str, **kwargs) -> str:
        """Async method to run the tool (currently just calls run)."""
        return self.run(action, **kwargs)

    def get_total_docs(self):
        """Return the total number of documents in the vector store."""
        if self.vectorstore:
            return len(self.vectorstore.docstore._dict)
        return 0
