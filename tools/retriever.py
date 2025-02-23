"""
Job Retriever Tool Module

This module provides functionality for searching and retrieving job listings
from a FAISS vector store using semantic similarity search.
"""

# Standard library imports
import warnings
from typing import Optional, Dict, Any, Union

# Third party imports
from pydantic import Field

# Langchain imports
from langchain.tools import BaseTool
from langchain_community.vectorstores import FAISS

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class RetrieverTool(BaseTool):
    """
    A tool for retrieving and searching stored job listings.

    This tool performs semantic similarity search on previously stored job descriptions
    using a FAISS vector store. It supports both string and dictionary-based queries
    and returns formatted job listings with relevant details.

    Attributes:
        name (str): Name identifier for the tool
        description (str): Description of the tool's functionality
        vectorstore (Optional[FAISS]): FAISS vector store containing job embeddings
    """

    name: str = "job_retriever"
    description: str = """
    Useful ONLY for searching through previously stored job descriptions.
    Use this tool ONLY when the user wants to search or filter existing stored jobs.
    Do not use for general conversation, greetings, or scraping new jobs.
    
    Input should be formatted as:
    {{"query": "{query}"}}
    
    Where {query} is replaced with the user's exact search question.
    """

    vectorstore: Optional[FAISS] = Field(default=None)

    def __init__(self, vectorstore: Optional[FAISS] = None, **data):
        """
        Initialize the RetrieverTool.

        Args:
            vectorstore (Optional[FAISS]): Vector store for job embeddings
            **data: Additional initialization parameters
        """
        super().__init__(**data)
        self.vectorstore = vectorstore

    def _run(self, inputs: Union[Dict[str, Any], str]) -> str:
        """
        Search for jobs matching the query.

        Args:
            inputs (Union[Dict[str, Any], str]): Search query as string or dict with query parameters

        Returns:
            str: Formatted string containing matching job listings or error message

        Raises:
            ValueError: If the vector store is not initialized or query is invalid
        """
        if not self.vectorstore:
            return "No job listings available. Please store some jobs first."

        try:
            # Handle input wrapped in 'inputs' key
            if isinstance(inputs, dict) and "inputs" in inputs:
                inputs = inputs["inputs"]

            # Handle both string and dictionary inputs
            if isinstance(inputs, str):
                query = inputs
                k = 3
            else:
                # If input is a dict, validate and extract query
                if "query" not in inputs:
                    # Convert simple dict to query if possible
                    query = str(inputs)
                    k = 3
                else:
                    query = inputs["query"]
                    k = int(inputs.get("k", 3))

            # Perform similarity search
            docs = self.vectorstore.similarity_search(query, k=k)

            # Handle no results
            if not docs:
                return "No matching jobs found."

            # Format results
            results = []
            for i, doc in enumerate(docs, 1):
                # Extract fields from page_content
                content_lines = doc.page_content.split("\n")
                job_details = {}
                for line in content_lines:
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        job_details[key] = value

                results.append(
                    f"\n{i}. **{job_details.get('Title', 'No title')}**\n"
                    f"   • Company: {job_details.get('Company', 'Not specified')}\n"
                    f"   • Location: {job_details.get('Location', 'Not specified')}\n"
                    f"   • Description: {job_details.get('Description', 'Not specified')}\n"
                    f"   • Link: {doc.metadata.get('link', 'No link available')}\n"
                    f"   • Job ID: {doc.metadata.get('job_id', 'No ID')}\n"
                )

            return "Here are the most relevant jobs:" + "".join(results)

        except Exception as e:
            return f"Error searching jobs: {str(e)}"

    def _arun(self, query: str):
        """Async version not implemented."""
        raise NotImplementedError("Async retrieval not implemented")
