from langchain.tools import BaseTool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from pydantic import Field
from typing import Optional, Dict, Any
import pandas as pd

class VectorizeTool(BaseTool):
    name: str = "job_storage"
    description: str = """
    Useful for storing job descriptions using embeddings.
    Input should be structured job data from the job_scraper tool.
    """
    
    vectorstore: Optional[FAISS] = Field(default=None)
    embeddings: OpenAIEmbeddings = Field(default_factory=lambda: OpenAIEmbeddings())
    
    def __init__(self, **data):
        super().__init__(**data)
        self.vectorstore = None
        self.embeddings = OpenAIEmbeddings()

    def _truncate_text(self, text: str, max_length: int = 200) -> str:
        """Helper to truncate long text for debug output."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    def get_vectorstore(self):
        """Return the current vector store."""
        return self.vectorstore

    def clear_store(self):
        """Clear the existing vector store."""
        self.vectorstore = None
        print("\n=== Debug: Store cleared ===")

    def add_single_job(self, job_description: str, metadata: Optional[Dict] = None) -> str:
        """
        Add a single job description to the vectorstore.
        Creates a fresh vectorstore with just this job description.
        """
        try:
            # Always start fresh
            self.clear_store()
            self.vectorstore = FAISS.from_documents([], self.embeddings)

            # Create document from job description
            doc = Document(
                page_content=job_description,
                metadata=metadata or {}
            )

            # Add to vectorstore
            self.vectorstore.add_documents([doc])
            print(f"\n=== Debug: Successfully added single job description ===")
            return "Successfully added job description"

        except Exception as e:
            print(f"\n=== Debug: Error adding job description ===\n{str(e)}")
            return f"Error adding job description: {str(e)}"

    def _run(self, action_input: Dict[str, Any], **kwargs) -> str:
        """Run the vectorizer tool."""
        print(f"\n=== Debug: Vectorizer State BEFORE action ===")
        if self.vectorstore is not None:
            print(f"Total documents in index: {self.vectorstore.index.ntotal}")
            # Add more details about the docs
            print("Document IDs:", list(self.vectorstore.docstore._dict.keys()))
        else:
            print("Vectorstore is None")

        action = action_input.get('action')
        jobs_data = action_input.get('jobs')

        print(f"\n=== Debug: Vectorizer Run ===")
        print(f"Action: {action}")
        
        if action == "add_single_job":
            job_description = action_input.get('job_description')
            metadata = action_input.get('metadata', {})
            if not job_description:
                return "Error: No job description provided"
            return self.add_single_job(job_description, metadata)
                
        elif action == "store_jobs":
            print(f"\n=== Debug: Store state before clearing ===")
            if self.vectorstore is not None:
                print(f"Documents before clear: {self.vectorstore.index.ntotal}")
                print("Document IDs before clear:", list(self.vectorstore.docstore._dict.keys()))
            
            self.clear_store()
            
            print(f"\n=== Debug: Store state after clearing ===")
            if self.vectorstore is not None:
                print(f"Documents after clear: {self.vectorstore.index.ntotal}")
                
            # Initialize docs list
            docs = []

            # Handle CSV input for initial setup
            if "csv_path" in kwargs:
                try:
                    print(f"\n=== Debug: CSV Processing ===")
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
                                "job_id": row.get("Job_ID", "")
                            }
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
                            "description": job.get("description", "")
                        }
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