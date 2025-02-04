from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events, EventData, EventMetrics
from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
from linkedin_jobs_scraper.filters import RelevanceFilters, TimeFilters, TypeFilters
from langchain.tools import BaseTool
from pydantic import Field
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path

class LinkedInJobScraperTool(BaseTool):
    name: str = "job_scraper"
    description: str = """
    Useful for searching jobs on LinkedIn.
    Input should be a search query in format: 'job titles in locations'
    Example: 'Product Manager in Berlin'
    """
    job_postings: List = Field(default_factory=list)
    scraper: LinkedinScraper = Field(default_factory=lambda: None) 
    data_dir: Path = Field(default_factory=lambda: None) 

    def __init__(self, **data):
        # Set default values for required fields
        data.update({
            "name": "job_scraper",
            "description": """
                Useful for searching jobs on LinkedIn.
                Input should be a search query in format: 'job titles in locations'
                Example: 'Product Manager in Berlin'
            """
        })
        
        super().__init__(**data)
        self.job_postings = []
        
        # Get the absolute path to the data directory
        self.data_dir = Path(__file__).parent.parent / 'data'
        
        
        # Define event handlers as functions
        def on_data(data: EventData):
            print(
                "[ON_DATA]",
                data.title,
                data.company,
                data.date,
                data.link,
                data.insights,
                len(data.description),
            )
            self.job_postings.append([
                data.job_id,
                data.location,
                data.title,
                data.company,
                data.date,
                data.link,
                data.description,
            ])
            
            try:
                df = pd.DataFrame(
                    self.job_postings,
                    columns=[
                        "Job_ID",
                        "Location",
                        "Title",
                        "Company",
                        "Date",
                        "Link",
                        "Description",
                    ],
                )
                # Use the absolute path when saving
                df.to_csv(self.data_dir / "jobs.csv", index=False)
            except Exception as e:
                print(f"Warning: Could not save to CSV: {str(e)}")

        def on_metrics(metrics: EventMetrics):   
            print("[ON_METRICS]", str(metrics))

        def on_error(error):
            print("[ON_ERROR]", error)

        def on_end():
            print("[ON_END]")

        # Initialise scraper with function handlers
        self.scraper = LinkedinScraper(
            chrome_executable_path=None,
            chrome_binary_location=None,
            chrome_options=None,
            headless=True,
            max_workers=1,
            slow_mo=1.5,
            page_load_timeout=60
        )

        # Add event listeners using the function handlers
        self.scraper.on(Events.DATA, on_data)
        self.scraper.on(Events.ERROR, on_error)
        self.scraper.on(Events.END, on_end)

    def _run(self, query: str) -> Dict[str, Any]:
        try:
            # Reset job postings for new search
            self.job_postings = []
            
            # Parse the query
            if " in " in query:
                title, location = query.split(" in ", 1)
                locations = [loc.strip() for loc in location.split(",")]
            else:
                title = query
                locations = ["Berlin"]

            # Create query
            queries = [
                Query(
                    query=title,
                    options=QueryOptions(
                        locations=locations,
                        apply_link=True,
                        skip_promoted_jobs=True,
                        limit=5,
                        filters=QueryFilters(
                            relevance=RelevanceFilters.RECENT,
                            time=TimeFilters.MONTH,
                            type=[TypeFilters.FULL_TIME],
                        ),
                    ),
                ),
            ]

            # Run scraper
            self.scraper.run(queries)

            if not self.job_postings:
                return {"jobs": [], "formatted_text": "No jobs found."}
            
            # Create structured data
            structured_jobs = []
            formatted_results = [f"I found {len(self.job_postings)} relevant job opportunities:"]
            
            for i, job in enumerate(self.job_postings[:10], 1):
                # Add structured data using correct list indices
                structured_jobs.append({
                    "job_id": job[0],
                    "location": job[1],
                    "title": job[2],
                    "company": job[3],
                    "date": job[4],
                    "link": job[5],
                    "description": job[6]
                })
                
                # Add formatted text
                formatted_results.append(f"\n{i}. **{job[2]}**")
                formatted_results.append(f"   • Company: {job[3]}")
                formatted_results.append(f"   • Location: {job[1]}")
                formatted_results.append(f"   • Apply here: {job[5]}")
            
            if len(self.job_postings) > 10:
                formatted_results.append("\n... and more positions available. Would you like to see more?")
            
            return {
                "jobs": structured_jobs,
                "formatted_text": "\n".join(formatted_results)
            }

        except Exception as e:
            return {"jobs": [], "formatted_text": f"Error during job search: {str(e)}"}

    def _arun(self, query: str):
        raise NotImplementedError("Async not implemented")

if __name__ == "__main__":
    # Test the tool directly
    scraper = LinkedInJobScraperTool()
    
    try:
        print("\n🔍 Starting job search...")
        response = scraper._run("Product Manager in Berlin")
        print(response)
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        print("\n🏁 Search completed")