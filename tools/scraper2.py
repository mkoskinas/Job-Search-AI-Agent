import logging
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events, EventData, EventMetrics
from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
from linkedin_jobs_scraper.filters import RelevanceFilters, TimeFilters, TypeFilters
from langchain_community.chat_models import ChatOpenAI   
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)

@dataclass
class ScraperConfig:
    """Configuration for LinkedIn scraper"""
    headless: bool = True
    max_workers: int = 1
    slow_mo: float = 0.5
    page_load_timeout: int = 40
    limit_per_search: int = 25

class LinkedInJobScraper:
    def __init__(self, config: Optional[ScraperConfig] = None, use_llm: bool = False, openai_api_key: Optional[str] = None):
        self.config = config or ScraperConfig()
        self.job_postings = []
        self.use_llm = use_llm
        self._setup_logging()
        self.scraper = self._initialize_scraper()
        
        if use_llm and openai_api_key:
            self.llm = ChatOpenAI(
                temperature=0,
                model="gpt-3.5-turbo",
                openai_api_key=openai_api_key
            )

    def _setup_logging(self):
        """Set up logging configuration"""
        self.logger = logging.getLogger(__name__)

    def _initialize_scraper(self) -> LinkedinScraper:
        """Initialize LinkedIn scraper with configuration"""
        scraper = LinkedinScraper(
            headless=self.config.headless,
            max_workers=self.config.max_workers,
            slow_mo=self.config.slow_mo,
            page_load_timeout=self.config.page_load_timeout
        )
        
        scraper.on(Events.DATA, lambda data: self._on_data(data))
        scraper.on(Events.ERROR, lambda error: self._on_error(error))
        scraper.on(Events.END, lambda: self._on_end())
        
        return scraper

    def _on_data(self, data: EventData):
        """Handle scraped job data"""
        job_data = [
            data.job_id,
            data.location,
            data.title,
            data.company,
            data.date,
            data.link,
            data.description
        ]
        
        self.job_postings.append(job_data)
        self._save_to_csv()
        
        print(f"[ON_DATA] {data.title} at {data.company}")

    def _on_error(self, error):
        """Handle errors during scraping"""
        self.logger.error(f"Scraping error: {error}")

    def _on_end(self):
        """Handle completion of scraping"""
        self.logger.info("Scraping completed")

    def _save_to_csv(self):
        """Save job postings to CSV"""
        df = pd.DataFrame(
            self.job_postings,
            columns=[
                "Job_ID",
                "Location",
                "Title",
                "Company",
                "Date",
                "Link",
                "Description"
            ]
        )
        df.to_csv("data2/jobs.csv")

    def _parse_user_input_with_llm(self, user_input: str) -> tuple[List[str], List[str]]:
        """Use LLM to intelligently parse user input into job titles and locations"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that extracts job titles and locations from user input.
            Return only a JSON object with two lists:
            - job_titles: list of job titles
            - locations: list of locations
            
            If no location is specified, return an empty list for locations.
            If no job titles are found, return an empty list for job_titles."""),
            ("user", f"Extract job titles and locations from: {user_input}")
        ])
        
        try:
            response = self.llm.invoke(prompt).content
            parsed = json.loads(response)
            return parsed['job_titles'], parsed['locations']
        except Exception as e:
            self.logger.warning(f"LLM parsing failed: {str(e)}. Falling back to standard parsing.")
            return self.parse_user_input(user_input)

    def parse_user_input(self, user_input: str) -> tuple[List[str], List[str]]:
        """Standard parsing for 'job titles in locations' format"""
        try:
            parts = user_input.split(" in ")
            if len(parts) != 2:
                raise ValueError("Input must contain 'in' to separate jobs and locations")
            
            job_titles = [title.strip() for title in parts[0].split(",")]
            locations = [loc.strip() for loc in parts[1].split(",")]
            
            return job_titles, locations
        except Exception as e:
            raise ValueError(f"Invalid input format. Error: {str(e)}")

    def search(self, user_input: str) -> List[Dict[str, Any]]:
        """Main search method that handles both LLM and standard parsing"""
        try:
            if self.use_llm and hasattr(self, 'llm'):
                job_titles, locations = self._parse_user_input_with_llm(user_input)
            else:
                job_titles, locations = self.parse_user_input(user_input)
            
            if not locations:
                locations = ["Berlin"]  # default location
            
            self.job_postings = []  # Reset for new search
            
            for title in job_titles:
                query = Query(
                    query=title,
                    options=QueryOptions(
                        locations=locations,
                        apply_link=True,
                        skip_promoted_jobs=True,
                        limit=25,
                        filters=QueryFilters(
                            relevance=RelevanceFilters.RELEVANT,
                            time=TimeFilters.MONTH,
                            type=[TypeFilters.FULL_TIME],
                        )
                    )
                )
                
                try:
                    self.scraper.run([query])
                except Exception as e:
                    self.logger.error(f"Error during search for {title}: {str(e)}")
            
            return self.job_postings
            
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            return []

class LinkedInJobScraperTool(BaseTool):
    name: str = "LinkedIn Job Scraper"
    description: str = """
    Useful for searching jobs on LinkedIn.
    Input should be a search query in natural language or in format: 'job titles in locations'
    Example: 'Product Manager in Berlin' or 'Looking for Software Engineer roles in London'
    """
    _scraper: LinkedInJobScraper = None

    def __init__(self, config: Optional[ScraperConfig] = None):
        super().__init__()
        self._scraper = LinkedInJobScraper(config=config)

    def _run(self, query: str) -> str:
        jobs = self._scraper.search(query)
        if not jobs:
            return "No jobs found."
        
        results = [f"### Found {len(jobs)} Jobs:"]
        for job in jobs:
            results.append(f"\n- **{job[2]}** at {job[3]}")  # title and company
            results.append(f"  - Location: {job[1]}")        # location
            results.append(f"  - Link: {job[5]}")           # link
        
        return "\n".join(results)

    def _arun(self, query: str):
        raise NotImplementedError("Async not implemented")

if __name__ == "__main__":
    scraper = LinkedInJobScraperTool()
    
    try:
        print("\n🔍 Starting job search...")
        response = scraper._run("Product Manager in Berlin")
        print(response)
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        print("\n🏁 Search completed")