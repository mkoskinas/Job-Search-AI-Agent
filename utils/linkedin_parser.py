import requests
from bs4 import BeautifulSoup
from typing import Optional
import re

def extract_linkedin_url(text: str) -> Optional[str]:
    """Extract LinkedIn job URL from text"""
    linkedin_pattern = r'https?://[^/]*linkedin\.com/jobs/view/[^\s)>]+'
    match = re.search(linkedin_pattern, text)
    return match.group(0) if match else None

def linkedin_to_str(message: str) -> Optional[str]:
    """Convert LinkedIn job posting URL to text"""
    try:
        # Extract URL from message
        url = extract_linkedin_url(message)
        if not url:
            print("No LinkedIn URL found in message")
            return None

        # Add headers to mimic browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        # Get the page
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find job description
        job_description = soup.find('div', {'class': 'show-more-less-html__markup'})
        
        if job_description:
            return job_description.get_text(strip=True)
        return None
        
    except Exception as e:
        print(f"Error parsing LinkedIn URL: {str(e)}")
        return None