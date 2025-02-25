"""
Web scraping module for collecting firefighter training data.
"""
import os
import json
import requests
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin

class WebScraper:
    """Web scraper for collecting firefighter-related data."""
    
    def __init__(self, output_dir: str):
        """Initialize the web scraper."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'scraping.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def scrape_page(self, url: str) -> Dict:
        """Scrape content from a single page."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content
            content = {
                'title': soup.title.string if soup.title else '',
                'text': soup.get_text(separator='\n', strip=True),
                'url': url,
                'timestamp': datetime.now().isoformat()
            }
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {str(e)}")
            return None
    
    def save_content(self, content: Dict, category: str):
        """Save scraped content to file."""
        if not content:
            return
        
        output_file = self.output_dir / f"{category}_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        try:
            with output_file.open('w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved content to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving content: {str(e)}")
    
    def process_url_list(self, urls: List[str], category: str):
        """Process a list of URLs and save their content."""
        for url in urls:
            self.logger.info(f"Processing URL: {url}")
            content = self.scrape_page(url)
            if content:
                self.save_content(content, category)
            
    def extract_links(self, url: str, base_url: str) -> List[str]:
        """Extract relevant links from a page."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            links = []
            
            for link in soup.find_all('a'):
                href = link.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    links.append(full_url)
            
            return links
            
        except Exception as e:
            self.logger.error(f"Error extracting links from {url}: {str(e)}")
            return []
