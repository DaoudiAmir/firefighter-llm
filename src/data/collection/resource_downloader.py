"""
Resource downloader for firefighter training materials.
"""
import os
import json
import yaml
import requests
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class ResourceDownloader:
    """Downloader for firefighter training resources."""
    
    def __init__(self, config_path: str):
        """Initialize the resource downloader."""
        self.config = self._load_config(config_path)
        self.base_dir = Path(self.config['data_paths']['raw'])
        self.setup_logging()
        self.setup_directories()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = self.base_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'downloader.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """Create necessary directories."""
        for category in ['protocols', 'training', 'incidents']:
            (self.base_dir / category).mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, category: str) -> Optional[Path]:
        """Download a file from URL."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Extract filename from URL or headers
            content_disposition = response.headers.get('content-disposition')
            if content_disposition:
                filename = content_disposition.split('filename=')[1].strip('"')
            else:
                filename = unquote(Path(urlparse(url).path).name)
                if not filename:
                    filename = f"document_{datetime.now():%Y%m%d_%H%M%S}.pdf"
            
            # Ensure the filename is safe
            filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
            
            # Save file
            output_path = self.base_dir / category / filename
            total_size = int(response.headers.get('content-length', 0))
            
            with output_path.open('wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
            
            self.logger.info(f"Downloaded {url} to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error downloading {url}: {str(e)}")
            return None
    
    def download_resources(self, resources: List[Dict]):
        """Download multiple resources in parallel."""
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for resource in resources:
                futures.append(
                    executor.submit(
                        self.download_file,
                        resource['url'],
                        resource['category']
                    )
                )
            
            for future in futures:
                future.result()
    
    def save_resource_metadata(self, resources: List[Dict]):
        """Save metadata about downloaded resources."""
        metadata_file = self.base_dir / 'metadata' / 'resources.json'
        metadata_file.parent.mkdir(exist_ok=True)
        
        metadata = {
            'download_date': datetime.now().isoformat(),
            'resources': resources
        }
        
        with metadata_file.open('w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved resource metadata to {metadata_file}")

# Example resource list structure
EXAMPLE_RESOURCES = [
    {
        "url": "https://www.whitehallfire.org/wp-content/uploads/2016/05/TRAINING-MANUAL-2016-Volume-2.pdf",
        "category": "training",
        "title": "Fire Training Manual Vol 2",
        "description": "Comprehensive training manual for firefighters"
    },
    {
        "url": "https://www.lehi-ut.gov/wp-content/uploads/2023/01/Lehi-Fire-Department-SOG-and-SOPs-updated-on12.5.22.pdf",
        "category": "protocols",
        "title": "Fire Department SOPs",
        "description": "Standard Operating Procedures for fire department"
    }
]

def main():
    """Main execution function."""
    config_path = 'configs/data/data_processing_config.yaml'
    downloader = ResourceDownloader(config_path)
    
    # Load resource list from configuration or external source
    resources = EXAMPLE_RESOURCES  # Replace with actual resource list
    
    # Download resources
    downloader.download_resources(resources)
    
    # Save metadata
    downloader.save_resource_metadata(resources)

if __name__ == '__main__':
    main()
