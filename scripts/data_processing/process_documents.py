"""
Script to process downloaded documents and extract their content.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import yaml
from tqdm import tqdm

class DocumentProcessor:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_dir = Path(self.config['data_paths']['raw'])
        self.processed_dir = Path(self.config['data_paths']['processed'])
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=self.config['logging']['level'],
            format=self.config['logging']['format'],
            filename=self.raw_dir / 'logs' / 'processor.log'
        )
    
    def process_pdf(self, file_path: Path) -> Dict:
        """Extract text and metadata from PDF file."""
        try:
            doc = fitz.open(file_path)
            content = []
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                content.append({
                    'page': page_num + 1,
                    'text': page.get_text()
                })
            
            metadata = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'keywords': doc.metadata.get('keywords', ''),
                'page_count': len(doc)
            }
            
            doc.close()
            return {'content': content, 'metadata': metadata}
            
        except Exception as e:
            logging.error(f"Error processing PDF {file_path}: {str(e)}")
            return None
    
    def process_html(self, file_path: Path) -> Dict:
        """Extract text and metadata from HTML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else ''
            
            # Extract main content
            content = []
            for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text = tag.get_text(strip=True)
                if text:
                    content.append({
                        'tag': tag.name,
                        'text': text
                    })
            
            return {
                'content': content,
                'metadata': {
                    'title': title,
                    'url': soup.find('link', rel='canonical').get('href', '') if soup.find('link', rel='canonical') else ''
                }
            }
            
        except Exception as e:
            logging.error(f"Error processing HTML {file_path}: {str(e)}")
            return None
    
    def save_processed_content(self, content: Dict, output_path: Path):
        """Save processed content to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
    
    def process_all_documents(self):
        """Process all documents in the raw directory."""
        # Load metadata of downloaded resources
        try:
            with open(self.raw_dir / 'metadata' / 'resources.json', 'r') as f:
                resources_metadata = json.load(f)
        except Exception as e:
            logging.error(f"Error loading metadata: {str(e)}")
            resources_metadata = []
        
        # Process each category
        for category in ['protocols', 'training', 'incidents']:
            category_dir = self.raw_dir / category
            if not category_dir.exists():
                continue
            
            for file_path in tqdm(list(category_dir.glob('*')), desc=f'Processing {category}'):
                # Skip non-file items
                if not file_path.is_file():
                    continue
                
                # Process based on file type
                if file_path.suffix.lower() == '.pdf':
                    content = self.process_pdf(file_path)
                elif file_path.suffix.lower() in ['.html', '.htm']:
                    content = self.process_html(file_path)
                else:
                    logging.warning(f"Unsupported file type: {file_path}")
                    continue
                
                if content:
                    # Add original metadata if available
                    for resource in resources_metadata:
                        if isinstance(resource, dict) and resource.get('title', '') in str(file_path):
                            content['original_metadata'] = resource
                            break
                    
                    # Save processed content
                    output_path = self.processed_dir / category / f"{file_path.stem}.json"
                    self.save_processed_content(content, output_path)
                    logging.info(f"Successfully processed {file_path}")

def main():
    base_dir = Path(__file__).parent.parent.parent
    config_path = base_dir / 'configs' / 'data' / 'data_processing_config.yaml'
    
    processor = DocumentProcessor(str(config_path))
    processor.process_all_documents()

if __name__ == '__main__':
    main()
