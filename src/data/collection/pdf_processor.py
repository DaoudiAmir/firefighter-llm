"""
PDF processing module for extracting firefighter training data from documents.
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import fitz  # PyMuPDF

class PDFProcessor:
    """PDF processor for extracting firefighter-related data from documents."""
    
    def __init__(self, output_dir: str):
        """Initialize the PDF processor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'pdf_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """Extract text content from a PDF file."""
        try:
            doc = fitz.open(pdf_path)
            content = {
                'title': Path(pdf_path).stem,
                'text': '',
                'pages': [],
                'metadata': doc.metadata,
                'source_file': pdf_path,
                'timestamp': datetime.now().isoformat()
            }
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                content['pages'].append({
                    'page_number': page_num + 1,
                    'text': text
                })
                content['text'] += text + '\n'
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return None
    
    def save_content(self, content: Dict, category: str):
        """Save extracted content to file."""
        if not content:
            return
        
        output_file = self.output_dir / f"{category}_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        try:
            with output_file.open('w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved content to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving content: {str(e)}")
    
    def process_pdf_directory(self, directory: str, category: str):
        """Process all PDF files in a directory."""
        pdf_dir = Path(directory)
        
        for pdf_file in pdf_dir.glob('*.pdf'):
            self.logger.info(f"Processing PDF: {pdf_file}")
            content = self.extract_text_from_pdf(str(pdf_file))
            if content:
                self.save_content(content, category)
    
    def extract_structured_data(self, content: Dict) -> List[Dict]:
        """Extract structured data from PDF content."""
        structured_data = []
        
        # Simple pattern matching for protocol-like content
        current_section = None
        current_text = []
        
        for line in content['text'].split('\n'):
            # Check for section headers
            if line.isupper() and len(line.strip()) > 10:
                if current_section:
                    structured_data.append({
                        'section': current_section,
                        'content': '\n'.join(current_text)
                    })
                current_section = line.strip()
                current_text = []
            else:
                current_text.append(line)
        
        # Add the last section
        if current_section:
            structured_data.append({
                'section': current_section,
                'content': '\n'.join(current_text)
            })
        
        return structured_data
