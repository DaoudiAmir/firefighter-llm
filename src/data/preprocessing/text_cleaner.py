"""
Text cleaning and normalization utilities.
"""
import re
import unicodedata
from typing import List, Dict, Set
import spacy
from spacy.language import Language
import yaml
from pathlib import Path
import json

class TextCleaner:
    def __init__(self, config_path: str):
        """Initialize the text cleaner with configuration."""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)['cleaning']
        
        # Load language models
        self.nlp_en = spacy.load('en_core_web_sm')
        self.nlp_fr = spacy.load('fr_core_news_sm')
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        self.page_number_pattern = re.compile(r'^\s*\d+\s*$')
        
        # Load domain terms
        self.domain_terms = set(self.config['domain_terms']['preserve'])
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        # Simple heuristic: use spaCy's language detection
        doc_en = self.nlp_en(text[:1000])  # Use first 1000 chars for efficiency
        doc_fr = self.nlp_fr(text[:1000])
        
        en_score = sum(1 for token in doc_en if not token.is_punct and not token.is_space)
        fr_score = sum(1 for token in doc_fr if not token.is_punct and not token.is_space)
        
        return 'en' if en_score > fr_score else 'fr'
    
    def normalize_text(self, text: str) -> str:
        """Apply basic text normalization."""
        if self.config['normalize']['fix_unicode']:
            text = unicodedata.normalize('NFKC', text)
        
        if self.config['normalize']['remove_urls']:
            text = self.url_pattern.sub(' ', text)
        
        if self.config['normalize']['remove_email']:
            text = self.email_pattern.sub(' ', text)
        
        if self.config['normalize']['normalize_quotes']:
            text = re.sub(r'["""]', '"', text)
            text = re.sub(r'[\''']', "'", text)
        
        if self.config['normalize']['normalize_whitespace']:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        return text
    
    def clean_section(self, text: str, lang: str) -> str:
        """Clean a section of text based on language-specific rules."""
        # Get the appropriate spaCy model
        nlp = self.nlp_en if lang == 'en' else self.nlp_fr
        
        # Basic normalization
        text = self.normalize_text(text)
        
        # Process with spaCy
        doc = nlp(text)
        
        # Apply language-specific cleaning
        tokens = []
        for token in doc:
            # Skip page numbers
            if self.config['filter']['remove_page_numbers'] and self.page_number_pattern.match(token.text):
                continue
            
            # Preserve domain terms
            if token.text.upper() in self.domain_terms:
                tokens.append(token.text.upper())
                continue
            
            # Handle abbreviations if configured
            if self.config['language'][lang]['abbreviation_expansion'] and token.lemma_ != token.text:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)
        
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def check_quality(self, text: str, lang: str) -> bool:
        """Check if the cleaned text meets quality thresholds."""
        # Get the appropriate spaCy model
        nlp = self.nlp_en if lang == 'en' else self.nlp_fr
        doc = nlp(text)
        
        # Check minimum length
        word_count = len([token for token in doc if not token.is_punct and not token.is_space])
        if word_count < self.config['quality']['min_words_per_section']:
            return False
        
        # Check unique words ratio
        unique_words = set(token.text.lower() for token in doc if not token.is_punct and not token.is_space)
        unique_ratio = len(unique_words) / word_count if word_count > 0 else 0
        if unique_ratio < self.config['quality']['min_unique_words_ratio']:
            return False
        
        return True
    
    def clean_document(self, content: Dict) -> List[Dict]:
        """Clean a document and split it into high-quality sections."""
        cleaned_sections = []
        
        # Process each content item
        for item in content['content']:
            if isinstance(item, dict):
                text = item.get('text', '')
                if not text:
                    continue
                
                # Detect language if not provided
                lang = content.get('language', self.detect_language(text))
                
                # Clean the section
                cleaned_text = self.clean_section(text, lang)
                
                # Check quality
                if self.check_quality(cleaned_text, lang):
                    section = {
                        'text': cleaned_text,
                        'language': lang,
                        'category': content.get('category', ''),
                        'metadata': {
                            'page': item.get('page'),
                            'source': content.get('metadata', {}).get('title', ''),
                            'original_metadata': content.get('original_metadata', {})
                        }
                    }
                    cleaned_sections.append(section)
        
        return cleaned_sections

class DataCleaner:
    def __init__(self, config_path: str):
        """Initialize the data cleaner."""
        self.text_cleaner = TextCleaner(config_path)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)['cleaning']
    
    def process_file(self, input_path: Path, output_path: Path):
        """Process a single file."""
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        # Clean the document
        cleaned_sections = self.text_cleaner.clean_document(content)
        
        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config['output']['format'] == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for section in cleaned_sections:
                    json.dump(section, f, ensure_ascii=False)
                    f.write('\n')
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_sections, f, ensure_ascii=False, indent=2)
    
    def process_directory(self, input_dir: Path, output_dir: Path):
        """Process all files in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Process each category directory
        for category_dir in input_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_output_dir = output_dir / category_dir.name
            category_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each file in the category
            for input_file in category_dir.glob('*.json'):
                output_file = category_output_dir / f"{input_file.stem}_cleaned.jsonl"
                self.process_file(input_file, output_file)
