# Data Cleaning Process

## Overview
This document describes the data cleaning pipeline for the firefighter training LLM project. The pipeline processes raw text from various sources (PDFs, HTML) and prepares it for model training.

## Pipeline Components

### 1. Configuration (`configs/data/cleaning_config.yaml`)
```yaml
cleaning:
  normalize:
    remove_multiple_spaces: true
    remove_urls: true
    remove_email: true
    normalize_quotes: true
    normalize_whitespace: true
    fix_unicode: true
  
  filter:
    min_text_length: 50
    max_text_length: 2000
    remove_headers_footers: true
    remove_page_numbers: true
  
  language:
    en:
      spell_check: true
      abbreviation_expansion: true
    fr:
      spell_check: true
      abbreviation_expansion: true
```

### 2. Text Cleaning (`src/data/preprocessing/text_cleaner.py`)
- **Language Detection**: Automatic detection of English and French text
- **Text Normalization**:
  - Unicode normalization
  - URL and email removal
  - Quote standardization
  - Whitespace normalization
- **Domain Term Preservation**:
  - Preserves important firefighting terms (SCBA, PPE, HAZMAT, etc.)
  - Maintains technical abbreviations
- **Quality Checks**:
  - Minimum text length
  - Unique word ratio
  - Maximum repeated phrases

### 3. Data Processing Scripts
- `clean_data.py`: Main script for running the cleaning pipeline
- `show_cleaned_data.py`: Analysis tool for viewing cleaned data statistics

## Results

### Training Manual Processing
- **Input**: PDF files with training procedures
- **Output**: 83 high-quality sections
- **Statistics**:
  - Total words: 16,006
  - Languages: 78 French, 5 English sections
  - Content: Training procedures, safety protocols

### GODR Protocol Document
- **Input**: French operational guide PDF
- **Output**: 145 high-quality sections
- **Statistics**:
  - Total words: 28,338
  - Languages: 142 French, 3 English sections
  - Content: Emergency response procedures

## Directory Structure
```
data/
├── raw/           # Original downloaded files
├── processed/     # Extracted text from PDFs/HTML
└── cleaned/       # Final cleaned data
    ├── protocols/
    ├── training/
    └── incidents/
```

## Usage
1. Configure cleaning parameters in `cleaning_config.yaml`
2. Run the cleaning pipeline:
```bash
python scripts/data_processing/clean_data.py
```
3. Analyze results:
```bash
python scripts/data_processing/show_cleaned_data.py data/cleaned/[category]/[file]_cleaned.jsonl
```

## Dependencies
- spaCy with English and French models
- PyMuPDF for PDF processing
- BeautifulSoup4 for HTML processing
- PyYAML for configuration
