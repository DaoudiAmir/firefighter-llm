# Data Preparation Process

## Overview
This document describes the complete data preparation pipeline for the firefighter LLM project, from raw data collection to training-ready examples.

## Pipeline Stages

### 1. Data Collection
- **Sources**: Training manuals, operational protocols, incident reports
- **Formats**: PDF, HTML
- **Languages**: French (primary), English (secondary)
- **Storage**: `data/raw/` organized by category

### 2. Data Processing
- **Text Extraction**:
  - PDF parsing with PyMuPDF
  - HTML parsing with BeautifulSoup4
- **Organization**:
  - By category (protocols, training, incidents)
  - With preserved metadata
- **Storage**: `data/processed/` in JSON format

### 3. Data Cleaning
- **Text Normalization**:
  - Unicode normalization
  - Quote standardization
  - Whitespace normalization
- **Content Filtering**:
  - Remove headers/footers
  - Remove page numbers
  - Preserve domain terms
- **Quality Checks**:
  - Minimum text length
  - Unique word ratio
  - Language detection
- **Storage**: `data/cleaned/` in JSONL format

### 4. Training Data Preparation
- **Example Creation**:
  - Task-based instructions
  - Context from surrounding sections
  - Language-specific formatting
- **Dataset Splits**:
  - Training: 80%
  - Validation: 10%
  - Test: 10%
- **Storage**: `data/training/` in JSONL format

## Dataset Statistics

### Training Set (56 examples)
- **Language Distribution**:
  - French: 35 (62.5%)
  - English: 21 (37.5%)
- **Category Distribution**:
  - Protocols: 28 (50.0%)
  - Training: 24 (42.9%)
  - Incidents: 4 (7.1%)
- **Task Distribution**:
  - Emergency Procedures: 25.0%
  - Hazard Identification: 23.2%
  - Training Requirements: 21.4%
  - Safety Protocols: 17.9%
  - Key Steps: 12.5%

### Example Format
```json
{
  "instruction": "As a firefighter, [task]",
  "input": "Context from relevant sections",
  "output": "Detailed response",
  "metadata": {
    "language": "fr|en",
    "category": "protocols|training|incidents",
    "source": "document source",
    "task": "task type",
    "page": "page number"
  }
}
```

## Quality Measures
1. **Content Quality**:
   - Minimum instruction length: 20 words
   - Minimum context length: 50 words
   - Maximum context length: 1000 words
   - Minimum output length: 30 words

2. **Distribution Balance**:
   - Language stratification
   - Category representation
   - Task variety

3. **Domain Relevance**:
   - Preserved technical terms
   - Maintained protocol structure
   - Context preservation

## Next Steps
1. **Data Augmentation**:
   - Add more French protocols
   - Include more incident reports
   - Expand medical procedures

2. **Quality Improvements**:
   - Enhanced language detection
   - Better context selection
   - More task variations

3. **Model Training**:
   - Fine-tune base model
   - Implement evaluation metrics
   - Create feedback loop
