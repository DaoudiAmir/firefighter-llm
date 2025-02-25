# Data Collection Process

## Overview
This document describes the process of collecting firefighter training data for our LLM project.

## Resource Configuration
Resources are defined in `configs/data/resource_list.yaml` with the following structure:
```yaml
resources:
  - url: "URL_TO_RESOURCE"
    category: "protocols|training|incidents"
    title: "Resource Title"
    description: "Resource Description"
    language: "en|fr"
    type: "operational_guide|training_manual|sop|medical_protocol|resource_collection"
```

## Directory Structure
```
data/
├── raw/
│   ├── protocols/
│   ├── training/
│   ├── incidents/
│   └── metadata/
└── processed/
    ├── train/
    ├── validation/
    └── test/
```

## Download Process
1. Configure download settings in `configs/data/data_processing_config.yaml`
2. Run the download script:
```bash
python scripts/data_collection/download_resources.py
```

## Resource Types
- Operational Guides (French: GTO, GODR)
- Training Manuals
- Standard Operating Procedures (SOPs)
- Medical Protocols
- Emergency Response Guidelines

## Validation Rules
- Required fields: url, category, title, description, language, type
- Categories: protocols, training, incidents
- Languages: en, fr
- Types: operational_guide, training_manual, sop, medical_protocol, resource_collection

## Error Handling
- Failed downloads are logged
- Metadata is saved for successful downloads
- Retries configured in data_processing_config.yaml
