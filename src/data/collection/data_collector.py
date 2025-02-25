"""
Data collection module for firefighter LLM training data.
"""
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

class DataCollector:
    """Base class for collecting firefighter operational data."""
    
    def __init__(self, config_path: str):
        """Initialize data collector with configuration."""
        self.config = self._load_config(config_path)
        self.raw_data_path = Path(self.config['data_paths']['raw'])
        
    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def collect_protocols(self) -> List[Dict]:
        """Collect medical and intervention protocols."""
        raise NotImplementedError
    
    def collect_training_materials(self) -> List[Dict]:
        """Collect training materials and guidelines."""
        raise NotImplementedError
    
    def collect_incident_reports(self) -> List[Dict]:
        """Collect historical incident reports."""
        raise NotImplementedError
    
    def _validate_entry(self, entry: Dict) -> bool:
        """Validate data entry against required fields."""
        required_fields = self.config['validation']['required_fields']
        return all(field in entry for field in required_fields)
    
    def _add_metadata(self, entry: Dict) -> Dict:
        """Add metadata to data entry."""
        entry.update({
            'collection_date': datetime.now().isoformat(),
            'version': '1.0',
        })
        return entry
    
    def save_data(self, data: List[Dict], category: str):
        """Save collected data to appropriate directory."""
        output_path = self.raw_data_path / category / f"{category}_{datetime.now():%Y%m%d}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open('w', encoding='utf-8') as f:
            for entry in data:
                if self._validate_entry(entry):
                    entry = self._add_metadata(entry)
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
