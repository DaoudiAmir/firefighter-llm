"""
Script to download firefighter training resources.
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List
from src.data.collection.resource_downloader import ResourceDownloader

def load_resource_list(resource_list_path: str) -> List[Dict]:
    """Load resource list from YAML file."""
    with open(resource_list_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['resources']

def validate_resource(resource: Dict, validation_rules: Dict) -> bool:
    """Validate a resource against rules."""
    # Check required fields
    for field in validation_rules['required_fields']:
        if field not in resource:
            logging.error(f"Missing required field: {field}")
            return False
    
    # Check category
    if resource['category'] not in validation_rules['categories']:
        logging.error(f"Invalid category: {resource['category']}")
        return False
    
    # Check language
    if resource['language'] not in validation_rules['languages']:
        logging.error(f"Invalid language: {resource['language']}")
        return False
    
    # Check type
    if resource['type'] not in validation_rules['types']:
        logging.error(f"Invalid type: {resource['type']}")
        return False
    
    return True

def main():
    """Main execution function."""
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    config_dir = base_dir / 'configs' / 'data'
    
    # Load configurations
    resource_list_path = config_dir / 'resource_list.yaml'
    with open(resource_list_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    resources = config['resources']
    validation_rules = config['validation']
    
    # Validate resources
    valid_resources = []
    for resource in resources:
        if validate_resource(resource, validation_rules):
            valid_resources.append(resource)
        else:
            logging.warning(f"Skipping invalid resource: {resource.get('title', 'Unknown')}")
    
    if not valid_resources:
        logging.error("No valid resources found")
        return
    
    # Initialize downloader
    downloader = ResourceDownloader(str(config_dir / 'data_processing_config.yaml'))
    
    # Download resources
    logging.info(f"Starting download of {len(valid_resources)} resources...")
    downloader.download_resources(valid_resources)
    
    # Save metadata
    downloader.save_resource_metadata(valid_resources)
    logging.info("Resource download completed")

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
