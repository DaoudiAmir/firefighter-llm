"""
Script to clean processed documents.
"""
import logging
from pathlib import Path
from src.data.preprocessing.text_cleaner import DataCleaner

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='data/raw/logs/cleaner.log'
    )
    
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    config_path = base_dir / 'configs' / 'data' / 'cleaning_config.yaml'
    input_dir = base_dir / 'data' / 'processed'
    output_dir = base_dir / 'data' / 'cleaned'
    
    # Initialize cleaner
    cleaner = DataCleaner(str(config_path))
    
    # Process all files
    logging.info("Starting data cleaning process...")
    cleaner.process_directory(input_dir, output_dir)
    logging.info("Data cleaning completed")

if __name__ == '__main__':
    main()
