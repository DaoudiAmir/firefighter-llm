"""
Script to prepare training data from cleaned text.
"""
import logging
from pathlib import Path
from src.data.preprocessing.training_prep import TrainingPrep

def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    config_path = base_dir / 'configs' / 'data' / 'training_prep_config.yaml'
    input_dir = base_dir / 'data' / 'cleaned'
    output_dir = base_dir / 'data' / 'training'
    
    # Initialize training prep
    prep = TrainingPrep(str(config_path))
    
    # Process all data
    logging.info("Starting training data preparation...")
    prep.process_all(input_dir, output_dir)
    logging.info("Training data preparation completed")

if __name__ == '__main__':
    main()
