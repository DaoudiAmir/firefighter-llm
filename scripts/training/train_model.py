"""
Script to train the firefighter assistant model.
"""
import os
from pathlib import Path
import argparse
import logging
from src.models.trainer import FirefighterTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train the firefighter assistant model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model/training_config.yaml',
        help='Path to training configuration file'
    )
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directories if they don't exist
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        # Initialize trainer
        trainer = FirefighterTrainer(config_path)
        
        # Setup CUDA device if available
        import torch
        if torch.cuda.is_available():
            logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logging.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        else:
            logging.warning("CUDA is not available. Training will be slow on CPU!")
        
        # Prepare model and dataset
        trainer.prepare_model()
        trainer.prepare_dataset()
        
        # Train model
        trainer.train()
        
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == '__main__':
    main()
