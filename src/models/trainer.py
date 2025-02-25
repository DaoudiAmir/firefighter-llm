"""
Model trainer for the firefighter assistant.
"""
import os
import logging
from pathlib import Path
import yaml
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

class FirefighterTrainer:
    def __init__(self, config_path: str):
        """Initialize the trainer with configuration."""
        self.config = self._load_config(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_logging()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging configuration."""
        os.makedirs(self.config['output']['logging_dir'], exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    Path(self.config['output']['logging_dir']) / 'training.log'
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_model(self):
        """Prepare the model for training."""
        self.logger.info(f"Loading model {self.config['model']['name']}")
        
        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['tokenizer'],
            padding_side="right",
            model_max_length=self.config['model']['max_length']
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model with basic configuration
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name'],
            torch_dtype=torch.float32,  # Use standard precision
            use_cache=False  # Important for training
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Print model info
        self.logger.info(f"Model loaded and moved to {self.device}")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def prepare_dataset(self):
        """Prepare the dataset for training."""
        self.logger.info("Preparing dataset")
        
        def preprocess_function(examples):
            """Preprocess the examples for training."""
            # Create a list of formatted texts
            texts = []
            for example in examples:
                # Format the prompt
                prompt = self.config['data']['prompt_template'].format(
                    instruction=example['instruction'],
                    input=example.get('input', '')  # Use get() with default empty string
                )
                # Add the output
                full_text = prompt + example.get('output', '')  # Use get() with default empty string
                texts.append(full_text)
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.config['model']['max_length'],
                padding="max_length",
                return_tensors="pt"
            )
            
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        try:
            # Load datasets from local files
            train_path = Path(self.config['data']['train_file'])
            val_path = Path(self.config['data']['validation_file'])
            
            if not train_path.exists():
                raise FileNotFoundError(f"Training file not found: {train_path}")
            if not val_path.exists():
                raise FileNotFoundError(f"Validation file not found: {val_path}")
            
            # Load JSON files directly
            import json
            
            def load_jsonl(file_path):
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Skipping invalid JSON line: {e}")
                            continue
                return data
            
            self.logger.info("Loading training data...")
            train_data = load_jsonl(train_path)
            self.logger.info(f"Loaded {len(train_data)} training examples")
            
            self.logger.info("Loading validation data...")
            val_data = load_jsonl(val_path)
            self.logger.info(f"Loaded {len(val_data)} validation examples")
            
            from datasets import Dataset
            
            self.dataset = {
                'train': Dataset.from_list(train_data),
                'validation': Dataset.from_list(val_data)
            }
            
            # Preprocess datasets
            self.tokenized_dataset = {
                split: dataset.map(
                    preprocess_function,
                    batched=True,
                    remove_columns=dataset.column_names,
                    desc=f"Preprocessing {split} dataset"
                )
                for split, dataset in self.dataset.items()
            }
            
            self.logger.info("Dataset preparation completed")
            self.logger.info(f"Train dataset size: {len(self.tokenized_dataset['train'])}")
            self.logger.info(f"Validation dataset size: {len(self.tokenized_dataset['validation'])}")
            
            return self.tokenized_dataset
            
        except Exception as e:
            self.logger.error(f"Error preparing dataset: {str(e)}")
            raise
    
    def train(self):
        """Train the model."""
        self.logger.info("Starting training")
        
        # Create output directory
        os.makedirs(self.config['output']['output_dir'], exist_ok=True)
        
        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=self.config['output']['output_dir'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            num_train_epochs=self.config['training']['num_epochs'],
            warmup_steps=self.config['training']['warmup_steps'],
            weight_decay=self.config['training']['weight_decay'],
            logging_dir=self.config['output']['logging_dir'],
            logging_steps=self.config['evaluation']['logging_steps'],
            evaluation_strategy="steps",
            eval_steps=self.config['evaluation']['eval_steps'],
            save_strategy="steps",
            save_steps=self.config['evaluation']['save_steps'],
            load_best_model_at_end=True,
            optim=self.config['training']['optimizer'],
            lr_scheduler_type=self.config['training']['lr_scheduler'],
            max_grad_norm=self.config['training']['max_grad_norm'],
            report_to="none",  # Disable wandb logging
            fp16=False  # Disable mixed precision
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['validation'],
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Train
        try:
            trainer.train()
            # Save the final model
            trainer.save_model()
            self.logger.info("Training completed successfully")
        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            raise
