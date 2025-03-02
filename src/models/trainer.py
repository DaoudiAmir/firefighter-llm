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
        
        # Log CUDA information if available
        if torch.cuda.is_available():
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        else:
            self.logger.info("CUDA not available, using CPU")
    
    def log_cuda_memory(self, checkpoint_name):
        """Log CUDA memory usage at a specific checkpoint."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**2
            reserved = torch.cuda.memory_reserved(0) / 1024**2
            self.logger.info(f"CUDA memory at {checkpoint_name}: Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
        else:
            self.logger.info(f"CUDA memory at {checkpoint_name}: Not using CUDA")
    
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
            
            # With batched=True, examples is a dict of lists, not a list of dicts
            # We need to iterate through the length of any of the lists
            batch_size = len(examples['instruction'])
            
            for i in range(batch_size):
                # Format the prompt
                prompt = self.config['data']['prompt_template'].format(
                    instruction=examples['instruction'][i],
                    input=examples.get('input', [''] * batch_size)[i]  # Use get() with default empty string list
                )
                # Add the output
                full_text = prompt + examples.get('output', [''] * batch_size)[i]  # Use get() with default empty string list
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
        """Train the model using a custom training loop to avoid accelerate compatibility issues."""
        self.logger.info("Starting training")
        
        try:
            # Create output directory
            os.makedirs(self.config['output']['output_dir'], exist_ok=True)
            
            # Validate configuration
            self._validate_config()
            
            # Set up training parameters with proper type conversion
            batch_size = int(self.config['training']['batch_size'])
            gradient_accumulation_steps = int(self.config['training']['gradient_accumulation_steps'])
            learning_rate = float(self.config['training']['learning_rate'])
            num_epochs = int(self.config['training']['num_epochs'])
            warmup_steps = int(self.config['training']['warmup_steps'])
            weight_decay = float(self.config['training']['weight_decay'])
            
            # Validate dataset
            if not self.tokenized_dataset or 'train' not in self.tokenized_dataset or 'validation' not in self.tokenized_dataset:
                raise ValueError("Dataset not properly prepared. Make sure to call prepare_dataset() first.")
            
            if len(self.tokenized_dataset['train']) == 0:
                raise ValueError("Training dataset is empty.")
            
            if len(self.tokenized_dataset['validation']) == 0:
                self.logger.warning("Validation dataset is empty. Training will proceed without validation.")
            
            # Create data loaders with error handling
            from torch.utils.data import DataLoader
            
            # Ensure batch size is not larger than dataset
            if batch_size > len(self.tokenized_dataset['train']):
                self.logger.warning(f"Batch size ({batch_size}) is larger than training dataset size ({len(self.tokenized_dataset['train'])}). Setting batch size to 1.")
                batch_size = 1
            
            train_dataloader = DataLoader(
                self.tokenized_dataset['train'],
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True if self.device == "cuda" else False,
            )
            
            eval_dataloader = DataLoader(
                self.tokenized_dataset['validation'],
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True if self.device == "cuda" else False,
            )
            
            # Check if model is on the correct device
            if next(self.model.parameters()).device.type != self.device:
                self.logger.warning(f"Model is not on {self.device}. Moving model to {self.device}.")
                self.model.to(self.device)
            
            # Set up optimizer with error handling
            from transformers import AdamW, get_linear_schedule_with_warmup
            
            # Validate learning rate and weight decay
            if learning_rate <= 0:
                self.logger.warning(f"Invalid learning rate: {learning_rate}. Setting to default 1e-5.")
                learning_rate = 1e-5
            
            if weight_decay < 0:
                self.logger.warning(f"Invalid weight decay: {weight_decay}. Setting to default 0.01.")
                weight_decay = 0.01
            
            # Create optimizer
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
            )
            
            # Set up learning rate scheduler with error handling
            total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
            
            # Ensure warmup steps is not larger than total steps
            if warmup_steps >= total_steps:
                self.logger.warning(f"Warmup steps ({warmup_steps}) is larger than total steps ({total_steps}). Setting warmup steps to 10% of total steps.")
                warmup_steps = max(1, int(total_steps * 0.1))
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            
            # Training loop
            self.logger.info(f"Starting training for {num_epochs} epochs")
            self.logger.info(f"Total optimization steps: {total_steps}")
            self.logger.info(f"Warmup steps: {warmup_steps}")
            self.logger.info(f"Learning rate: {learning_rate}")
            self.logger.info(f"Batch size: {batch_size} (effective batch size: {batch_size * gradient_accumulation_steps})")
            
            global_step = 0
            best_eval_loss = float('inf')
            
            # Convert evaluation parameters to integers
            logging_steps = int(self.config['evaluation']['logging_steps'])
            eval_steps = int(self.config['evaluation']['eval_steps'])
            
            # Ensure logging and eval steps are reasonable
            if logging_steps <= 0:
                logging_steps = max(1, len(train_dataloader) // 10)
                self.logger.warning(f"Invalid logging steps. Setting to {logging_steps}.")
            
            if eval_steps <= 0:
                eval_steps = len(train_dataloader)
                self.logger.warning(f"Invalid eval steps. Setting to {eval_steps}.")
            
            # Training loop with error handling
            for epoch in range(num_epochs):
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
                self.model.train()
                epoch_loss = 0
                
                # Training
                for step, batch in enumerate(train_dataloader):
                    try:
                        # Move batch to device
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        
                        # Forward pass
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        
                        # Check for NaN loss
                        if torch.isnan(loss).item():
                            self.logger.warning(f"NaN loss detected at step {step+1}. Skipping batch.")
                            continue
                        
                        # Normalize loss for gradient accumulation
                        loss = loss / gradient_accumulation_steps
                        
                        # Backward pass with error handling
                        try:
                            loss.backward()
                        except RuntimeError as e:
                            self.logger.error(f"Error in backward pass: {str(e)}")
                            self.logger.warning("Skipping this batch and continuing with training.")
                            optimizer.zero_grad()
                            continue
                        
                        epoch_loss += loss.item()
                        
                        # Update weights after gradient accumulation steps
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            # Clip gradients to prevent exploding gradients
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            
                            # Update weights
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            global_step += 1
                        
                        # Log progress
                        if (step + 1) % logging_steps == 0:
                            avg_loss = epoch_loss/(step+1)
                            lr = scheduler.get_last_lr()[0]
                            self.logger.info(f"Epoch {epoch+1}, Step {step+1}/{len(train_dataloader)}, Loss: {avg_loss:.4f}, LR: {lr:.8f}")
                        
                        # Evaluation
                        if global_step > 0 and global_step % eval_steps == 0:
                            eval_loss = self.evaluate(eval_dataloader)
                            self.logger.info(f"Evaluation at step {global_step}: Loss: {eval_loss:.4f}")
                            
                            # Save best model
                            if eval_loss < best_eval_loss:
                                best_eval_loss = eval_loss
                                self.save_model(os.path.join(self.config['output']['output_dir'], "best_model"))
                                self.logger.info(f"New best model saved with loss: {best_eval_loss:.4f}")
                            
                            # Set model back to training mode
                            self.model.train()
                            
                    except Exception as batch_error:
                        self.logger.error(f"Error processing batch at step {step+1}: {str(batch_error)}")
                        self.logger.warning("Skipping this batch and continuing with training.")
                        optimizer.zero_grad()
                        continue
                
                # Save checkpoint after each epoch
                checkpoint_path = os.path.join(self.config['output']['output_dir'], f"checkpoint-epoch-{epoch+1}")
                self.save_model(checkpoint_path)
                self.logger.info(f"Epoch {epoch+1} completed. Average loss: {epoch_loss/len(train_dataloader):.4f}")
                self.logger.info(f"Checkpoint saved to {checkpoint_path}")
            
            # Save final model
            final_model_path = os.path.join(self.config['output']['output_dir'], "final_model")
            self.save_model(final_model_path)
            self.logger.info(f"Training completed successfully. Final model saved to {final_model_path}")
            
        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _validate_config(self):
        """Validate the configuration."""
        required_sections = ['model', 'training', 'data', 'evaluation', 'output']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Check required training parameters
        required_training_params = ['batch_size', 'gradient_accumulation_steps', 'learning_rate', 'num_epochs']
        for param in required_training_params:
            if param not in self.config['training']:
                raise ValueError(f"Missing required training parameter: {param}")
        
        # Check required output parameters
        required_output_params = ['output_dir']
        for param in required_output_params:
            if param not in self.config['output']:
                raise ValueError(f"Missing required output parameter: {param}")
    
    def evaluate(self, eval_dataloader):
        """Evaluate the model on the validation set."""
        self.model.eval()
        eval_loss = 0
        eval_steps = 0
        
        try:
            with torch.no_grad():
                for batch in eval_dataloader:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Check for NaN loss
                    if torch.isnan(loss).item():
                        self.logger.warning("NaN loss detected during evaluation. Skipping batch.")
                        continue
                    
                    eval_loss += loss.item()
                    eval_steps += 1
            
            # Calculate average loss
            avg_loss = eval_loss / max(1, eval_steps)  # Avoid division by zero
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            # Return a high loss value to avoid saving this as best model
            avg_loss = float('inf')
        
        return avg_loss
    
    def save_model(self, output_path):
        """Save the model and tokenizer."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            
            # Save model
            self.model.save_pretrained(output_path)
            self.logger.info(f"Model saved to {output_path}")
            
            # Save tokenizer
            self.tokenizer.save_pretrained(output_path)
            self.logger.info(f"Tokenizer saved to {output_path}")
            
            # Save training configuration
            config_path = os.path.join(output_path, "training_config.json")
            with open(config_path, 'w') as f:
                import json
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Training configuration saved to {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
