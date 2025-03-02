#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to interact with the fine-tuned firefighter assistant model.
"""

import os
import argparse
import logging
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def load_model(model_path, device="cuda"):
    """Load the fine-tuned model and tokenizer."""
    logger.info(f"Loading model from {model_path}")
    
    # Check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Log CUDA info if using GPU
    if device == "cuda":
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("Tokenizer loaded successfully")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    logger.info("Model loaded successfully")
    
    return model, tokenizer

def create_prompt(instruction, input_text=""):
    """Create a prompt using the template from training."""
    template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    return template.format(instruction=instruction, input=input_text)

def generate_response(model, tokenizer, prompt, max_length=1024, temperature=0.7):
    """Generate a response from the model."""
    # Create text generation pipeline
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True
    )
    
    # Generate response
    generated_text = text_generator(prompt)[0]["generated_text"]
    
    # Extract only the response part
    response_prefix = "### Response:"
    if response_prefix in generated_text:
        response = generated_text.split(response_prefix)[1].strip()
    else:
        response = generated_text
    
    return response

def interactive_mode(model, tokenizer):
    """Run an interactive session with the model."""
    print("\n" + "="*50)
    print("🚒 Firefighter Assistant - Interactive Mode 🚒")
    print("="*50)
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'clear' to clear the conversation history.")
    print("="*50 + "\n")
    
    while True:
        # Get user instruction
        instruction = input("\n🧑‍🚒 Instruction: ")
        if instruction.lower() in ["exit", "quit"]:
            break
        if instruction.lower() == "clear":
            print("\nConversation cleared.\n")
            continue
        
        # Get optional input context
        input_text = input("📝 Context (optional, press Enter to skip): ")
        
        # Create prompt and generate response
        prompt = create_prompt(instruction, input_text)
        print("\nGenerating response...")
        
        try:
            response = generate_response(model, tokenizer, prompt)
            print("\n🤖 Response:")
            print(response)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            print(f"\n❌ Error: {str(e)}")
        
        print("\n" + "-"*50)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Interact with the fine-tuned firefighter assistant model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="models/firefighter-assistant-v1",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/model/training_config.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run the model on"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update model path if not specified
    if args.model_path == "models/firefighter-assistant-v1":
        args.model_path = config["output"]["output_dir"]
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path, args.device)
    
    # Run interactive mode
    interactive_mode(model, tokenizer)

if __name__ == "__main__":
    main()
