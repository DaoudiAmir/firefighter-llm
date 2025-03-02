import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("batch_test_results.log")
    ]
)
logger = logging.getLogger(__name__)

# Test prompts
TEST_PROMPTS = [
    {
        "name": "Fire Safety Protocol",
        "instruction": "Explain the standard protocol for responding to a residential fire.",
        "input": "A two-story house with potential occupants inside."
    },
    {
        "name": "Chemical Hazard",
        "instruction": "What are the safety procedures for handling a chemical spill?",
        "input": "Industrial accident involving unknown chemicals with strong odor."
    },
    {
        "name": "First Aid",
        "instruction": "Provide first aid instructions for burn victims.",
        "input": "Second-degree burns on arms and hands."
    },
    {
        "name": "Evacuation Plan",
        "instruction": "Outline an effective evacuation plan for a multi-story building.",
        "input": "Office building with 200 occupants during working hours."
    },
    {
        "name": "Equipment Usage",
        "instruction": "Explain how to properly use a fire extinguisher.",
        "input": "Class B fire in a kitchen environment."
    },
    {
        "name": "Emergency Communication",
        "instruction": "What information should be communicated when reporting a fire?",
        "input": "Witnessing a fire at a neighbor's house."
    },
    {
        "name": "Wildfire Response",
        "instruction": "What are the best practices for responding to a wildfire?",
        "input": "Rapidly spreading fire in a forested area near residential zones."
    },
    {
        "name": "French Language Test",
        "instruction": "Expliquez les procédures d'évacuation d'un immeuble en feu.",
        "input": "Un immeuble résidentiel de 10 étages avec environ 50 résidents."
    },
    {
        "name": "Technical Equipment",
        "instruction": "Describe the proper maintenance procedure for SCBA equipment.",
        "input": "Monthly inspection of Self-Contained Breathing Apparatus."
    },
    {
        "name": "Command Structure",
        "instruction": "Explain the incident command structure for a major fire response.",
        "input": "Multi-agency response to a commercial building fire."
    }
]

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

def generate_response(model, tokenizer, prompt, device, temperature=0.7, max_length=512):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    response_prefix = "### Response:"
    if response_prefix in generated_text:
        response = generated_text.split(response_prefix)[1].strip()
    else:
        response = generated_text
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Batch test the firefighter assistant model")
    parser.add_argument("--model_path", type=str, default="models/firefighter-assistant-v1/final_model",
                        help="Path to the model directory")
    parser.add_argument("--output_file", type=str, default="test_results.json",
                        help="Path to save test results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cuda or cpu)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for text generation")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum length of generated text")
    
    args = parser.parse_args()
    
    # Log configuration
    logger.info(f"Running batch test with configuration:")
    logger.info(f"  Model path: {args.model_path}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Max length: {args.max_length}")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto" if args.device == "cuda" else None,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
    )
    
    device = torch.device(args.device)
    if args.device == "cpu":
        model = model.to(device)
    
    logger.info(f"Model loaded successfully on {args.device}")
    
    # Run tests
    results = []
    
    for test_case in tqdm(TEST_PROMPTS, desc="Processing test cases"):
        logger.info(f"Processing test case: {test_case['name']}")
        
        prompt = create_prompt(test_case["instruction"], test_case["input"])
        
        try:
            response = generate_response(
                model,
                tokenizer,
                prompt,
                device,
                temperature=args.temperature,
                max_length=args.max_length
            )
            
            result = {
                "name": test_case["name"],
                "instruction": test_case["instruction"],
                "input": test_case["input"],
                "response": response,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing test case {test_case['name']}: {str(e)}")
            result = {
                "name": test_case["name"],
                "instruction": test_case["instruction"],
                "input": test_case["input"],
                "response": None,
                "status": "error",
                "error": str(e)
            }
        
        results.append(result)
    
    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Test results saved to {args.output_file}")
    
    # Print summary
    success_count = sum(1 for r in results if r["status"] == "success")
    logger.info(f"Test summary: {success_count}/{len(results)} tests completed successfully")

if __name__ == "__main__":
    main()
