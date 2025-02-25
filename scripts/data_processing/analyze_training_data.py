"""
Script to analyze the prepared training data.
"""
import json
from pathlib import Path
from collections import defaultdict
import logging
from typing import Dict, List
import numpy as np

def load_jsonl(file_path: Path) -> List[Dict]:
    """Load examples from a JSONL file."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def analyze_examples(examples: List[Dict]) -> Dict:
    """Analyze a set of examples."""
    stats = {
        'count': len(examples),
        'languages': defaultdict(int),
        'categories': defaultdict(int),
        'tasks': defaultdict(int),
        'lengths': {
            'instruction': [],
            'input': [],
            'output': []
        }
    }
    
    for ex in examples:
        # Count languages
        stats['languages'][ex['metadata']['language']] += 1
        
        # Count categories
        stats['categories'][ex['metadata']['category']] += 1
        
        # Count tasks
        stats['tasks'][ex['metadata']['task']] += 1
        
        # Record lengths
        stats['lengths']['instruction'].append(len(ex['instruction'].split()))
        stats['lengths']['input'].append(len(ex['input'].split()))
        stats['lengths']['output'].append(len(ex['output'].split()))
    
    # Calculate length statistics
    for field in stats['lengths']:
        lengths = stats['lengths'][field]
        stats['lengths'][field] = {
            'min': min(lengths),
            'max': max(lengths),
            'mean': np.mean(lengths),
            'median': np.median(lengths)
        }
    
    return stats

def print_stats(name: str, stats: Dict):
    """Print statistics in a readable format."""
    print(f"\n{name} Set Statistics")
    print("=" * 80)
    
    print(f"\nTotal examples: {stats['count']}")
    
    print("\nLanguage Distribution:")
    for lang, count in stats['languages'].items():
        percentage = (count / stats['count']) * 100
        print(f"  {lang}: {count} ({percentage:.1f}%)")
    
    print("\nCategory Distribution:")
    for cat, count in stats['categories'].items():
        percentage = (count / stats['count']) * 100
        print(f"  {cat}: {count} ({percentage:.1f}%)")
    
    print("\nTask Distribution:")
    for task, count in stats['tasks'].items():
        percentage = (count / stats['count']) * 100
        print(f"  {task}: {count} ({percentage:.1f}%)")
    
    print("\nLength Statistics (in words):")
    for field, lengths in stats['lengths'].items():
        print(f"\n  {field.capitalize()}:")
        print(f"    Min: {lengths['min']}")
        print(f"    Max: {lengths['max']}")
        print(f"    Mean: {lengths['mean']:.1f}")
        print(f"    Median: {lengths['median']:.1f}")

def show_examples(examples: List[Dict], num_examples: int = 2):
    """Show a few examples from the dataset."""
    print("\nSample Examples")
    print("=" * 80)
    
    for i, ex in enumerate(examples[:num_examples]):
        print(f"\nExample {i+1}:")
        print(f"\nInstruction:")
        print(ex['instruction'])
        print(f"\nInput (first 200 chars):")
        print(ex['input'][:200] + "..." if len(ex['input']) > 200 else ex['input'])
        print(f"\nOutput (first 200 chars):")
        print(ex['output'][:200] + "..." if len(ex['output']) > 200 else ex['output'])
        print(f"\nMetadata:")
        print(json.dumps(ex['metadata'], indent=2))
        print("-" * 80)

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / 'data' / 'training'
    
    # Load and analyze each split
    splits = {
        'Training': load_jsonl(data_dir / 'train.jsonl'),
        'Validation': load_jsonl(data_dir / 'validation.jsonl'),
        'Test': load_jsonl(data_dir / 'test.jsonl')
    }
    
    # Print statistics for each split
    for name, examples in splits.items():
        stats = analyze_examples(examples)
        print_stats(name, stats)
    
    # Show some examples from the training set
    print("\nTraining Examples Preview")
    show_examples(splits['Training'])

if __name__ == '__main__':
    main()
