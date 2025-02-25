"""
Script to show cleaned data statistics and samples.
"""
import json
from pathlib import Path
from collections import defaultdict
import sys

def analyze_jsonl_file(file_path: Path):
    """Analyze a cleaned JSONL file."""
    print(f"\nAnalyzing: {file_path.name}")
    print("=" * 80)
    
    stats = {
        'sections': 0,
        'total_words': 0,
        'languages': defaultdict(int),
        'categories': defaultdict(int)
    }
    
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            section = json.loads(line)
            
            # Update stats
            stats['sections'] += 1
            stats['languages'][section['language']] += 1
            stats['categories'][section['category']] += 1
            
            # Count words
            words = section['text'].split()
            stats['total_words'] += len(words)
            
            # Store sample if it's interesting (more than 50 words)
            if len(words) > 50 and len(samples) < 2:
                samples.append(section)
    
    # Print statistics
    print("\nStatistics:")
    print(f"Total sections: {stats['sections']}")
    print(f"Total words: {stats['total_words']}")
    print("\nLanguage distribution:")
    for lang, count in stats['languages'].items():
        print(f"  {lang}: {count} sections")
    print("\nCategory distribution:")
    for cat, count in stats['categories'].items():
        print(f"  {cat}: {count} sections")
    
    # Print samples
    print("\nSample sections:")
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(f"Language: {sample['language']}")
        print(f"Category: {sample['category']}")
        print("Text:")
        print(sample['text'][:300] + "..." if len(sample['text']) > 300 else sample['text'])
        print("-" * 40)

def main():
    if len(sys.argv) != 2:
        print("Usage: python show_cleaned_data.py <jsonl_file_path>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    analyze_jsonl_file(file_path)

if __name__ == '__main__':
    main()
