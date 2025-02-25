"""
Script to show a summary of processed content.
"""
import json
from pathlib import Path
import sys

def show_file_summary(file_path: str, max_items: int = 5):
    """Show a summary of the processed content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nFile: {Path(file_path).name}")
    print("=" * 80)
    
    # Show metadata
    if 'metadata' in data:
        print("\nMetadata:")
        for key, value in data['metadata'].items():
            print(f"  {key}: {value}")
    
    # Show content summary
    if 'content' in data:
        print(f"\nContent Summary (showing first {max_items} items):")
        for item in data['content'][:max_items]:
            if isinstance(item, dict):
                if 'page' in item:
                    print(f"\nPage {item['page']}:")
                    print(item['text'][:200] + "..." if len(item['text']) > 200 else item['text'])
                elif 'tag' in item:
                    print(f"\n{item['tag'].upper()}:")
                    print(item['text'][:200] + "..." if len(item['text']) > 200 else item['text'])
    
    print("\n" + "=" * 80)

def main():
    if len(sys.argv) != 2:
        print("Usage: python show_processed_content.py <json_file_path>")
        sys.exit(1)
    
    show_file_summary(sys.argv[1])

if __name__ == '__main__':
    main()
