"""
Training data preparation utilities.
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
from collections import defaultdict
from sklearn.model_selection import train_test_split
import logging

class TrainingPrep:
    def __init__(self, config_path: str):
        """Initialize training data preparation."""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)['training_prep']
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_cleaned_data(self, input_dir: Path) -> List[Dict]:
        """Load all cleaned data from the input directory."""
        all_data = []
        
        # Process each category directory
        for category_dir in input_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            # Process each JSONL file
            for file_path in category_dir.glob('*_cleaned.jsonl'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        section = json.loads(line)
                        section['category'] = category_dir.name
                        all_data.append(section)
        
        return all_data
    
    def create_example(self, section: Dict, surrounding_sections: List[Dict]) -> List[Dict]:
        """Create training examples from a section and its context."""
        examples = []
        
        # Create context by combining relevant parts of surrounding sections
        context_parts = []
        for s in surrounding_sections:
            # Add relevant metadata if available
            if s.get('metadata', {}).get('page'):
                context_parts.append(f"Page {s['metadata']['page']}:")
            context_parts.append(s['text'][:200])  # Use first 200 chars of each section
        
        # Add current section's metadata
        if section.get('metadata', {}).get('source'):
            context_parts.append(f"Source: {section['metadata']['source']}")
        
        context = "\n".join(context_parts)
        
        # Generate examples using different tasks and questions
        task_questions = {
            "explain the safety protocol": "What are the key safety considerations and protocols to follow?",
            "describe the emergency procedure": "What are the steps to follow in this emergency situation?",
            "outline the training requirements": "What training requirements and qualifications are needed?",
            "list the key steps": "What are the main steps or procedures to follow?",
            "identify potential hazards": "What potential hazards should I be aware of?"
        }
        
        for task, question in task_questions.items():
            # Create instruction using the template
            instruction = self.config['examples']['instruction_template'].format(
                task=task,
                context=context[:500],  # Limit context in instruction
                question=question
            )
            
            # Create example
            example = {
                'instruction': instruction,
                'input': context,
                'output': section['text'],
                'metadata': {
                    'language': section['language'],
                    'category': section['category'],
                    'source': section.get('metadata', {}).get('source', ''),
                    'task': task,
                    'page': section.get('metadata', {}).get('page', '')
                }
            }
            
            # Validate example lengths
            if (len(example['instruction']) >= self.config['quality']['min_instruction_length'] and
                len(example['input']) >= self.config['quality']['min_context_length'] and
                len(example['input']) <= self.config['quality']['max_context_length'] and
                len(example['output']) >= self.config['quality']['min_output_length']):
                
                # Add language-specific task prefix
                if example['metadata']['language'] == 'fr':
                    example['instruction'] = "En tant que pompier, " + example['instruction']
                
                examples.append(example)
        
        return examples
    
    def prepare_training_data(self, sections: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Prepare training data from cleaned sections."""
        # Group sections by language and category
        grouped_sections = defaultdict(list)
        for section in sections:
            key = (section['language'], section['category'])
            grouped_sections[key].append(section)
        
        # Create examples for each group
        all_examples = []
        for key, group_sections in grouped_sections.items():
            logging.info(f"Processing group: language={key[0]}, category={key[1]}")
            
            # Create examples with surrounding context
            for i, section in enumerate(group_sections):
                # Get surrounding sections as context
                start_idx = max(0, i - self.config['examples']['context_window'])
                end_idx = min(len(group_sections), i + self.config['examples']['context_window'] + 1)
                surrounding = group_sections[start_idx:i] + group_sections[i+1:end_idx]
                
                # Create examples
                examples = self.create_example(section, surrounding)
                all_examples.extend(examples)
        
        # Balance examples if configured
        if self.config['sampling']['balance_languages']:
            balanced_examples = self.balance_examples(all_examples)
        else:
            balanced_examples = all_examples
        
        if not balanced_examples:
            logging.warning("No valid examples generated")
            return [], [], []
        
        # Create stratification labels if needed
        if self.config['sampling']['stratify_by']:
            stratify_labels = []
            for ex in balanced_examples:
                label = []
                for field in self.config['sampling']['stratify_by']:
                    label.append(ex['metadata'].get(field, 'unknown'))
                stratify_labels.append('_'.join(label))
        else:
            stratify_labels = None
        
        # Split into train/validation/test
        try:
            train_examples, temp_examples = train_test_split(
                balanced_examples,
                train_size=self.config['splits']['train'],
                stratify=stratify_labels,
                random_state=42
            )
            
            # Split remaining data into validation and test
            val_size = self.config['splits']['validation'] / (self.config['splits']['validation'] + self.config['splits']['test'])
            
            # Create new stratification labels for the temp split if needed
            if stratify_labels:
                temp_stratify = []
                for ex in temp_examples:
                    label = []
                    for field in self.config['sampling']['stratify_by']:
                        label.append(ex['metadata'].get(field, 'unknown'))
                    temp_stratify.append('_'.join(label))
            else:
                temp_stratify = None
            
            val_examples, test_examples = train_test_split(
                temp_examples,
                train_size=val_size,
                stratify=temp_stratify,
                random_state=42
            )
            
        except ValueError as e:
            logging.warning(f"Could not stratify splits: {str(e)}. Falling back to random split.")
            # Fallback to simple random split
            train_size = int(len(balanced_examples) * self.config['splits']['train'])
            val_size = int(len(balanced_examples) * self.config['splits']['validation'])
            
            train_examples = balanced_examples[:train_size]
            val_examples = balanced_examples[train_size:train_size + val_size]
            test_examples = balanced_examples[train_size + val_size:]
        
        return train_examples, val_examples, test_examples
    
    def balance_examples(self, examples: List[Dict]) -> List[Dict]:
        """Balance examples across languages and categories."""
        if not examples:
            return []
            
        # Group by language
        by_language = defaultdict(list)
        for example in examples:
            by_language[example['metadata']['language']].append(example)
        
        if not by_language:
            return examples
            
        # Find minimum number of examples per language
        min_examples = min(len(group) for group in by_language.values())
        min_examples = max(min_examples, self.config['sampling']['min_samples_per_class'])
        
        # Sample equally from each language
        balanced = []
        for language, group in by_language.items():
            if len(group) >= min_examples:
                balanced.extend(random.sample(group, min_examples))
            else:
                balanced.extend(group)  # Keep all examples if less than min_examples
        
        return balanced
    
    def save_examples(self, examples: List[Dict], output_path: Path):
        """Save examples to a JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
    
    def process_all(self, input_dir: Path, output_dir: Path):
        """Process all cleaned data and create training examples."""
        # Load all cleaned data
        sections = self.load_cleaned_data(input_dir)
        logging.info(f"Loaded {len(sections)} cleaned sections")
        
        # Prepare training data
        train_examples, val_examples, test_examples = self.prepare_training_data(sections)
        
        # Save examples
        self.save_examples(train_examples, output_dir / 'train.jsonl')
        self.save_examples(val_examples, output_dir / 'validation.jsonl')
        self.save_examples(test_examples, output_dir / 'test.jsonl')
        
        # Log statistics
        logging.info(f"Created {len(train_examples)} training examples")
        logging.info(f"Created {len(val_examples)} validation examples")
        logging.info(f"Created {len(test_examples)} test examples")
