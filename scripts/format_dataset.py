# format_dataset.py
import json
from sklearn.model_selection import train_test_split
import os

def load_labeled_data(filename='fitness_response_pairs_labeled.json'):
    """Load the labeled preference data"""
    with open(filename, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} labeled examples")
    return data

def analyze_labels(data):
    """Analyze the labeled dataset"""
    print("\n" + "="*80)
    print("DATASET ANALYSIS")
    print("="*80)
    
    total = len(data)
    preferences = [item.get('preference', 'unknown') for item in data]
    equal_count = sum(1 for item in data if item.get('equal', False))
    
    print(f"\nTotal examples: {total}")
    print(f"\nPreference distribution:")
    print(f"  Response A preferred: {preferences.count('a')} ({preferences.count('a')/total*100:.1f}%)")
    print(f"  Response B preferred: {preferences.count('b')} ({preferences.count('b')/total*100:.1f}%)")
    print(f"  Equal quality: {equal_count} ({equal_count/total*100:.1f}%)")
    
    # Check for reasoning
    with_reasoning = sum(1 for item in data if item.get('reasoning', '').strip())
    print(f"\nExamples with reasoning: {with_reasoning} ({with_reasoning/total*100:.1f}%)")
    
    # Temperature analysis
    temp_a = data[0]['metadata']['temp_a']
    temp_b = data[0]['metadata']['temp_b']
    a_wins = preferences.count('a')
    b_wins = preferences.count('b')
    
    if a_wins + b_wins > 0:
        print(f"\nTemperature preference:")
        print(f"  Temp {temp_a}: {a_wins} wins ({a_wins/(a_wins+b_wins)*100:.1f}%)")
        print(f"  Temp {temp_b}: {b_wins} wins ({b_wins/(a_wins+b_wins)*100:.1f}%)")

def convert_to_standard_format(data, include_equal=False):
    """
    Convert to standard RLHF format
    
    Args:
        data: List of labeled examples
        include_equal: Whether to include examples marked as equal
    """
    formatted_data = []
    
    for item in data:
        # Skip equal preferences unless explicitly included
        if item.get('equal', False) and not include_equal:
            continue
        
        formatted_item = {
            'prompt': item['prompt'],
            'chosen': item['chosen'],
            'rejected': item['rejected'],
            'metadata': {
                'preference': item.get('preference', 'unknown'),
                'reasoning': item.get('reasoning', ''),
                'equal': item.get('equal', False),
                'labeled_at': item.get('labeled_at', ''),
                'temp_a': item['metadata']['temp_a'],
                'temp_b': item['metadata']['temp_b'],
                'model': item['metadata'].get('model', 'claude-sonnet-4-20250514'),
                'domain': 'fitness'
            }
        }
        
        formatted_data.append(formatted_item)
    
    print(f"\nFormatted {len(formatted_data)} examples")
    if not include_equal:
        equal_count = sum(1 for item in data if item.get('equal', False))
        if equal_count > 0:
            print(f"Excluded {equal_count} examples marked as 'equal'")
    
    return formatted_data

def create_splits(data, test_size=0.1, val_size=0.1, random_seed=42):
    """
    Create train/validation/test splits
    
    Args:
        data: List of formatted examples
        test_size: Proportion for test set (0.1 = 10%)
        val_size: Proportion for validation set (0.1 = 10%)
        random_seed: Random seed for reproducibility
    """
    
    # First split: separate test set
    train_val, test = train_test_split(
        data, 
        test_size=test_size, 
        random_state=random_seed
    )
    
    # Second split: separate validation from training
    val_proportion = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, 
        test_size=val_proportion, 
        random_state=random_seed
    )
    
    splits = {
        'train': train,
        'validation': val,
        'test': test
    }
    
    print("\n" + "="*80)
    print("DATASET SPLITS")
    print("="*80)
    print(f"Train: {len(train)} examples ({len(train)/len(data)*100:.1f}%)")
    print(f"Validation: {len(val)} examples ({len(val)/len(data)*100:.1f}%)")
    print(f"Test: {len(test)} examples ({len(test)/len(data)*100:.1f}%)")
    print(f"Total: {len(data)} examples")
    
    return splits

def save_splits(splits, output_dir='fitness_dataset'):
    """Save splits as JSONL files"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in splits.items():
        output_file = os.path.join(output_dir, f'{split_name}.jsonl')
        
        with open(output_file, 'w') as f:
            for item in split_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Saved {split_name}: {output_file}")
    
    # Also save as regular JSON for easier viewing
    for split_name, split_data in splits.items():
        output_file = os.path.join(output_dir, f'{split_name}.json')
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=2)
    
    print(f"\nAll files saved to: {output_dir}/")
    return output_dir

def create_dataset_card(splits, output_dir='fitness_dataset'):
    """Create a comprehensive dataset card (README.md)"""
    
    total_examples = sum(len(split) for split in splits.values())
    
    # Calculate preference statistics from training set
    train_data = splits['train']
    preferences = [item['metadata']['preference'] for item in train_data]
    
    dataset_card = f"""---
license: mit
task_categories:
- text-generation
- reinforcement-learning
language:
- en
tags:
- rlhf
- preferences
- fitness
- health
- exercise
size_categories:
- n<1K
pretty_name: Fitness Preferences Dataset
---

# Fitness Preferences Dataset for RLHF

## Dataset Description

This dataset contains human preference pairs for fitness and exercise-related questions, designed for training reward models and fine-tuning language models with RLHF (Reinforcement Learning from Human Feedback).

### Dataset Summary

- **Total Size**: {total_examples} preference pairs
- **Domain**: Fitness, exercise, nutrition, and health
- **Task**: Preference learning for fitness advice generation
- **Language**: English
- **License**: MIT

### Supported Tasks

- Training reward models for fitness advice generation
- Fine-tuning language models with RLHF for health and fitness domains
- Evaluating fitness advice quality
- Preference learning research

## Dataset Structure

### Data Instances

Each example contains a fitness question with two responses, where one response was preferred over the other by human labelers.
```json
{{
  "prompt": "How many days per week should I do strength training?",
  "chosen": "For most people, 2-3 days per week of strength training is ideal...",
  "rejected": "You should do strength training every day to see results...",
  "metadata": {{
    "preference": "a",
    "reasoning": "First response provides balanced, evidence-based advice",
    "domain": "fitness",
    ...
  }}
}}
```

### Data Fields

- **prompt** (string): The fitness/exercise question
- **chosen** (string): The preferred response
- **rejected** (string): The less preferred response
- **metadata** (dict): Additional information
  - **preference** (string): Which response was chosen ('a' or 'b')
  - **reasoning** (string): Explanation for the preference
  - **equal** (boolean): Whether responses were equal quality
  - **labeled_at** (string): Timestamp of labeling
  - **temp_a** (float): Temperature used for response A
  - **temp_b** (float): Temperature used for response B
  - **model** (string): Model used to generate responses
  - **domain** (string): "fitness"

### Data Splits

| Split | Examples | Percentage |
|-------|----------|------------|
| Train | {len(splits['train'])} | {len(splits['train'])/total_examples*100:.1f}% |
| Validation | {len(splits['validation'])} | {len(splits['validation'])/total_examples*100:.1f}% |
| Test | {len(splits['test'])} | {len(splits['test'])/total_examples*100:.1f}% |
| **Total** | **{total_examples}** | **100%** |

## Dataset Creation

### Source Data

#### Prompt Generation

Prompts were generated using both human expertise and Claude (Anthropic) with the goal of creating diverse fitness questions covering:
- Workout routines and exercise techniques
- Form and safety
- Nutrition and diet
- Recovery and rest
- Equipment usage
- Goal-specific advice (weight loss, muscle gain, endurance)
- Various fitness levels (beginner to advanced)
- Peloton usage and workout preference

#### Response Generation

Two responses were generated for each prompt using Claude Sonnet 4 with different temperature parameters:
- Response A: Temperature 0.7 (more focused/conservative)
- Response B: Temperature 1.0 (more varied/creative)

This approach creates meaningful variation between responses while maintaining quality.

### Annotation Process

#### Labeling Criteria

Preferences were labeled by a fitness-knowledgeable annotator following these criteria:

1. **CORRECT** (Most Important) - Information is accurate and not made up. Any medical or scientific advice is accurate and can be verified.  When providing advice responses should distinguish between what is objectiver vs what is unique to each person.  
2. **CLARITY**  Clear explanations.  Wherever possible, responses should clarify the difference between beginner and expert levels.  More categorization early in the response is most helpful.  Where relevant, suggestions should include a specific example - particularly when a user asks for sample workouts or exercises.  And the more concrete those examples are, the better.  Even expert athletes may have knowledge gaps and examples provide the most value.  The response should have internal consistency, so if the prompt asks for a 20 min workout, the response should total 20mins (or less).
3. **SIMPLICITY**  Fitness can have niche responses, but unless a very specific niche is asked for in the prompt, the response should assume that the person is asking a generalized question and the response should be relevant in the broadest application.  Limit the use of jargon unless it is particularly relevant.  Match the user's implied fitness level if provided.
4. **SAFETY** - Addresses potential issues and provides suggestions for safety practices associated with fitness.  Include safety warnings when appropriate, but does not recommend consulting a doctor when unnecessary or causes undue medical anxiety
5. **CONCISENESS** - Not overly verbose - Gets to the point - But not cryptic and response is well organized.



#### Quality Control

- Consistent rubric applied across all examples
- {sum(1 for item in train_data if item['metadata'].get('reasoning', '').strip())} examples ({sum(1 for item in train_data if item['metadata'].get('reasoning', '').strip())/len(train_data)*100:.1f}%) include detailed reasoning
- Regular review of labeled examples for consistency

#### Preference Distribution

In the training set:
- Response A preferred: {preferences.count('a')} ({preferences.count('a')/len(preferences)*100:.1f}%)
- Response B preferred: {preferences.count('b')} ({preferences.count('b')/len(preferences)*100:.1f}%)

### Personal and Sensitive Information

The dataset contains no personal or sensitive information. All examples are general fitness questions and AI-generated responses.


### Social Impact

**Intended Use:**
This dataset aims to improve AI systems' ability to provide helpful, accurate fitness advice, potentially making fitness guidance more accessible.

**Positive Impacts:**
- Better AI fitness assistants
- More accessible fitness information
- Safer, more accurate automated advice

**Potential Risks:**
- AI fitness advice should not replace professional medical advice
- Users with medical conditions should consult healthcare providers
- Responses may not account for individual circumstances

### Limitations

1. **Scope**: Limited to general fitness; doesn't cover medical conditions
2. **Language**: English only
3. **Cultural Context**: May reflect Western fitness culture
4. **Labeler Bias**: Single labeler's preferences and knowledge
5. **Temporal**: Fitness science evolves; may need updates
6. **Scale**: Relatively small dataset compared to large-scale RLHF datasets

### Recommendations for Use

- Combine with medical safety guidelines
- Add disclaimers about consulting healthcare providers
- Consider additional validation for specific populations
- Use as part of a larger, multi-domain preference dataset
- Monitor for potential bias or harmful advice

### Dataset Curators

Created by Victor Davidson as part of an RLHF learning project.

### Licensing Information

This dataset is released under the MIT License.

### Citation Information

If you use this dataset, please cite:
```bibtex
@dataset{{fitness_preferences_2025,
  title={{Fitness Preferences Dataset for RLHF}},
  author={{Victor Davidson}},
  year={{2025}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/victor203/fitness-preferences}}
}}
```

### Contact

- **Maintainer**: [Victor Davidson]
- **GitHub**: [https://github.com/victorDBi] 

### Acknowledgments

- Built using Claude (Anthropic) for prompt and response generation
- Inspired by Anthropic's HH-RLHF dataset methodology
- Created as an educational project to learn RLHF data collection

### Version History

- **v1.0** (2025-01): Initial release with {total_examples} examples

## Usage Example
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("victor203/fitness-preferences")

# Access different splits
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Example: Print first training example
print(train_data[0])

# Use for reward model training
# ... your training code here ...
```

## Dataset Statistics

### Response Length Distribution

Average lengths in training set:
- Chosen responses: {sum(len(item['chosen']) for item in train_data)/len(train_data):.0f} characters
- Rejected responses: {sum(len(item['rejected']) for item in train_data)/len(train_data):.0f} characters

### Prompt Diversity

The dataset covers various fitness topics including strength training, Peloton, cardio, nutrition, recovery, and form guidance across different skill levels.
"""

    # Save the dataset card
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(dataset_card)
    
    print(f"\nDataset card saved to: {readme_path}")
    return readme_path

def preview_dataset(splits, num_examples=3):
    """Preview some examples from each split"""
    print("\n" + "="*80)
    print("DATASET PREVIEW")
    print("="*80)
    
    for split_name, split_data in splits.items():
        print(f"\n{'─'*80}")
        print(f"{split_name.upper()} SET - Showing {min(num_examples, len(split_data))} examples")
        print(f"{'─'*80}")
        
        for i, item in enumerate(split_data[:num_examples], 1):
            print(f"\nExample {i}:")
            print(f"Prompt: {item['prompt']}")
            print(f"Preference: {item['metadata']['preference']}")
            print(f"Chosen: {item['chosen'][:150]}...")
            print(f"Rejected: {item['rejected'][:150]}...")
            if item['metadata'].get('reasoning'):
                print(f"Reasoning: {item['metadata']['reasoning'][:100]}...")

# Main execution
if __name__ == "__main__":
    print("DATASET FORMATTING AND SPLITTING")
    print("="*80)
    
    # Step 1: Load labeled data
    labeled_data = load_labeled_data('fitness_response_pairs_labeled.json')
    
    # Step 2: Analyze the labels
    analyze_labels(labeled_data)
    
    # Step 3: Convert to standard format
    print("\n" + "="*80)
    include_equal = input("\nInclude examples marked as 'equal'? (y/n): ").lower() == 'y'
    formatted_data = convert_to_standard_format(labeled_data, include_equal=include_equal)
    
    if len(formatted_data) == 0:
        print("\nError: No data to format!")
        exit(1)
    
    # Step 4: Create splits
    print("\nCreating train/validation/test splits...")
    splits = create_splits(formatted_data, test_size=0.1, val_size=0.1, random_seed=42)
    
    # Step 5: Save splits
    output_dir = save_splits(splits, output_dir='fitness_dataset')
    
    # Step 6: Create dataset card
    create_dataset_card(splits, output_dir=output_dir)
    
    # Step 7: Preview
    preview_dataset(splits, num_examples=2)
    
    print("\n" + "="*80)
    print("FORMATTING COMPLETE!")
    print("="*80)
    print(f"\nYour dataset is ready in the '{output_dir}/' directory")
    print("\nFiles created:")
    print(f"  - train.jsonl / train.json")
    print(f"  - validation.jsonl / validation.json")
    print(f"  - test.jsonl / test.json")
    print(f"  - README.md (dataset card)")
    print("\nNext step: Upload to Hugging Face!")