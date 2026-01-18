

# analyze_dataset.py
from datasets import load_dataset
import pandas as pd

# Load HH-RLHF dataset
print("Loading dataset...")
dataset = load_dataset("Anthropic/hh-rlhf")

# Understand the structure
print("\nDataset structure:")
print(dataset)

# Examine a single example
print("\nFirst example:")
print(dataset['train'][0])

# Convert to pandas for easier analysis
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"\nTraining examples: {len(train_df)}")
print(f"Test examples: {len(test_df)}")

# Look at length distributions
train_df['chosen_length'] = train_df['chosen'].str.len()
train_df['rejected_length'] = train_df['rejected'].str.len()

# Basic statistics
print("\nLength statistics:")
print(train_df[['chosen_length', 'rejected_length']].describe())

import re
from collections import Counter

def extract_first_human_message(conversation):
    """Extract the first thing the human says"""
    # Split by Human: and Assistant:
    match = re.search(r'Human: (.*?)(?:Assistant:|Human:|$)', conversation, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

# Extract all initial queries
train_df['first_query'] = train_df['chosen'].apply(extract_first_human_message)

# Sample some queries to understand domains
print("Sample queries:")
for i in range(20):
    print(f"{i+1}. {train_df['first_query'].iloc[i][:100]}...")

# Define topic categories with keywords
topic_keywords = {
    'coding': ['code', 'python', 'javascript', 'programming', 'function', 'algorithm'],
    'science': ['physics', 'chemistry', 'biology', 'scientific', 'experiment'],
    'math': ['equation', 'calculate', 'mathematics', 'algebra', 'geometry'],
    'creative_writing': ['story', 'poem', 'write a', 'creative', 'fiction'],
    'advice': ['should i', 'how do i', 'advice', 'recommend', 'suggest'],
    'factual': ['who is', 'what is', 'when did', 'where is', 'history of'],
    'harmful': ['illegal', 'weapon', 'hack', 'steal', 'hurt'],
    'personal': ['my', 'i am', 'i feel', 'help me with my'],
    'fitness': ['fitness', 'yoga', 'peloton', 'weight lifting', 'exercise', 'swimming'],
    'pets': [' pet ', 'dog', ' cat ']

}

def categorize_query(query):
    """Simple keyword-based categorization"""
    query_lower = query.lower()
    categories = []
    for topic, keywords in topic_keywords.items():
        if any(kw in query_lower for kw in keywords):
            categories.append(topic)
    return categories if categories else ['other']

train_df['categories'] = train_df['first_query'].apply(categorize_query)

# Flatten and count
from itertools import chain
all_categories = list(chain.from_iterable(train_df['categories']))
category_counts = Counter(all_categories)

print("Category distribution:")
for cat, count in category_counts.most_common():
    print(f"{cat}: {count} ({count/len(train_df)*100:.1f}%)")

# Manually review a sample to identify failure modes
import random
random.seed(42)

def review_sample(df, n=50, category_filter='fitness'):
    """Helper to review random samples"""

# Filter by category if specified
    if category_filter is not None:
        if isinstance(category_filter, str):
            category_filter = [category_filter]
        
        # Filter rows where ANY of the specified categories appear
        filtered_df = df[df['categories'].apply(
            lambda cats: any(cat in category_filter for cat in cats)
        )]
        
        print(f"Filtered to {len(filtered_df)} examples with categories: {category_filter}")
        print(f"Original dataset had {len(df)} examples")
        
        if len(filtered_df) == 0:
            print("No examples found with specified categories!")
            return {}
    else:
        filtered_df = df
    
    # Sample from filtered data
    if len(filtered_df) < n:
        print(f"Only {len(filtered_df)} examples available, reviewing all of them")
        sample_indices = list(range(len(filtered_df)))
    else:
        sample_indices = random.sample(range(len(filtered_df)), n)

    failure_modes = {
        'too_short': [],
        'too_verbose': [],
        'unhelpful': [],
        'inconsistent': [],
        'unclear_preference': [],
        'factual_errors': []
    }
    
    for idx in sample_indices:
        row = filtered_df.iloc[idx]
        print(f"\n{'='*80}")
        print(f"Example {idx}")
        print(f"\nCHOSEN:\n{row['chosen'][:500]}...")
        print(f"\nREJECTED:\n{row['rejected'][:500]}...")
        print(f"\nCategories: {row['categories']}")
        
        # Manual input
        print("\nFailure modes (if any)? Enter comma-separated:")
        print("1=too_short, 2=too_verbose, 3=unhelpful, 4=inconsistent, 5=unclear, 6=factual_error, 0=none")
        
        response = input(">> ")
        # You'd log these manually during review
    
    return failure_modes

# You would run this and take notes
failure_analysis = review_sample(train_df, n=50, category_filter='fitness')
