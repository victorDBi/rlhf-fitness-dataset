# generate_responses.py
import anthropic
import json
import time
from datetime import datetime
import os


# Initialize the Anthropic client
client = anthropic.Anthropic(
     api_key=os.environ.get("ANTHROPIC_API_KEY")
)

def load_prompts(filename='fitness_prompts_filtered.json'):
    """Load prompts from JSON file"""
    with open(filename, 'r') as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts from {filename}")
    return prompts

def generate_response_pair(prompt, temp_a=0.7, temp_b=1.0, model="claude-sonnet-4-20250514"):
    """
    Generate two different responses to the same prompt using different temperatures
    
    Args:
        prompt: The fitness question
        temp_a: Temperature for first response (lower = more focused)
        temp_b: Temperature for second response (higher = more creative)
        model: Which Claude model to use
    
    Returns:
        dict with prompt and two responses
    """
    
    # System prompt to guide Claude's fitness responses
    system_prompt = """You are a knowledgeable fitness assistant. Provide helpful, 
accurate, and safe fitness advice. When answering:

- Be specific and actionable
- Consider safety and proper form
- Adjust advice based on fitness level when mentioned
- Include relevant warnings or precautions
- Be encouraging but realistic
- If a question involves potential injury or medical concerns, recommend consulting a professional

Your responses should be informative but concise (2-4 paragraphs typically)."""

    responses = {}
    
    # Generate Response A (lower temperature - more focused/conservative)
    try:
        print(f"  Generating response A (temp={temp_a})...")
        message_a = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=temp_a,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        responses['response_a'] = message_a.content[0].text
        
        # Small delay between calls
        time.sleep(0.5)
        
    except Exception as e:
        print(f"  Error generating response A: {e}")
        return None
    
    # Generate Response B (higher temperature - more varied/creative)
    try:
        print(f"  Generating response B (temp={temp_b})...")
        message_b = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=temp_b,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        responses['response_b'] = message_b.content[0].text
        
    except Exception as e:
        print(f"  Error generating response B: {e}")
        return None
    
    return responses

def generate_all_response_pairs(prompts, output_file='fitness_response_pairs.json',
                                temp_a=0.7, temp_b=1.0, max_prompts=None,
                                checkpoint_frequency=10):
    """
    Generate response pairs for all prompts with checkpointing
    
    Args:
        prompts: List of prompt strings
        output_file: Where to save the results
        temp_a: Temperature for first response
        temp_b: Temperature for second response
        max_prompts: Only process this many prompts (None = all)
        checkpoint_frequency: Save progress every N prompts
    """
    
    # Limit number of prompts if specified
    if max_prompts:
        prompts = prompts[:max_prompts]
    
    response_pairs = []
    failed_prompts = []
    
    print(f"\n{'='*80}")
    print(f"GENERATING RESPONSE PAIRS")
    print(f"{'='*80}")
    print(f"Total prompts to process: {len(prompts)}")
    print(f"Temperature A: {temp_a} (more focused)")
    print(f"Temperature B: {temp_b} (more creative)")
    print(f"Checkpoint frequency: every {checkpoint_frequency} prompts\n")
    
    start_time = time.time()
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Processing: {prompt[:60]}...")
        
        # Generate the response pair
        responses = generate_response_pair(prompt, temp_a=temp_a, temp_b=temp_b)
        
        if responses is None:
            print(f"  ⚠️  Failed to generate responses")
            failed_prompts.append(prompt)
            continue
        
        # Create the data structure
        pair = {
            'prompt': prompt,
            'response_a': responses['response_a'],
            'response_b': responses['response_b'],
            'chosen': None,  # To be labeled later
            'metadata': {
                'temp_a': temp_a,
                'temp_b': temp_b,
                'model': 'claude-sonnet-4-20250514',
                'generated_at': datetime.now().isoformat(),
                'prompt_index': i - 1
            }
        }
        
        response_pairs.append(pair)
        print(f"  ✓ Generated pair successfully")
        
        # Checkpoint: save progress periodically
        if i % checkpoint_frequency == 0:
            checkpoint_file = output_file.replace('.json', f'_checkpoint_{i}.json')
            with open(checkpoint_file, 'w') as f:
                json.dump(response_pairs, f, indent=2)
            print(f"\n  💾 Checkpoint saved: {checkpoint_file}")
        
        # Rate limiting: pause between prompts to avoid hitting rate limits
        time.sleep(1)
    
    # Save final results
    with open(output_file, 'w') as f:
        json.dump(response_pairs, f, indent=2)
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    success_rate = len(response_pairs) / len(prompts) * 100
    
    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Successful pairs: {len(response_pairs)}/{len(prompts)} ({success_rate:.1f}%)")
    print(f"Failed prompts: {len(failed_prompts)}")
    print(f"Output saved to: {output_file}")
    
    if failed_prompts:
        print(f"\nFailed prompts:")
        for fp in failed_prompts:
            print(f"  - {fp}")
    
    return response_pairs, failed_prompts

def preview_pairs(pairs, num_to_show=3):
    """Preview some generated pairs"""
    print(f"\n{'='*80}")
    print(f"PREVIEW OF GENERATED PAIRS")
    print(f"{'='*80}")
    
    for i, pair in enumerate(pairs[:num_to_show], 1):
        print(f"\n{'─'*80}")
        print(f"PAIR {i}")
        print(f"{'─'*80}")
        print(f"\nPROMPT:\n{pair['prompt']}")
        print(f"\n{'─'*40}")
        print(f"RESPONSE A (temp={pair['metadata']['temp_a']}):")
        print(pair['response_a'][:300] + "..." if len(pair['response_a']) > 300 else pair['response_a'])
        print(f"\n{'─'*40}")
        print(f"RESPONSE B (temp={pair['metadata']['temp_b']}):")
        print(pair['response_b'][:300] + "..." if len(pair['response_b']) > 300 else pair['response_b'])

# Main execution
if __name__ == "__main__":
    print("STEP 2: Generate response pairs using Claude")
    print("="*80)
    
    # Load prompts
    prompts = load_prompts('fitness_prompts_filtered.json')
    
    # Ask user for configuration
    print("\nConfiguration options:")
    max_prompts_input = input(f"How many prompts to process? (Enter for all {len(prompts)}): ").strip()
    max_prompts = int(max_prompts_input) if max_prompts_input else None
    
    temp_a_input = input("Temperature for Response A? (default 0.7): ").strip()
    temp_a = float(temp_a_input) if temp_a_input else 0.7
    
    temp_b_input = input("Temperature for Response B? (default 1.0): ").strip()
    temp_b = float(temp_b_input) if temp_b_input else 1.0
    
    # Generate response pairs
    pairs, failed = generate_all_response_pairs(
        prompts,
        output_file='fitness_response_pairs.json',
        temp_a=temp_a,
        temp_b=temp_b,
        max_prompts=max_prompts,
        checkpoint_frequency=10
    )
    
    # Preview some results
    if pairs:
        preview_pairs(pairs, num_to_show=3)
    
    print("\n" + "="*80)
    print("DONE! Response pairs generated.")
    print("Next step: Label preferences using the labeling tool")