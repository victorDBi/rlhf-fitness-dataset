# generate_prompts.py
import anthropic
import json
import time
import os

# Initialize the Anthropic client
client = anthropic.Anthropic(
     api_key=os.environ.get("ANTHROPIC_API_KEY")
)

def generate_fitness_prompts(num_prompts=100, batch_size=20):
    """
    Generate diverse fitness-related prompts using Claude
    
    Args:
        num_prompts: Total number of prompts to generate
        batch_size: How many prompts to generate per API call
    """
    all_prompts = []
    num_batches = (num_prompts + batch_size - 1) // batch_size  # Ceiling division
    
    for batch_num in range(num_batches):
        print(f"\nGenerating batch {batch_num + 1}/{num_batches}...")
        
        # Create the prompt for Claude
        system_prompt = """You are an expert at creating diverse fitness and exercise questions 
that people might ask a fitness assistant. Generate questions that cover a wide range of topics 
including:

- Workout routines and exercises
- Form and technique
- Nutrition and diet
- Recovery and rest
- Equipment usage
- Injury prevention
- Goal setting (weight loss, muscle gain, endurance)
- Different fitness levels (beginner, intermediate, advanced)
- Various exercise types (cardio, strength training, yoga, swimming, etc.)

Make the questions natural and varied in complexity."""

        user_prompt = f"""Generate exactly {batch_size} diverse fitness-related questions.
        
Requirements:
- Each question should be realistic (something a real person would ask)
- Vary the difficulty from beginner to advanced
- Include different types of questions: how-to, advice-seeking, explanations, recommendations
- Make them specific enough to be useful but not so narrow they're trivial
- Avoid repetition with different wording

Format: Return ONLY a JSON array of strings, one question per string. No other text.

Example format:
["Question 1 here", "Question 2 here", ...]
"""

        try:
            # Call Claude API
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0.8,  # Higher temperature for more diversity
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract the response
            response_text = message.content[0].text
            
            # Parse the JSON response
            # Claude might wrap it in ```json ``` tags, so clean that up
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith("```"):
                response_text = response_text[3:]  # Remove ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove trailing ```
            response_text = response_text.strip()
            
            # Parse JSON
            batch_prompts = json.loads(response_text)
            
            # Validate we got a list
            if not isinstance(batch_prompts, list):
                print(f"Warning: Expected list but got {type(batch_prompts)}")
                continue
                
            print(f"Generated {len(batch_prompts)} prompts in this batch")
            
            # Add to our collection
            all_prompts.extend(batch_prompts)
            
            # Show a sample
            if batch_prompts:
                print(f"Sample: {batch_prompts[0]}")
            
            # Rate limiting: pause between requests
            time.sleep(1)
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response was: {response_text[:200]}...")
            continue
        except Exception as e:
            print(f"Error in API call: {e}")
            continue
    
    # Remove duplicates while preserving order
    seen = set()
    unique_prompts = []
    for prompt in all_prompts:
        if prompt not in seen:
            seen.add(prompt)
            unique_prompts.append(prompt)
    
    print(f"\n{'='*80}")
    print(f"Total prompts generated: {len(all_prompts)}")
    print(f"Unique prompts: {len(unique_prompts)}")
    print(f"Duplicates removed: {len(all_prompts) - len(unique_prompts)}")
    
    return unique_prompts

def save_prompts(prompts, filename='fitness_prompts.json'):
    """Save prompts to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(prompts, f, indent=2)
    print(f"\nPrompts saved to {filename}")

def review_and_filter_prompts(prompts):
    """
    Manually review prompts and filter out bad ones
    """
    print("\n" + "="*80)
    print("REVIEWING GENERATED PROMPTS")
    print("="*80)
    print("\nInstructions: For each prompt, type:")
    print("  'y' = keep it")
    print("  'n' = remove it")
    print("  'q' = quit reviewing and save what you have")
    print("  '' (just press Enter) = keep it\n")
    
    kept_prompts = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt}")
        response = input("Keep? (y/n/q): ").lower().strip()
        
        if response == 'q':
            print(f"\nStopping review. Kept {len(kept_prompts)} prompts so far.")
            break
        elif response == 'n':
            print("  → Removed")
            continue
        else:  # 'y' or empty (default to keep)
            kept_prompts.append(prompt)
            print("  → Kept")
    
    print(f"\n{'='*80}")
    print(f"Final count: {len(kept_prompts)} prompts kept out of {len(prompts)}")
    return kept_prompts

# Main execution
if __name__ == "__main__":
    print("STEP 1: Generate prompts using Claude")
    print("="*80)
    
    # Generate prompts
    prompts = generate_fitness_prompts(num_prompts=100, batch_size=20)
    
    # Save raw generated prompts
    save_prompts(prompts, 'fitness_prompts_raw.json')
    
    # Display all prompts for quick review
    print("\n" + "="*80)
    print("GENERATED PROMPTS:")
    print("="*80)
    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. {prompt}")
    
    # Optional: Manual review and filtering
    print("\n" + "="*80)
    do_review = input("\nDo you want to manually review and filter prompts? (y/n): ")
    
    if do_review.lower() == 'y':
        filtered_prompts = review_and_filter_prompts(prompts)
        save_prompts(filtered_prompts, 'fitness_prompts_filtered.json')
        final_prompts = filtered_prompts
    else:
        final_prompts = prompts
    
    print("\n" + "="*80)
    print(f"DONE! You have {len(final_prompts)} fitness prompts ready.")
    print("Next step: Generate response pairs using these prompts")