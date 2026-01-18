# label_preferences.py
import json
from datetime import datetime
import os

class PreferenceLabelingTool:
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.current_idx = 0
        self.labeled_data = []
        self.session_start = datetime.now()
        
        # Load existing progress if available
        self.progress_file = data_file.replace('.json', '_labeled.json')
        if os.path.exists(self.progress_file):
            load_progress = input(f"\nFound existing progress file. Load it? (y/n): ")
            if load_progress.lower() == 'y':
                with open(self.progress_file, 'r') as f:
                    self.labeled_data = json.load(f)
                print(f"Loaded {len(self.labeled_data)} previously labeled examples")
        
    def display_pair(self, idx):
        item = self.data[idx]
        print("\n" + "="*80)
        print(f"Example {idx + 1} / {len(self.data)}")
        print("="*80)
        print(f"\nPROMPT:\n{item['prompt']}")
        print(f"\n{'─'*80}")
        print(f"\nRESPONSE A (temp={item['metadata']['temp_a']}):")
        print(item['response_a'])
        print(f"\n{'─'*80}")
        print(f"\nRESPONSE B (temp={item['metadata']['temp_b']}):")
        print(item['response_b'])
        print(f"\n{'─'*80}")
        
    def get_preference(self):
        """Get user's preference with validation"""
        while True:
            choice = input("\nWhich is better? (a/b/equal/skip/quit): ").lower().strip()
            if choice in ['a', 'b', 'equal', 'skip', 'q', 'quit']:
                return choice
            print("Invalid input. Please enter 'a', 'b', 'equal', 'skip', or 'quit'")
    
    def get_reasoning(self):
        """Get optional reasoning for the choice"""
        print("\nWhy did you choose this? (press Enter to skip):")
        print("Consider: accuracy, clarity, completeness, safety, helpfulness")
        return input("Reasoning: ")
    
    def get_quality_ratings(self):
        """Optional: Get detailed quality ratings"""
        print("\n(Optional) Rate on 1-5 scale (or press Enter to skip):")
        
        ratings = {}
        criteria = ['accuracy', 'clarity', 'completeness', 'safety', 'helpfulness']
        
        for criterion in criteria:
            rating = input(f"  {criterion.capitalize()}: ")
            if rating.strip():
                try:
                    ratings[criterion] = int(rating)
                except ValueError:
                    pass
        
        return ratings if ratings else None
    
    def label_session(self, start_idx=0, num_examples=None, auto_save_frequency=5):
        """
        Main labeling session
        
        Args:
            start_idx: Which example to start from
            num_examples: How many to label (None = all remaining)
            auto_save_frequency: Save progress every N examples
        """
        # Skip already labeled examples
        already_labeled_prompts = {item['prompt'] for item in self.labeled_data}
        
        self.current_idx = start_idx
        end_idx = len(self.data) if num_examples is None else start_idx + num_examples
        
        labeled_this_session = 0
        skipped_count = 0
        
        print("\n" + "="*80)
        print("LABELING INSTRUCTIONS")
        print("="*80)
        print("""
When comparing responses, consider:

1. **CORRECT** (Most Important) - Information is accurate and not made up. Any medical or scientific advice is accurate and can be verified.  When providing advice responses should distinguish between what is objectiver vs what is unique to each person.  
2. **CLARITY**  Clear explanations.  Wherever possible, responses should clarify the difference between beginner and expert levels.  More categorization early in the response is most helpful.  Where relevant, suggestions should include a specific example - particularly when a user asks for sample workouts or exercises.  And the more concrete those examples are, the better.  Even expert athletes may have knowledge gaps and examples provide the most value.  The response should have internal consistency, so if the prompt asks for a 20 min workout, the response should total 20mins (or less).
3. **SIMPLICITY**  Fitness can have niche responses, but unless a very specific niche is asked for in the prompt, the response should assume that the person is asking a generalized question and the response should be relevant in the broadest application.  Limit the use of jargon unless it is particularly relevant.  Match the user's implied fitness level if provided.
4. **SAFETY** - Addresses potential issues and provides suggestions for safety practices associated with fitness.  Include safety warnings when appropriate, but does not recommend consulting a doctor when unnecessary or causes undue medical anxiety
5. **CONCISENESS** - Not overly verbose - Gets to the point - But not cryptic and response is well organized.

Choose 'a' if Response A is better
Choose 'b' if Response B is better
Choose 'equal' if both are roughly the same quality
Choose 'skip' if you're unsure or need to revisit
Choose 'quit' to stop and save progress
""")
        
        input("Press Enter to start labeling...")
        
        for idx in range(start_idx, min(end_idx, len(self.data))):
            # Skip if already labeled
            if self.data[idx]['prompt'] in already_labeled_prompts:
                print(f"\nSkipping example {idx + 1} (already labeled)")
                continue
            
            self.display_pair(idx)
            preference = self.get_preference()
            
            if preference in ['q', 'quit']:
                print("\nStopping labeling session...")
                break
            
            if preference == 'skip':
                skipped_count += 1
                print("  → Skipped")
                continue
            
            # Get reasoning
            reasoning = self.get_reasoning()
            
            # Optional: Get detailed ratings
            # ratings = self.get_quality_ratings()
            
            # Record the label
            item = self.data[idx].copy()
            
            if preference == 'a':
                item['chosen'] = item['response_a']
                item['rejected'] = item['response_b']
                item['preference'] = 'a'
            elif preference == 'b':
                item['chosen'] = item['response_b']
                item['rejected'] = item['response_a']
                item['preference'] = 'b'
            else:  # equal
                item['chosen'] = item['response_a']
                item['rejected'] = item['response_b']
                item['equal'] = True
                item['preference'] = 'equal'
            
            item['reasoning'] = reasoning
            # item['quality_ratings'] = ratings  # Uncomment if using ratings
            item['labeled_at'] = datetime.now().isoformat()
            item['labeler'] = 'primary'  # Useful if you have multiple labelers
            
            self.labeled_data.append(item)
            labeled_this_session += 1
            
            print(f"  ✓ Labeled as: {preference}")
            
            # Auto-save progress
            if labeled_this_session % auto_save_frequency == 0:
                self.save_progress()
                print(f"\n  💾 Auto-saved! ({len(self.labeled_data)} total labeled)")
        
        # Final save
        self.save_progress()
        
        # Summary statistics
        session_duration = (datetime.now() - self.session_start).total_seconds() / 60
        
        print(f"\n{'='*80}")
        print(f"LABELING SESSION COMPLETE")
        print(f"{'='*80}")
        print(f"Labeled this session: {labeled_this_session}")
        print(f"Skipped: {skipped_count}")
        print(f"Total labeled so far: {len(self.labeled_data)}")
        print(f"Remaining: {len(self.data) - len(self.labeled_data)}")
        print(f"Session duration: {session_duration:.1f} minutes")
        if labeled_this_session > 0:
            print(f"Average time per label: {session_duration/labeled_this_session:.1f} minutes")
        print(f"\nProgress saved to: {self.progress_file}")
    
    def save_progress(self):
        """Save current progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.labeled_data, f, indent=2)
    
    def show_statistics(self):
        """Display labeling statistics"""
        if not self.labeled_data:
            print("No labeled data yet!")
            return
        
        preferences = [item.get('preference', 'unknown') for item in self.labeled_data]
        
        print(f"\n{'='*80}")
        print(f"LABELING STATISTICS")
        print(f"{'='*80}")
        print(f"Total labeled: {len(self.labeled_data)}")
        print(f"\nPreference breakdown:")
        print(f"  Response A preferred: {preferences.count('a')}")
        print(f"  Response B preferred: {preferences.count('b')}")
        print(f"  Equal quality: {preferences.count('equal')}")
        
        # Show which temperature was preferred more
        a_count = preferences.count('a')
        b_count = preferences.count('b')
        if a_count + b_count > 0:
            temp_a = self.data[0]['metadata']['temp_a']
            temp_b = self.data[0]['metadata']['temp_b']
            print(f"\nTemperature analysis:")
            print(f"  Temp {temp_a} (Response A) won: {a_count} times ({a_count/(a_count+b_count)*100:.1f}%)")
            print(f"  Temp {temp_b} (Response B) won: {b_count} times ({b_count/(a_count+b_count)*100:.1f}%)")

def review_labeled_samples(labeled_file, num_samples=5):
    """Review some random labeled examples"""
    import random
    
    with open(labeled_file, 'r') as f:
        labeled = json.load(f)
    
    if len(labeled) == 0:
        print("No labeled examples yet!")
        return
    
    samples = random.sample(labeled, min(num_samples, len(labeled)))
    
    print(f"\n{'='*80}")
    print(f"REVIEWING {len(samples)} RANDOM LABELED EXAMPLES")
    print(f"{'='*80}")
    
    for i, item in enumerate(samples, 1):
        print(f"\n{'─'*80}")
        print(f"SAMPLE {i}")
        print(f"{'─'*80}")
        print(f"\nPrompt: {item['prompt']}")
        print(f"\nPreference: {item.get('preference', 'unknown')}")
        print(f"Reasoning: {item.get('reasoning', 'No reasoning provided')}")
        print(f"\nChosen response: {item['chosen'][:200]}...")
        print(f"\nRejected response: {item['rejected'][:200]}...")

# Main execution
if __name__ == "__main__":
    print("FITNESS PREFERENCE LABELING TOOL")
    print("="*80)
    
    # Load the response pairs
    data_file = 'fitness_response_pairs.json'
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        print("Make sure you've run the response generation script first.")
        exit(1)
    
    labeler = PreferenceLabelingTool(data_file)
    
    # Main menu
    while True:
        print("\n" + "="*80)
        print("MENU")
        print("="*80)
        print("1. Start/Continue labeling")
        print("2. Show statistics")
        print("3. Review random samples")
        print("4. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == '1':
            # Ask how many to label
            num_input = input(f"\nHow many examples to label? (Enter for all remaining): ").strip()
            num_to_label = int(num_input) if num_input else None
            
            labeler.label_session(
                start_idx=len(labeler.labeled_data),
                num_examples=num_to_label,
                auto_save_frequency=5
            )
        
        elif choice == '2':
            labeler.show_statistics()
        
        elif choice == '3':
            if os.path.exists(labeler.progress_file):
                review_labeled_samples(labeler.progress_file, num_samples=5)
            else:
                print("No labeled data yet!")
        
        elif choice == '4':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice!")
