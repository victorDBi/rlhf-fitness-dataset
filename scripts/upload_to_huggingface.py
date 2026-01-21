# upload_to_huggingface.py
from huggingface_hub import HfApi, create_repo, upload_folder
from datasets import Dataset, DatasetDict, load_dataset
import json
import os

def load_dataset_from_jsonl(data_dir='fitness_dataset'):
    """Load the dataset from JSONL files"""
    
    def load_jsonl(file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    # Load each split
    train_data = load_jsonl(os.path.join(data_dir, 'train.jsonl'))
    val_data = load_jsonl(os.path.join(data_dir, 'validation.jsonl'))
    test_data = load_jsonl(os.path.join(data_dir, 'test.jsonl'))
    
    # Create Dataset objects
    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })
    
    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Validation: {len(val_data)} examples")
    print(f"  Test: {len(test_data)} examples")
    print(f"  Total: {len(train_data) + len(val_data) + len(test_data)} examples")
    
    return dataset_dict

def upload_to_hf(dataset_dict, repo_name, data_dir='fitness_dataset'):
    """
    Upload dataset to Hugging Face Hub
    
    Args:
        dataset_dict: DatasetDict with train/val/test splits
        repo_name: Name in format "username/dataset-name"
        data_dir: Directory containing README.md
    """
    
    print("\n" + "="*80)
    print("UPLOADING TO HUGGING FACE")
    print("="*80)
    
    # Verify repo name format
    if '/' not in repo_name:
        print("\nError: repo_name must be in format 'username/dataset-name'")
        print("Example: 'victor-davidson/fitness-preferences'")
        return False
    
    try:
        # Step 1: Create repository
        print(f"\nStep 1: Creating repository '{repo_name}'...")
        create_repo(
            repo_id=repo_name,
            repo_type="dataset",
            exist_ok=True,
            private=False  # Set to True if you want a private dataset
        )
        print("  ✓ Repository created/verified")
        
        # Step 2: Push dataset
        print(f"\nStep 2: Pushing dataset to Hub...")
        dataset_dict.push_to_hub(repo_name)
        print("  ✓ Dataset files uploaded")
        
        # Step 3: Upload README.md (dataset card)
        print(f"\nStep 3: Uploading dataset card (README.md)...")
        api = HfApi()
        readme_path = os.path.join(data_dir, 'README.md')
        
        if os.path.exists(readme_path):
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_name,
                repo_type="dataset",
            )
            print("  ✓ Dataset card uploaded")
        else:
            print("  ⚠️  README.md not found, skipping")
        
        print("\n" + "="*80)
        print("UPLOAD COMPLETE!")
        print("="*80)
        print(f"\n🎉 Your dataset is now live at:")
        print(f"https://huggingface.co/victor203/{repo_name}")
        print(f"\nView it in your browser or use it in code:")
        print(f"```python")
        print(f"from datasets import load_dataset")
        print(f"dataset = load_dataset('{repo_name}')")
        print(f"```")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during upload: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Check your repo name format: 'username/dataset-name'")
        print("3. Verify your token has write permissions")
        return False

def test_dataset_loading(repo_name):
    """Test that the uploaded dataset can be loaded"""
    print("\n" + "="*80)
    print("TESTING DATASET LOADING")
    print("="*80)
    
    try:
        print(f"\nAttempting to load '{repo_name}'...")
        dataset = load_dataset(repo_name)
        
        print("  ✓ Dataset loaded successfully!")
        print(f"\nDataset structure:")
        print(dataset)
        
        print(f"\nFirst training example:")
        print(dataset['train'][0])
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        return False

# Main execution
if __name__ == "__main__":
    print("HUGGING FACE DATASET UPLOAD")
    print("="*80)
    
    # Step 1: Login check
    print("\nMake sure you're logged in to Hugging Face:")
    print("Run: huggingface-cli login")
    print("\nIf you haven't logged in yet, press Ctrl+C to exit and login first.")
    input("\nPress Enter to continue...")
    
    # Step 2: Get repository name
    print("\n" + "="*80)
    print("REPOSITORY SETUP")
    print("="*80)
    
    username = input("\nEnter your Hugging Face username: ").strip()
    dataset_name = input("Enter dataset name (e.g., 'fitness-preferences'): ").strip()
    
    # Create full repo name
    repo_name = f"{username}/{dataset_name}"
    
    print(f"\nYour dataset will be uploaded to:")
    print(f"https://huggingface.co/datasets/{repo_name}")
    
    confirm = input("\nProceed with upload? (yes/no): ").lower().strip()
    
    if confirm not in ['yes', 'y']:
        print("Upload cancelled.")
        exit(0)
    
    # Step 3: Load dataset
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    
    dataset_dict = load_dataset_from_jsonl('fitness_dataset')
    
    # Step 4: Upload
    success = upload_to_hf(dataset_dict, repo_name, data_dir='fitness_dataset')
    
    # Step 5: Test loading (optional)
    if success:
        test_load = input("\nTest loading the dataset? (y/n): ").lower().strip()
        if test_load in ['y', 'yes']:
            # Wait a moment for propagation
            print("\nWaiting 5 seconds for dataset to propagate...")
            import time
            time.sleep(5)
            test_dataset_loading(repo_name)
    
    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    
    if success:
        print("\n✅ Your first RLHF dataset is complete!")
        print("\nWhat you accomplished:")
        print("  ✓ Analyzed existing RLHF datasets")
        print("  ✓ Generated domain-specific prompts")
        print("  ✓ Created response pairs")
        print(f"  ✓ Labeled {len(dataset_dict['train']) + len(dataset_dict['validation']) + len(dataset_dict['test'])} preferences")
        print("  ✓ Formatted and split the dataset")
        print("  ✓ Published to Hugging Face")
        print("\nNext steps for v2:")
        print("  • Generate responses with more differentiation")
        print("  • Add more prompts to reach 100+ examples")
        print("  • Consider adding multiple labelers")
        print("  • Expand to other fitness subtopics")