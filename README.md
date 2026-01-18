# RLHF Fitness Dataset Project

A complete pipeline for creating a Reinforcement Learning from Human Feedback (RLHF) preference dataset in the fitness domain.

## 🎯 Project Overview

This project demonstrates the end-to-end process of creating an RLHF dataset:
1. Analyzing existing datasets to find gaps
2. Generating domain-specific prompts using LLMs
3. Creating response pairs with varied approaches
4. Human preference labeling
5. Dataset formatting and publication

## 📊 Dataset

The resulting dataset is published on Hugging Face:
**[Your HuggingFace Dataset URL]**

- **Domain**: Fitness and exercise advice
- **Size**: ~60 preference pairs (v1.0)
- **Splits**: Train (80%) / Validation (10%) / Test (10%)
- **License**: MIT

## 🛠️ Pipeline Scripts

### 1. Dataset Analysis
```bash
python scripts/analyze_dataset.py
```
Analyzes the HH-RLHF dataset to identify domain gaps and underrepresented topics.

### 2. Prompt Generation
```bash
python scripts/generate_prompts.py
```
Uses Claude API to generate diverse fitness-related questions.

### 3. Response Generation
```bash
python scripts/generate_responses.py
```
Generates multiple responses per prompt using different temperature settings.

### 4. Preference Labeling
```bash
python scripts/label_preferences.py
```
Interactive tool for labeling response preferences with quality criteria.

### 5. Dataset Formatting
```bash
python scripts/format_dataset.py
```
Formats labeled data and creates train/validation/test splits.

### 6. Upload to Hugging Face
```bash
python scripts/upload_to_huggingface.py
```
Publishes the formatted dataset to Hugging Face Hub.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Anthropic API key
- Hugging Face account

### Installation

1. Clone this repository:
```bash
git clone https://github.com/[your-username]/rlhf-fitness-dataset.git
cd rlhf-fitness-dataset
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your API key:
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

### Usage

Run the scripts in order:
```bash
# 1. Analyze existing datasets
python scripts/analyze_dataset.py

# 2. Generate prompts
python scripts/generate_prompts.py

# 3. Generate response pairs
python scripts/generate_responses.py

# 4. Label preferences
python scripts/label_preferences.py

# 5. Format dataset
python scripts/format_dataset.py

# 6. Upload to Hugging Face
python scripts/upload_to_huggingface.py
```

## 📝 Labeling Criteria

Preferences were labeled based on:
1. **Accuracy**: Scientific and medical correctness
2. **Safety**: Appropriate warnings and precautions
3. **Clarity**: Easy to understand and well-organized
4. **Completeness**: Fully addresses the question
5. **Actionability**: Practical and specific advice

## 📈 Results

**v1.0 Statistics:**
- Total examples: ~60
- Equal responses excluded: ~40%
- Train/Val/Test: 80/10/10 split

**Lessons Learned:**
- High equal rate (40%) suggests need for more response differentiation
- Future versions will use varied system prompts instead of just temperature
- Quality over quantity for initial dataset

## 🔮 Future Improvements (v2.0)

- [ ] Increase dataset size to 100+ examples
- [ ] Use different response generation strategies (not just temperature)
- [ ] Reduce equal response rate to <20%
- [ ] Add multiple labelers for inter-annotator agreement
- [ ] Expand to additional fitness subtopics

## 🤝 Contributing

This is a learning project, but contributions and suggestions are welcome! Feel free to:
- Open issues for bugs or suggestions
- Submit PRs for improvements
- Share your experience using the dataset

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Built using Claude (Anthropic) for prompt and response generation
- Inspired by Anthropic's HH-RLHF dataset methodology
- Thanks to the Hugging Face team for the datasets library

## 📬 Contact

[Your Name]
- GitHub: [@your-username](https://github.com/your-username)
- HuggingFace: [your-hf-username](https://huggingface.co/your-hf-username)
- Email: your.email@example.com (optional)

---

**Tags:** `RLHF` `preference-learning` `dataset` `fitness` `machine-learning` `huggingface`
