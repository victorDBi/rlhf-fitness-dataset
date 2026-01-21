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
git clone https://github.com/victorDBi/rlhf-fitness-dataset.git
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
1. **CORRECT** (Most Important) - Information is accurate and not made up. Any medical or scientific advice is accurate and can be verified.  When providing advice responses should distinguish between what is objectiver vs what is unique to each person.  
2. **CLARITY**  Clear explanations.  Wherever possible, responses should clarify the difference between beginner and expert levels.  More categorization early in the response is most helpful.  Where relevant, suggestions should include a specific example - particularly when a user asks for sample workouts or exercises.  And the more concrete those examples are, the better.  Even expert athletes may have knowledge gaps and examples provide the most value.  The response should have internal consistency, so if the prompt asks for a 20 min workout, the response should total 20mins (or less).
3. **SIMPLICITY**  Fitness can have niche responses, but unless a very specific niche is asked for in the prompt, the response should assume that the person is asking a generalized question and the response should be relevant in the broadest application.  Limit the use of jargon unless it is particularly relevant.  Match the user's implied fitness level if provided.
4. **SAFETY** - Addresses potential issues and provides suggestions for safety practices associated with fitness.  Include safety warnings when appropriate, but does not recommend consulting a doctor when unnecessary or causes undue medical anxiety
5. **CONCISENESS** - Not overly verbose - Gets to the point - But not cryptic and response is well organized.


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
- GitHub: [@victorDBi](https://github.com/victorDBi)
- HuggingFace: [victor203](https://huggingface.co/victor203)
- Email: victorjdavidson@gmail.com 

---

**Tags:** `RLHF` `preference-learning` `dataset` `fitness` `machine-learning` `huggingface`
