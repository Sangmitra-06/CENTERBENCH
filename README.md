# 🧩CENTERBENCH

This is the repository for [The Dog the Cat Chased Stumped the Model:
Measuring When Language Models Abandon Structure for Shortcuts](https://arxiv.org/abs/2502.05331). 

**📦 Dataset:** Available on [Hugging Face](https://huggingface.co/datasets/Sangmitra-06/CENTERBENCH)

Authors: Sangmitra Madhusudan, Kaige Chen, and Ali Emami

## 📄 Paper abstract

When language models correctly parse "The cat that the dog chased meowed," are they analyzing syntax or simply familiar with dogs chasing cats? Despite extensive benchmarking, we lack methods to distinguish structural understanding from semantic pattern matching. We introduce **CENTERBENCH**, a dataset of 9,720 comprehension questions on center-embedded sentences (like "The cat [that the dog chased] meowed") where relative clauses nest recursively, creating processing demands from simple to deeply nested structures. Each sentence has a syntactically identical but semantically implausible counterpart (e.g., mailmen prescribe medicine, doctors deliver mail) and six comprehension questions testing surface understanding, syntactic dependencies, and causal reasoning. Testing six models reveals that performance gaps between plausible and implausible sentences widen systematically with complexity, with models showing median gaps up to 26.8 percentage points, quantifying when they abandon structural analysis for semantic associations. Notably, semantic plausibility harms performance on questions about resulting actions, where following causal relationships matters more than semantic coherence. Reasoning models improve accuracy but their traces show semantic shortcuts, overthinking, and answer refusal. Unlike models whose plausibility advantage systematically widens with complexity, humans shows variable semantic effects. CenterBench provides the first framework to identify when models shift from structural analysis to pattern matching.

## 📁 Repository Structure
```
.
├── Dataset/
│   ├── plausible_subset.json
│   └── implausible_subset.json
└── Experiments/
    ├── Dataset_Creation/
    │   ├── nouns_and_verbs_implausible.json
    │   ├── sentence_generation_plausible.py
    │   ├── sentence_generation_implausible.py
    │   ├── question_and_answer_generation_plausible.py
    │   └── question_and_answer_generation_implausible.py
    └── Evaluation/
        └── generate_response_and_evaluate.py
```

## 🗃️ Dataset

The `Dataset/` folder contains the CenterBench dataset files:

- **`plausible_subset.json`**: Sentences and question-answer pairs in the plausible subset
- **`implausible_subset.json`**: Sentences and question-answer pairs in the implausible subset

### Dataset Structure

Each JSON file is organized by complexity level (`complexity_1` to `complexity_6`). For each complexity level, there is a `sentences` array containing sentence objects with the following structure:

- **`id`**: Unique identifier for the sentence
- **`sentence`**: The center-embedded sentence text
- **`structure`**: The event chain as a list of subject-action-object triples (ordered from main clause to deepest embedding)
- **`middle_entity`**: The entity at the center of the embedding
- **`all_entities`**: List of all entities in the sentence
- **`questions_by_entity`**: For each entity, a list of question-answer pairs with:
  - `question`: The question text
  - `answer`: The correct answer
  - `type`: Question type (action_performed, agent_identification, entity_count, nested_dependency, causal_sequence, chain_consequence)
  - `difficulty`: Difficulty level (easy, medium, hard)
  - `entity`: The entity this question focuses on
  - `is_middle_entity`: Boolean indicating if this is the middle entity
- **`total_questions`**: Total number of questions for this sentence

### Example Entry

```json
{
  "id": "complexity_1_sentence_1",
  "sentence": "The train that the airplane whistled at taxied.",
  "structure": [
    {
      "subject": "train",
      "action": "taxied",
      "object": null
    },
    {
      "subject": "airplane",
      "action": "whistled at",
      "object": "train"
    }
  ],
  "middle_entity": "train",
  "all_entities": ["airplane", "train"],
  "questions_by_entity": {
    "airplane": [
      {
        "question": "What did the airplane do?",
        "answer": "whistle at the train",
        "type": "action_performed",
        "difficulty": "easy",
        "entity": "airplane",
        "is_middle_entity": false
      },
      {
        "question": "How many distinct entities are in the sentence?",
        "answer": "2",
        "type": "entity_count",
        "difficulty": "medium",
        "entity": "airplane",
        "is_middle_entity": false
      }
    ],
    "train": [
      {
        "question": "What did the train do?",
        "answer": "taxi",
        "type": "action_performed",
        "difficulty": "easy",
        "entity": "train",
        "is_middle_entity": true
      }
    ]
  },
  "total_questions": 12
}
```

## 🔧 Experiments

The `Experiments/` folder contains scripts for dataset generation and model evaluation.

### Dataset Creation (`Experiments/Dataset_Creation/`)

Scripts and resources for generating the CenterBench dataset:

- **`nouns_and_verbs_implausible.json`**: List of nouns and verbs used for generating implausible sentences
- **`sentence_generation_plausible.py`**: Generates plausible center-embedded sentences
- **`sentence_generation_implausible.py`**: Generates implausible center-embedded sentences
- **`question_and_answer_generation_plausible.py`**: Generates question-answer pairs for plausible sentences
- **`question_and_answer_generation_implausible.py`**: Generates question-answer pairs for implausible sentences

### Evaluation (`Experiments/Evaluation/`)

Scripts for evaluating model performance:

- **`generate_response_and_evaluate.py`**: Main script for generating model responses and evaluating them against dataset answers

## 🖥️ Usage

### 1. Clone the Repository

Clone this repository to access the complete CenterBench dataset:
```bash
git clone https://github.com/Sangmitra-06/CENTERBENCH.git
cd CENTERBENCH
```

### 2. Install Dependencies

Install the required dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Note for Gemini users:** You'll also need to install the Google Cloud CLI and authenticate:
```bash
# Install gcloud CLI (see https://cloud.google.com/sdk/docs/install)
gcloud auth login
```

### 3. Set Up API Keys

Create a `.env` file in the root directory with your API keys:
```env
TOGETHER_API_KEY=your_together_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
GEMINI_API_KEY=your_gemini_api_key
```

**For Gemini Vertex AI users:** Update the project parameters in `Experiments/Evaluation/generate_response_and_evaluate.py` (line 952):
```python
client = genai.Client(vertexai=True, project="your_project_name", location="your_project_location")
```

### 4. Load the Dataset
```python
import json

# Load plausible sentences
with open('Dataset/plausible_subset.json', 'r') as f:
    plausible_data = json.load(f)

# Load implausible sentences
with open('Dataset/implausible_subset.json', 'r') as f:
    implausible_data = json.load(f)

# Access sentences by complexity level
complexity_1_sentences = plausible_data['complexity_1']['sentences']
```

### 5. Run Evaluation

Use our evaluation script to test models on the dataset:
```bash
# Basic usage with a specific model
python Experiments/Evaluation/generate_response_and_evaluate.py Dataset/plausible_subset.json --model deepseek-ai/DeepSeek-V3

# Evaluate on implausible subset
python Experiments/Evaluation/generate_response_and_evaluate.py Dataset/implausible_subset.json --model claude-3-5-sonnet-20241022

# Additional options
python Experiments/Evaluation/generate_response_and_evaluate.py Dataset/plausible_subset.json \
    --model gemini-2.0-flash-thinking-exp-01-21 \
    --difficulty hard \
    --middle-entity \
    --output results.json
```

### Supported Models

The evaluation script supports models from:
- **Together AI**: Various open-source models
- **Anthropic**: Claude models (use `--extended-thinking` for reasoning traces)
- **DeepSeek**: DeepSeek-V3 and other DeepSeek models
- **Google**: Gemini models (requires Vertex AI setup)

## ✏️ Reference

Please use the following bibtex citation if this paper was a part of your work, thank you!
