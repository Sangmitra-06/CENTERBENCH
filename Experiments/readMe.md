# CenterBench Experiments

This folder contains all scripts and resources for generating the CenterBench dataset and evaluating model performance. The folder is organized into two main subfolders:

## Subfolders and Contents

### `Dataset_Creation/`
Scripts and resources for generating the dataset:
- **nouns_and_verbs_implausible.json**: List of nouns and verbs used for generating implausible sentences.
- **sentence_generation_plausible.py**: Script for generating plausible center-embedded sentences.
- **sentence_generation_implausible.py**: Script for generating implausible center-embedded sentences.
- **question_and_answer_generation_plausible.py**: Script for generating question-answer pairs for plausible sentences.
- **question_and_answer_generation_implausible.py**: Script for generating question-answer pairs for implausible sentences.

### `Evaluation/`
Scripts for evaluating model responses on the dataset:
- **generate_response_and_evaluate.py**: Main script for generating model responses and evaluating them against the dataset answers.