# CenterBench Dataset

This folder contains representative slices of the CenterBench dataset, the full version of which will be released upon publication. The dataset is split into two main files:

- [`plausible_subset.json`](plausible_subset.json): Contains sample sentences and question-answer pairs from the plausible subset.
- [`implausible_subset.json`](implausible_subset.json): Contains sample sentences and question-answer pairs from the implausible subset.

## File Structure

Each JSON file is organized by complexity level (`complexity_1` to `complexity_6`). For each level, you will find:

- `sentences`: A list of sentence objects, each with:
  - `id`: Unique identifier for the sentence.
  - `sentence`: The center-embedded sentence.
  - `structure`: The event chain as a list of subject-action-object triples.
  - `middle_entity`: The entity at the center of the embedding.
  - `all_entities`: List of all entities in the sentence.
  - `questions_by_entity`: For each entity, a list of question-answer pairs probing comprehension and reasoning (e.g., action performed, agent identification, entity count, causal sequence, consequences, and nested dependencies).

- `total_questions`: The total number of questions for that complexity level.

## Example Entry

```json
{
  "id": "complexity_2_sentence_1",
  "sentence": "The bicycle that the car that the truck hit bumped fell over.",
  "structure": [
    {"subject": "bicycle", "action": "fell over", "object": null},
    {"subject": "car", "action": "bumped", "object": "bicycle"},
    {"subject": "truck", "action": "hit", "object": "car"}
  ],
  "middle_entity": "car",
  "all_entities": ["truck", "bicycle", "car"],
  "questions_by_entity": {
    "truck": [
      {
        "question": "What did the truck do?",
        "answer": "hit the car",
        "type": "action_performed",
        "difficulty": "easy",
        "entity": "truck",
        "is_middle_entity": false
      },
      ...
    ],
    ...
  }
}
```