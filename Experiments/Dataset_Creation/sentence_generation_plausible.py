"""
Plausible Sentence Generator

This script generates center-embedded sentences that follow plausible expectations
using OpenAI's GPT-4.


"""
from openai import OpenAI
import json
import os
from datetime import datetime
import re

api_key = "api_key"

# Define models and their prices (per 1M tokens)
pricing_per_million = {
    "gpt-4": {"prompt": 30.00, "completion": 60.00}
}

model = "gpt-4"
client = OpenAI(api_key=api_key)


def get_level_specifications(complexity_level):
    """
        Get detailed specifications for each complexity level.

        Center-embedded sentences have different structural requirements based on
        their complexity level (number of nested clauses).

        Args:
            complexity_level (int): The complexity level (1-6 supported)

        Returns:
            dict: Dictionary containing:
                - entities (int): Number of entities required
                - verbs (int): Number of verbs required
                - structure (str): Grammatical structure pattern
                - semantic_rule (str): Rule for action flow
                - example (str): Example sentence at this level
                - flow (str): Action sequence explanation
        """
    specifications = {
    1: {
        "entities": 2, "verbs": 2,
        "structure": "[A] that [B] [verb1] [verb2]",
        "semantic_rule": "B performs verb1 TO A, then A performs verb2",
        "example": "The mouse that the cat chased escaped.",
        "flow": "CAT chased MOUSE → MOUSE escaped"
    },
    2: {
        "entities": 3, "verbs": 3,
        "structure": "[A] that [B] that [C] [verb1] [verb2] [verb3]",
        "semantic_rule": "C performs verb1 TO B, B performs verb2 TO A, A performs verb3",
        "example": "The fly that the spider that the bird saw stalked buzzed.",
        "flow": "BIRD saw SPIDER → SPIDER stalked FLY → FLY buzzed"
    },
    3: {
        "entities": 4, "verbs": 4,
        "structure": "[A] that [B] that [C] that [D] [verb1] [verb2] [verb3] [verb4]",
        "semantic_rule": "D performs verb1 TO C, C performs verb2 TO B, B performs verb3 TO A, A performs verb4",
        "example": "The worm that the bird that the cat that the dog chased saw ate died.",
        "flow": "DOG chased CAT → CAT saw BIRD → BIRD ate WORM → WORM died"
    },
    4: {
        "entities": 5, "verbs": 5,
        "structure": "[A] that [B] that [C] that [D] that [E] [verb1] [verb2] [verb3] [verb4] [verb5]",
        "semantic_rule": "E performs verb1 TO D, D performs verb2 TO C, C performs verb3 TO B, B performs verb4 TO A, A performs verb5",
        "example": "The mouse that the cat that the dog that the owner that the neighbor called trained chased caught squeaked.",
        "flow": "NEIGHBOR called OWNER → OWNER trained DOG → DOG chased CAT → CAT caught MOUSE → MOUSE squeaked"
    },
    5: {
        "entities": 6, "verbs": 6,
        "structure": "[A] that [B] that [C] that [D] that [E] that [F] [verb1] [verb2] [verb3] [verb4] [verb5] [verb6]",
        "semantic_rule": "F performs verb1 TO E, E performs verb2 TO D, D performs verb3 TO C, C performs verb4 TO B, B performs verb5 TO A, A performs verb6",
        "example": "The ant that the spider that the lizard that the snake that the hawk that the hunter saw spotted followed grabbed saw crawled.",
        "flow": "HUNTER saw HAWK → HAWK spotted SNAKE → SNAKE followed LIZARD → LIZARD grabbed SPIDER → SPIDER saw ANT → ANT crawled"
    },
    6: {
        "entities": 7, "verbs": 7,
        "structure": "[A] that [B] that [C] that [D] that [E] that [F] that [G] [verb1] [verb2] [verb3] [verb4] [verb5] [verb6] [verb7]",
        "semantic_rule": "G performs verb1 TO F, F performs verb2 TO E, E performs verb3 TO D, D performs verb4 TO C, C performs verb5 TO B, B performs verb6 TO A, A performs verb7",
        "example": "The crumb that the ant that the spider that the lizard that the snake that the hawk that the eagle observed followed chased startled carried dropped rolled.",
        "flow": "EAGLE observed HAWK → HAWK followed SNAKE → SNAKE chased LIZARD → LIZARD startled SPIDER → SPIDER carried ANT → ANT dropped CRUMB → CRUMB rolled"
    }
    }
    # Return specification for the requested level
    return specifications.get(complexity_level, {
        "entities": complexity_level + 1,
        "verbs": complexity_level + 1,
        "structure": f"Pattern with {complexity_level} embeddings",
        "semantic_rule": f"Innermost entity acts first, actions flow outward",
        "example": f"Complex example needed for level {complexity_level}",
        "flow": f"Action sequence for {complexity_level} embeddings"
    })


def get_system_prompt(complexity_level):
    """
        Generate comprehensive system prompt for GPT-4 to create center-embedded sentences.

        The system prompt provides detailed instructions about grammatical structure,
        semantic rules, temporal consistency, and quality requirements for generating
        linguistically valid center-embedded sentences.

        Args:
            complexity_level (int): Target complexity level for sentence generation

        Returns:
            str: Formatted system prompt with all specifications and constraints
    """
    spec = get_level_specifications(complexity_level)

    return f"""You are a linguistics expert specializing in center-embedded sentences.

    # COMPLEXITY LEVEL {complexity_level} SPECIFICATIONS
    - Required entities: {spec['entities']}
    - Required verbs: {spec['verbs']}
    - Required "that" clauses: {complexity_level}
    - Embedding depth: {complexity_level} levels
    
    # GRAMMATICAL STRUCTURE
    Pattern: {spec['structure']}
    
    # SEMANTIC FLOW RULES (CRITICAL)
    Core Rule: {spec['semantic_rule']}
    
    Example: {spec['example']}
    → Action Flow: {spec['flow']}
    
    # TEMPORAL CONSISTENCY RULES (MANDATORY)
    1. Actions must follow logical temporal sequence
    2. Dead entities CANNOT perform subsequent actions
    3. Caught/trapped/seized entities CANNOT act on other entities
    4. Eaten entities CANNOT perform actions after being consumed
    5. Actions must respect cause-and-effect relationships
    6. No temporal paradoxes or impossibilities allowed
    
    # STRUCTURAL REQUIREMENTS
    1. Use "that" as relative pronoun for ALL embeddings
    2. Verbs appear in REVERSE order of entity introduction
    3. Last entity introduced performs FIRST action
    4. First entity performs FINAL action (must be intransitive)
    5. Each "that" clause introduces exactly ONE entity
    6. Actions flow from innermost clause outward
    
    # SEMANTIC PLAUSIBILITY CONSTRAINTS
    Predator-Prey Relationships:
    - Must reflect realistic natural hierarchies
    - Size/strength differences must be logical
    - Hunting behaviors must be species-appropriate
    
    Professional Relationships:
    - Authority structures must be realistic
    - Professional interactions must be plausible
    - Skills must match occupations
    
    Physical Capabilities:
    - Actions must match entity capabilities
    - Environmental constraints must be respected
    - Biological limitations must be observed
    
    # MEMORY AID: NESTED ACTION PRINCIPLE
    Think of Russian dolls opening from inside out:
    - Innermost doll (last entity) acts first
    - Each outer doll (entity) acts on the result
    - Outermost doll (first entity) performs final action
    - Each action must be temporally possible given previous actions
    
    # QUALITY REQUIREMENTS
    - Sentences must be grammatically perfect
    - Semantic relationships must be crystal clear
    - No ambiguous temporal references
    - All actions must be logically sequenced
    - Maintain subject-verb agreement throughout"""


def generate_user_prompt(complexity_level):
    """
        Generate specific user prompt requesting center-embedded sentences.

        Creates a detailed prompt that specifies exact requirements for the target
        complexity level, including entity categories, structural constraints, and
        output formatting instructions.

        Args:
            complexity_level (int): Target complexity level for sentence generation

        Returns:
            str: Formatted user prompt with requirements and entity vocabulary
    """
    spec = get_level_specifications(complexity_level)

    return f"""Generate 30 unique center-embedded sentences at complexity level {complexity_level}.

    # STRICT REQUIREMENTS
    ✓ Exactly {spec['entities']} different entities
    ✓ Exactly {spec['verbs']} verbs  
    ✓ Exactly {complexity_level} "that" clauses
    ✓ Perfect temporal consistency - NO temporal violations
    ✓ Semantically plausible relationships
    ✓ Grammatically correct structure
    
    # ENTITY CATEGORIES
    
    Animals:
    eagle, sparrow, crow, pigeon, parrot, chicken, duck, owl, goose, vulture, salmon, tuna, goldfish, trout, shark, whale, piranha, starfish, dog, cat, horse, lion, cow, tiger, bear, elephant, deer, pig, giraffe, mouse, sheep, goat, rat, wolf, zebra, donkey, rabbit, raccoon, squirrel, coyote, cougar, moose, cheetah, rhinoceros, fox, mule, hippopotamus, beaver, camel, ferret, frog, jaguar, lamb, leopard, lizard, llama, skunk, rattlesnake, cobra, python, anaconda, viper, black mamba, ant, fly, bee, spider, beetle, mosquito, cockroach, wasp, ladybug, butterfly, grasshopper, moth, gnat, cricket, caterpillar, worm, centipede, termite, praying mantis, bug, dragonfly
    
    People (Occupations):
    doctor, lawyer, teacher, nurse, police officer, firefighter, accountant, dentist, engineer, plumber, carpenter, salesperson, secretary, cashier, mechanic, professor, electrician, chef, scientist, clerk, banker, actor, truck driver, mailman, artist, athlete, attorney, bus driver, CEO, construction worker, garbage man, janitor, judge, manager, musician, pilot, politician, programmer, surgeon, veterinarian, waiter, writer
    
    Vehicles:
    car, truck, motorcycle, airplane, bicycle, bus, SUV, boat, train, van, jeep, tank, sedan, tractor, scooter, RV, taxi, ATV, ambulance, convertible, golf cart, helicopter, pickup truck, ship, sled, snowmobile, sports car
    
    
    # CRITICAL REMINDERS
    - After an entity is caught/killed/eaten, it CANNOT perform actions
    - Predator-prey relationships must be biologically accurate  
    - Professional hierarchies must be realistic
    - Actions must follow logical temporal sequence
    - Each sentence must tell a coherent, plausible story
    
    # OUTPUT FORMAT
    Number each sentence 1-30, one per line:
    1. [sentence]
    2. [sentence]
    ...
    30. [sentence]
    
    """


def validate_sentence_structure(sentence, complexity_level):
    """
    Validate sentence structure.

    Performs multiple validation checks to ensure the generated sentence
    meets the structural requirements for center-embedded
    sentences at the specified complexity level.

    Args:
        sentence (str): The sentence to validate
        complexity_level (int): Expected complexity level

    Returns:
        tuple: (is_valid (bool), reason (str))
            - is_valid: True if sentence passes all validation checks
            - reason: Explanation of validation result or failure reason
    """

    # Count 'that' clauses
    that_matches = re.findall(r'\bthat\b', sentence.lower())
    that_count = len(that_matches)

    if that_count != complexity_level:
        return False, f"Expected {complexity_level} 'that' clauses, found {that_count}"

    # Check sentence completion
    if not sentence.strip().endswith('.'):
        return False, "Sentence doesn't end with period"

    # Check minimum length
    words = sentence.split()
    expected_min_words = complexity_level * 2 + 4
    if len(words) < expected_min_words:
        return False, f"Sentence too short: {len(words)} words, expected at least {expected_min_words}"

    return True, "Valid"


def parse_and_validate_sentences(response_content, complexity_level):
    """
        Parse GPT-4 response and validate each sentence.

        Extracts numbered sentences from the model response and validates
        each one for structural correctness.

        Args:
            response_content (str): Raw response text from GPT-4
            complexity_level (int): Expected complexity level for validation

        Returns:
            tuple: (valid_sentences (list), invalid_sentences (list))
                - valid_sentences: List of sentences that passed validation
                - invalid_sentences: List of tuples (sentence, failure_reason)
    """
    lines = response_content.strip().split('\n')
    valid_sentences = []
    invalid_sentences = []

    for line in lines:
        line = line.strip()
        if line and line[0].isdigit():
            # Extract sentence
            if '. ' in line:
                sentence = line.split('. ', 1)[1].strip()
            elif '- ' in line:
                sentence = line.split('- ', 1)[1].strip()
            else:
                continue

            if sentence:
                is_valid, reason = validate_sentence_structure(sentence, complexity_level)
                if is_valid:
                    valid_sentences.append(sentence)
                else:
                    invalid_sentences.append((sentence, reason))

    return valid_sentences, invalid_sentences


def generate_sentences_for_complexity(complexity_level, max_attempts=10):
    """
        Generate 30 valid center-embedded sentences for a specific complexity level.

        Uses iterative generation with validation to ensure we get exactly 30
        high-quality sentences. Will retry up to max_attempts times if needed.

        Args:
            complexity_level (int): Target complexity level (1-6)
            max_attempts (int): Maximum number of generation attempts

        Returns:
            dict: Results dictionary containing:
                - sentences (list): List of valid generated sentences
                - metadata (dict): Generation metadata including cost, attempts, etc.
    """
    all_valid_sentences = []
    attempt = 0
    total_cost = 0

    # Continue generating until we have 30 valid sentences or hit max attempts
    while len(all_valid_sentences) < 30 and attempt < max_attempts:
        attempt += 1
        print(f"\nAttempt {attempt} for complexity level {complexity_level}...")

        system_prompt = get_system_prompt(complexity_level)
        user_prompt = generate_user_prompt(complexity_level)

        system_message = {"role": "system", "content": system_prompt}
        user_message = {"role": "user", "content": user_prompt}

        response = client.chat.completions.create(
            model=model,
            messages=[system_message, user_message],
            temperature=0.7
        )

        content = response.choices[0].message.content
        usage = response.usage

        # Calculate cost
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        cost = (prompt_tokens / 1_000_000) * pricing_per_million[model]["prompt"] + \
               (completion_tokens / 1_000_000) * pricing_per_million[model]["completion"]
        total_cost += cost

        # Parse and validate
        valid_sentences, invalid_sentences = parse_and_validate_sentences(content, complexity_level)

        print(f"Generated {len(valid_sentences)} valid sentences, {len(invalid_sentences)} invalid")

        # Show examples of invalid sentences for debugging
        if invalid_sentences:
            print("\nInvalid sentences (showing first 3):")
            for sent, reason in invalid_sentences[:3]:
                print(f"- {reason}: {sent[:60]}...")

        # Add valid sentences
        all_valid_sentences.extend(valid_sentences)
        all_valid_sentences = list(dict.fromkeys(all_valid_sentences))  # Remove duplicates

        # Check if we have enough sentences
        if len(all_valid_sentences) >= 30:
            all_valid_sentences = all_valid_sentences[:30]
            print(f"\nSuccess! Generated 30 valid sentences for complexity {complexity_level}")
        else:
            print(f"\nNeed {30 - len(all_valid_sentences)} more sentences...")

    print(f"\nTotal cost for complexity {complexity_level}: ${total_cost:.5f}")

    return {
        "sentences": all_valid_sentences,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "complexity_level": complexity_level,
            "attempts": attempt,
            "total_cost": total_cost,
            "valid_count": len(all_valid_sentences)
        }
    }


def save_to_json(data, filename="output_file.json"):
    """
    Save or update JSON file with sentence generation results.

    Loads existing data if file exists, updates it with new results,
    and saves back to disk. This allows incremental updates across
    multiple complexity levels.

    Args:
        data (dict): New data to save/update
        filename (str): Output JSON filename
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    existing_data.update(data)

    with open(filename, 'w') as f:
        json.dump(existing_data, f, indent=2)

    print(f"\nData saved to {filename}")


def main():
    """
        Main function to generate center-embedded sentences for specified complexity levels.

        Orchestrates the entire generation process:
        1. Iterates through specified complexity levels
        2. Generates 30 sentences for each level
        3. Saves results incrementally
        4. Reports costs and shows sample outputs
        5. Provides final summary
    """
    complexity_levels = [6]  # Specify desired complexity levels

    all_results = {}
    total_cost = 0

    for level in complexity_levels:
        print(f"\n{'=' * 60}")
        print(f"GENERATING COMPLEXITY LEVEL {level} SENTENCES")
        print(f"{'=' * 60}")

        result = generate_sentences_for_complexity(level)

        if len(result["sentences"]) < 30:
            print(f"\nWarning: Only generated {len(result['sentences'])} valid sentences for level {level}")

        all_results[f"complexity_{level}"] = result
        total_cost += result["metadata"]["total_cost"]

        # Save after each level
        save_to_json(all_results)

        # Show samples
        print(f"\nSample sentences from complexity {level}:")
        for i, sent in enumerate(result["sentences"][:3], 1):
            print(f"{i}. {sent}")

    print(f"\n{'=' * 50}")
    print(f"TOTAL COST: ${total_cost:.5f} USD")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()