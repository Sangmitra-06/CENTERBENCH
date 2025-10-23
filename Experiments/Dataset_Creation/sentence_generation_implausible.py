"""
Implausible Sentence Generator

This script generates center-embedded sentences that violate plausible expectations
by swapping verbs between entities in a circular pattern.


"""
import json
import random


def load_verb_data(filename):
    """
    Load verb data from a JSON file.

    Args:
        filename (str): Path to the JSON file containing verb data

    Returns:
        dict or None: Parsed JSON data if successful, None if error occurred

    Raises:
        Prints error messages for FileNotFoundError and JSONDecodeError
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find file '{filename}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file '{filename}'")
        return None


def get_all_entities_from_domain(domain_data):
    """
    Extract all entities from a domain, handling both flat and nested data structures.

    The function handles two data structures:
    1. Flat: {entity: {transitive: [...], intransitive: [...]}}
    2. Nested: {subdomain: {entity: {transitive: [...], intransitive: [...]}}}

    Args:
        domain_data (dict): Domain data containing entities and their verbs

    Returns:
        list: List of all entity names found in the domain
    """
    entities = []

    # Check if this is a flat structure (like vehicles) or nested (like animals)
    first_key = next(iter(domain_data.keys()))
    first_value = domain_data[first_key]

    if isinstance(first_value, dict) and 'transitive' in first_value:
        # Flat structure: direct entity -> verbs mapping
        entities = list(domain_data.keys())
    else:
        # Nested structure: subdomain -> entity -> verbs mapping
        for subdomain in domain_data.values():
            entities.extend(subdomain.keys())

    return entities


def get_entity_verbs(entity_name, domain_data):
    """
    Retrieve verb data for a specific entity, handling different data structures.

    Args:
        entity_name (str): Name of the entity to find verbs for
        domain_data (dict): Domain data containing entities and their verbs

    Returns:
        dict or None: Dictionary containing 'transitive' and 'intransitive' verb lists,
                     or None if entity not found
    """
    # Check if this is a flat structure first
    if entity_name in domain_data:
        return domain_data[entity_name]

    # If not flat, search in subdomains
    for subdomain in domain_data.values():
        if isinstance(subdomain, dict) and entity_name in subdomain:
            return subdomain[entity_name]

    return None


def generate_implausible_sentence(complexity, domain_name, domain_data, used_combinations):
    """
    Generate a single implausible center-embedded sentence.

    Implausible assignment works by circular verb swapping:
    - Entity A gets verbs typically used by Entity B
    - Entity B gets verbs typically used by Entity C
    - Entity C gets verbs typically used by Entity A

    Args:
        complexity (int): Sentence complexity level (1-6), determines number of embeddings
        domain_name (str): Name of the domain being used
        domain_data (dict): Verb data for the specified domain
        used_combinations (set): Set of previously used combinations to ensure uniqueness

    Returns:
        tuple: (sentence_string, domain_name)

    Raises:
        Exception: If not enough entities available or unable to generate unique sentence
    """
    # Complexity determines number of entities (complexity + 1)
    num_entities = complexity + 1

    # Get all available entities from the domain
    all_entities = get_all_entities_from_domain(domain_data)

    # Ensure we have enough entities for the requested complexity
    if len(all_entities) < num_entities:
        raise Exception(f"Not enough entities in {domain_name} domain for complexity {complexity}")

    # Generate unique combinations until we find one not used
    max_attempts = 1000
    for attempt in range(max_attempts):
        # Randomly select entities
        selected_entities = random.sample(all_entities, num_entities)

        # Apply circular verb swapping
        assigned_verbs = []
        for i in range(num_entities):
            current_entity = selected_entities[i]
            next_entity = selected_entities[(i + 1) % num_entities]

            # Get verbs from the next entity
            next_entity_verbs = get_entity_verbs(next_entity, domain_data)

            if next_entity_verbs is None:
                raise Exception(f"Could not find verbs for entity: {next_entity}")

            # Determine verb type based on position
            if i == 0:  # First entity gets intransitive (final position in sentence)
                verb_type = "intransitive"
            else:  # All other entities get transitive (embedded positions)
                verb_type = "transitive"

            # Randomly select a verb of the correct type
            available_verbs = next_entity_verbs[verb_type]
            selected_verb = random.choice(available_verbs)
            assigned_verbs.append(selected_verb)

        # Create combination key for uniqueness check
        combination_key = (domain_name, tuple(selected_entities), tuple(assigned_verbs))

        # Check if this combination has been used before
        if combination_key not in used_combinations:
            used_combinations.add(combination_key)
            break
    else:
        # If we couldn't generate a unique sentence after max attempts
        raise Exception(f"Could not generate unique sentence after {max_attempts} attempts for complexity {complexity}")

    # Build the final sentence structure
    sentence = construct_sentence(selected_entities, assigned_verbs)
    return sentence, domain_name


def construct_sentence(entities, verbs):
    """
    Construct a center-embedded sentence from entities and verbs.

    Creates sentences with the structure:
    Complexity 1: "The A that the B [verb2] [verb1]."
    Complexity 2: "The A that the B that the C [verb3] [verb2] [verb1]."
    And so on

    Args:
        entities (list): List of entity names in order
        verbs (list): List of verbs corresponding to each entity

    Returns:
        str: Completed center-embedded sentence with proper punctuation
    """
    if len(entities) == 2:
        # Complexity 1: "The A that the B [transitive] [intransitive]"
        return f"The {entities[0]} that the {entities[1]} {verbs[1]} {verbs[0]}."

    # For complexity 2+, build nested structure
    sentence = f"The {entities[0]} that "

    # Build the nested "that" clauses
    for i in range(1, len(entities) - 1):
        sentence += f"the {entities[i]} that "

    # Add the final entity and its verb
    sentence += f"the {entities[-1]} {verbs[-1]} "

    # Add the remaining verbs in reverse order
    for i in range(len(verbs) - 2, -1, -1):
        sentence += f"{verbs[i]} "

    return sentence.strip() + "."


def generate_mixed_sentences(verb_data):
    """
    Generate 30 sentences per complexity level, randomly mixing domains.

    Creates a comprehensive dataset of implausible sentences across
    different complexity levels (1-6) and domains (animals, people, vehicles).
    Each complexity level gets 30 sentences with random domain selection.

    Args:
        verb_data (dict): Complete verb data structure containing all domains

    Returns:
        dict: Nested dictionary with structure:
              {complexity_level: [{'sentence': str, 'domain': str}, ...]}
    """
    all_sentences = {}

    domains = ["animals", "people", "vehicles"]

    # Filter out domains that don't exist in the data
    available_domains = [d for d in domains if d in verb_data]

    if not available_domains:
        print("Error: No valid domains found in verb data")
        return {}

    print(f"Available domains: {available_domains}")

    for complexity in range(1, 7):  # Complexity 1-6
        print(f"Generating complexity {complexity} sentences...")
        used_combinations = set()
        sentences = []

        for sentence_num in range(30):  # Generate 30 sentences per complexity
            try:
                # Randomly select a domain
                domain_name = random.choice(available_domains)
                domain_data = verb_data[domain_name]

                sentence, used_domain = generate_implausible_sentence(
                    complexity, domain_name, domain_data, used_combinations
                )
                sentences.append({
                    "sentence": sentence,
                    "domain": used_domain
                })

            except Exception as e:
                print(f"    Error generating sentence {sentence_num + 1}: {e}")
                # Try a different domain
                for fallback_domain in available_domains:
                    if fallback_domain != domain_name:
                        try:
                            domain_data = verb_data[fallback_domain]
                            sentence, used_domain = generate_implausible_sentence(
                                complexity, fallback_domain, domain_data, used_combinations
                            )
                            sentences.append({
                                "sentence": sentence,
                                "domain": used_domain
                            })
                            break
                        except:
                            continue
                else:
                    # If all domains failed, log the failure
                    print(f"    Could not generate sentence {sentence_num + 1} for complexity {complexity}")
        # Store sentences for this complexity level
        all_sentences[f"complexity_{complexity}"] = sentences
        print(f"    Generated {len(sentences)} sentences")

    return all_sentences


def main():
    """
        Main function to orchestrate the sentence generation process.

        Loads verb data from JSON file, generates mixed implausible sentences
        across all complexity levels, saves results to file, and displays summary
        statistics and sample outputs.
        """
    # Load verb data from file
    print("Loading verb data...")
    verb_data = load_verb_data("filename.json")

    if verb_data is None:
        return

    print("Verb data loaded successfully!")

    # Generate mixed sentences
    print("\nGenerating mixed implausible sentences...")
    try:
        all_sentences = generate_mixed_sentences(verb_data)

        # Save to file
        output_filename = "output_file"
        with open(output_filename, "w") as f:
            json.dump(all_sentences, f, indent=2)

        print(f"\nSentences saved to '{output_filename}'")

        # Display sample results
        print("\nSample Mixed Implausible Sentences:")
        print("=" * 50)

        for complexity in range(1, 4):  # Show first 3 complexities
            complexity_key = f"complexity_{complexity}"
            if complexity_key in all_sentences:
                print(f"\nComplexity {complexity}:")
                sentences = all_sentences[complexity_key]
                for i, item in enumerate(sentences[:5]):  # Show first 5
                    print(f"  {i + 1}. [{item['domain']}] {item['sentence']}")

        # Print domain distribution summary
        print("\n\nDomain Distribution Summary:")
        print("=" * 30)
        for complexity in range(1, 7):
            complexity_key = f"complexity_{complexity}"
            if complexity_key in all_sentences:
                domain_count = {}
                for item in all_sentences[complexity_key]:
                    domain = item['domain']
                    domain_count[domain] = domain_count.get(domain, 0) + 1

                print(f"Complexity {complexity}: {domain_count}")

        # Print total summary
        total_sentences = sum(len(sentences) for sentences in all_sentences.values())
        print(f"\nTotal sentences generated: {total_sentences}")

    except Exception as e:
        print(f"Error generating sentences: {e}")


if __name__ == "__main__":
    main()