"""
Center-Embedded Question Generator (Implausible)

This script generates comprehension questions for center-embedded sentences in the implausible subset.
The generator creates questions of varying difficulty levels (easy, medium, hard)
for each entity in the sentence, with special attention to middle entities.

The script it optimized to handle the nouns and verbs appearing in the implausible subset.
"""
import json
import re
from typing import List, Dict, Tuple, Set, Optional


class CenterEmbeddedQuestionGenerator:
    """
        A comprehensive question generator for center-embedded sentences for the implausible subset

        This class parses complex nested sentence structures and generates
        comprehension questions for each entity mentioned.

        The generator produces exactly 6 questions per entity:
        - 2 easy questions (direct action/agent identification)
        - 2 medium questions (entity counting, nested dependencies)
        - 2 hard questions (causal sequences, chain consequences)

        Attributes:
            valid_animals (set): Set of recognized animal entities
            valid_people (set): Set of recognized person/profession entities
            valid_vehicles (set): Set of recognized vehicle entities
            all_valid_entities (set): Combined set of all valid entities
            entity_list (list): Entities sorted by length for optimal matching
            irregular_verbs (dict): Mapping of irregular verb forms
            verb_particles (set): Common verb particles for phrasal verbs
            adverbs (set): Common adverbs that modify actions
    """
    def __init__(self):
        """
            Initialize the question generator with entity lists and linguistic rules.
        """
        self.valid_animals = {
            # Birds
            'parrot', 'duck', 'vulture', 'pelican', 'seagull', 'heron', 'bat',
            # Mammals
            'dog', 'cat', 'horse', 'lion', 'cow', 'bear', 'elephant', 'deer', 'pig',
            'giraffe', 'mouse', 'sheep', 'donkey', 'rabbit', 'squirrel', 'coyote',
            'moose', 'beaver', 'camel', 'skunk', 'crab', 'dolphin', 'whale',
            # Reptiles & Amphibians
            'frog', 'lizard', 'cobra',
            # Insects
            'bee', 'spider', 'cockroach', 'butterfly', 'grasshopper', 'worm',
            # Aquatic
            'tuna', 'shark', 'starfish', 'octopus'
        }

        self.valid_people = {
            'doctor', 'lawyer', 'teacher', 'nurse', 'police officer', 'firefighter',
            'surgeon', 'mailman', 'waiter', 'judge', 'photographer', 'principal'
        }

        self.valid_vehicles = {
            'truck', 'motorcycle', 'airplane', 'bicycle', 'boat', 'train',
            'spaceship', 'ambulance'
        }

        # Clear tier2_extensions since we're only using the provided data
        self.tier2_extensions = set()

        # Combine all valid entities
        self.all_valid_entities = (self.valid_animals | self.valid_people |
                                   self.valid_vehicles | self.tier2_extensions)

        # Create sorted list for multi-word matching (longest first)
        self.entity_list = sorted(self.all_valid_entities, key=len, reverse=True)

        # Build comprehensive verb dictionary from the provided data
        self.irregular_verbs = {
            # Birds
            'mimicked': ('mimic', 'mimicked', 'mimicking'),
            'chattered': ('chatter', 'chattered', 'chattering'),
            'paddled toward': ('paddle toward', 'paddled toward', 'paddling toward'),
            'dabbled peacefully': ('dabble peacefully', 'dabbled peacefully', 'dabbling peacefully'),
            'scavenged from': ('scavenge from', 'scavenged from', 'scavenging from'),
            'scavenged': ('scavenge', 'scavenged', 'scavenging'),
            'scooped at': ('scoop at', 'scooped at', 'scooping at'),
            'pouched fish': ('pouch fish', 'pouched fish', 'pouching fish'),
            'dive-bombed': ('dive-bomb', 'dive-bombed', 'dive-bombing'),
            'wheeled above the harbor': (
            'wheel above the harbor', 'wheeled above the harbor', 'wheeling above the harbor'),
            'speared': ('spear', 'speared', 'spearing'),
            'waded': ('wade', 'waded', 'wading'),
            'echolocated': ('echolocate', 'echolocated', 'echolocating'),
            'hung from a branch': ('hang from a branch', 'hung from a branch', 'hanging from a branch'),

            # Mammals
            'barked at': ('bark at', 'barked at', 'barking at'),
            'barked': ('bark', 'barked', 'barking'),
            'wagged his tail': ('wag his tail', 'wagged his tail', 'wagging his tail'),
            'meowed at': ('meow at', 'meowed at', 'meowing at'),
            'purred at': ('purr at', 'purred at', 'purring at'),
            'meowed': ('meow', 'meowed', 'meowing'),
            'purred': ('purr', 'purred', 'purring'),
            'neighed at': ('neigh at', 'neighed at', 'neighing at'),
            'whinnied at': ('whinny at', 'whinnied at', 'whinnying at'),
            'galloped toward': ('gallop toward', 'galloped toward', 'galloping toward'),
            'neighed': ('neigh', 'neighed', 'neighing'),
            'whinnied': ('whinny', 'whinnied', 'whinnying'),
            'galloped': ('gallop', 'galloped', 'galloping'),
            'roared at': ('roar at', 'roared at', 'roaring at'),
            'prowled toward': ('prowl toward', 'prowled toward', 'prowling toward'),
            'roared': ('roar', 'roared', 'roaring'),
            'prowled': ('prowl', 'prowled', 'prowling'),
            'mooed at': ('moo at', 'mooed at', 'mooing at'),
            'lowed at': ('low at', 'lowed at', 'lowing at'),
            'mooed': ('moo', 'mooed', 'mooing'),
            'lowed': ('low', 'lowed', 'lowing'),
            'grazed': ('graze', 'grazed', 'grazing'),
            'growled at': ('growl at', 'growled at', 'growling at'),
            'swiped at': ('swipe at', 'swiped at', 'swiping at'),
            'lumbered toward': ('lumber toward', 'lumbered toward', 'lumbering toward'),
            'lumbered': ('lumber', 'lumbered', 'lumbering'),
            'hibernated': ('hibernate', 'hibernated', 'hibernating'),
            'trumpeted at': ('trumpet at', 'trumpeted at', 'trumpeting at'),
            'charged at': ('charge at', 'charged at', 'charging at'),
            'trumpeted': ('trumpet', 'trumpeted', 'trumpeting'),
            'stomped': ('stomp', 'stomped', 'stomping'),
            'bounded toward': ('bound toward', 'bounded toward', 'bounding toward'),
            'leaped at': ('leap at', 'leaped at', 'leaping at'),
            'bounded': ('bound', 'bounded', 'bounding'),
            'leaped': ('leap', 'leaped', 'leaping'),
            'oinked at': ('oink at', 'oinked at', 'oinking at'),
            'snorted at': ('snort at', 'snorted at', 'snorting at'),
            'rooted at': ('root at', 'rooted at', 'rooting at'),
            'oinked': ('oink', 'oinked', 'oinking'),
            'snorted': ('snort', 'snorted', 'snorting'),
            'rooted into the ground': ('root into the ground', 'rooted into the ground', 'rooting into the ground'),
            'stretched toward': ('stretch toward', 'stretched toward', 'stretching toward'),
            'craned at the treetop': ('crane at the treetop', 'craned at the treetop', 'craning at the treetop'),
            'squeaked at': ('squeak at', 'squeaked at', 'squeaking at'),
            'scurried toward': ('scurry toward', 'scurried toward', 'scurrying toward'),
            'squeaked': ('squeak', 'squeaked', 'squeaking'),
            'scurried': ('scurry', 'scurried', 'scurrying'),
            'bleated at': ('bleat at', 'bleated at', 'bleating at'),
            'baaed at': ('baa at', 'baaed at', 'baaing at'),
            'bleated': ('bleat', 'bleated', 'bleating'),
            'baaed': ('baa', 'baaed', 'baaing'),
            'brayed at': ('bray at', 'brayed at', 'braying at'),
            'kicked': ('kick', 'kicked', 'kicking'),
            'brayed': ('bray', 'brayed', 'braying'),
            'thumped at': ('thump at', 'thumped at', 'thumping at'),
            'hopped toward': ('hop toward', 'hopped toward', 'hopping toward'),
            'thumped': ('thump', 'thumped', 'thumping'),
            'hopped': ('hop', 'hopped', 'hopping'),
            'chittered at': ('chitter at', 'chittered at', 'chittering at'),
            'scampered up the tree': ('scamper up the tree', 'scampered up the tree', 'scampering up the tree'),
            'yipped at': ('yip at', 'yipped at', 'yipping at'),
            'howled at': ('howl at', 'howled at', 'howling at'),
            'yipped': ('yip', 'yipped', 'yipping'),
            'howled': ('howl', 'howled', 'howling'),
            'bellowed at': ('bellow at', 'bellowed at', 'bellowing at'),
            'bellowed': ('bellow', 'bellowed', 'bellowing'),
            'gnawed at': ('gnaw at', 'gnawed at', 'gnawing at'),
            'slapped': ('slap', 'slapped', 'slapping'),
            'spat at': ('spit at', 'spat at', 'spitting at'),
            'humped': ('hump', 'humped', 'humping'),
            'sprayed at': ('spray at', 'sprayed at', 'spraying at'),
            'waddled': ('waddle', 'waddled', 'waddling'),
            'scuttled toward': ('scuttle toward', 'scuttled toward', 'scuttling toward'),
            'scuttled': ('scuttle', 'scuttled', 'scuttling'),
            'clicked at': ('click at', 'clicked at', 'clicking at'),
            'clicked': ('click', 'clicked', 'clicking'),
            'sang at': ('sing at', 'sang at', 'singing at'),
            'spouted at': ('spout at', 'spouted at', 'spouting at'),
            'sang': ('sing', 'sang', 'singing'),
            'spouted': ('spout', 'spouted', 'spouting'),

            # Reptiles & Amphibians
            'croaked at': ('croak at', 'croaked at', 'croaking at'),
            'croaked': ('croak', 'croaked', 'croaking'),
            'basked toward': ('bask toward', 'basked toward', 'basking toward'),
            'darted': ('dart', 'darted', 'darting'),
            'hissed at': ('hiss at', 'hissed at', 'hissing at'),
            'reared at': ('rear at', 'reared at', 'rearing at'),
            'hissed': ('hiss', 'hissed', 'hissing'),
            'reared': ('rear', 'reared', 'rearing'),
            'coiled': ('coil', 'coiled', 'coiling'),

            # Insects
            'buzzed at': ('buzz at', 'buzzed at', 'buzzing at'),
            'stung': ('sting', 'stung', 'stinging'),
            'buzzed': ('buzz', 'buzzed', 'buzzing'),
            'swarmed': ('swarm', 'swarmed', 'swarming'),
            'crawled toward': ('crawl toward', 'crawled toward', 'crawling toward'),
            'spun a web': ('spin a web', 'spun a web', 'spinning a web'),
            'crawled': ('crawl', 'crawled', 'crawling'),
            'fluttered toward': ('flutter toward', 'fluttered toward', 'fluttering toward'),
            'fluttered': ('flutter', 'fluttered', 'fluttering'),
            'chirped at': ('chirp at', 'chirped at', 'chirping at'),
            'chirped': ('chirp', 'chirped', 'chirping'),
            'wriggled toward': ('wriggle toward', 'wriggled toward', 'wriggling toward'),
            'wriggled': ('wriggle', 'wriggled', 'wriggling'),
            'burrowed': ('burrow', 'burrowed', 'burrowing'),

            # Aquatic
            'swam toward': ('swim toward', 'swam toward', 'swimming toward'),
            'schooled with': ('school with', 'schooled with', 'schooling with'),
            'swam': ('swim', 'swam', 'swimming'),
            'jumped': ('jump', 'jumped', 'jumping'),
            'circled around': ('circle around', 'circled around', 'circling around'),
            'attacked': ('attack', 'attacked', 'attacking'),
            'circled': ('circle', 'circled', 'circling'),
            'clung to': ('cling to', 'clung to', 'clinging to'),
            'clung': ('cling', 'clung', 'clinging'),
            'squeezed': ('squeeze', 'squeezed', 'squeezing'),
            'inked at': ('ink at', 'inked at', 'inking at'),
            'inked': ('ink', 'inked', 'inking'),

            # People - Professions
            'prescribed to': ('prescribe to', 'prescribed to', 'prescribing to'),
            'diagnosed': ('diagnose', 'diagnosed', 'diagnosing'),
            'examined with stethoscope': (
            'examine with stethoscope', 'examined with stethoscope', 'examining with stethoscope'),
            'prescribed medicines': ('prescribe medicines', 'prescribed medicines', 'prescribing medicines'),
            'cross-examined': ('cross-examine', 'cross-examined', 'cross-examining'),
            'defended in court': ('defend in court', 'defended in court', 'defending in court'),
            'subpoenaed': ('subpoena', 'subpoenaed', 'subpoenaing'),
            'argued': ('argue', 'argued', 'arguing'),
            'objected to the statement': (
            'object to the statement', 'objected to the statement', 'objecting to the statement'),
            'graded': ('grade', 'graded', 'grading'),
            'lectured to': ('lecture to', 'lectured to', 'lecturing to'),
            'assigned homework to': ('assign homework to', 'assigned homework to', 'assigning homework to'),
            'lectured': ('lecture', 'lectured', 'lecturing'),
            'taught': ('teach', 'taught', 'teaching'),
            'bandaged': ('bandage', 'bandaged', 'bandaging'),
            'administered shots to': ('administer shots to', 'administered shots to', 'administering shots to'),
            'took temperature of': ('take temperature of', 'took temperature of', 'taking temperature of'),
            'triaged': ('triage', 'triaged', 'triaging'),
            'nursed': ('nurse', 'nursed', 'nursing'),
            'arrested': ('arrest', 'arrested', 'arresting'),
            'handcuffed': ('handcuff', 'handcuffed', 'handcuffing'),
            'read rights to': ('read rights to', 'read rights to', 'reading rights to'),
            'patrolled': ('patrol', 'patrolled', 'patrolling'),
            'responded to a 911 call': ('respond to a 911 call', 'responded to a 911 call', 'responding to a 911 call'),
            'rescued': ('rescue', 'rescued', 'rescuing'),
            'carried out': ('carry out', 'carried out', 'carrying out'),
            'extinguished fire': ('extinguish fire', 'extinguished fire', 'extinguishing fire'),
            'operated on': ('operate on', 'operated on', 'operating on'),
            'sutured': ('suture', 'sutured', 'suturing'),
            'incised': ('incise', 'incised', 'incising'),
            'operated': ('operate', 'operated', 'operating'),
            'delivered mail to': ('deliver mail to', 'delivered mail to', 'delivering mail to'),
            'delivered mail': ('deliver mail', 'delivered mail', 'delivering mail'),
            'sorted mail': ('sort mail', 'sorted mail', 'sorting mail'),
            'served': ('serve', 'served', 'serving'),
            'took orders from': ('take orders from', 'took orders from', 'taking orders from'),
            'seated': ('seat', 'seated', 'seating'),
            'served food': ('serve food', 'served food', 'serving food'),
            'waited tables': ('wait tables', 'waited tables', 'waiting tables'),
            'sentenced': ('sentence', 'sentenced', 'sentencing'),
            'gaveled': ('gavel', 'gaveled', 'gaveling'),
            'ruled against': ('rule against', 'ruled against', 'ruling against'),
            'presided': ('preside', 'presided', 'presiding'),
            'photographed': ('photograph', 'photographed', 'photographing'),
            'developed photos of': ('develop photos of', 'developed photos of', 'developing photos of'),
            'shot': ('shoot', 'shot', 'shooting'),
            'expelled': ('expel', 'expelled', 'expelling'),
            'suspended': ('suspend', 'suspended', 'suspending'),
            'disciplined': ('discipline', 'disciplined', 'disciplining'),
            'called an assembly': ('call an assembly', 'called an assembly', 'calling an assembly'),
            'toured the school': ('tour the school', 'toured the school', 'touring the school'),

            # Vehicles
            'jackknifed into': ('jackknife into', 'jackknifed into', 'jackknifing into'),
            'towed': ('tow', 'towed', 'towing'),
            'jackknifed': ('jackknife', 'jackknifed', 'jackknifing'),
            'lane-split past': ('lane-split past', 'lane-split past', 'lane-splitting past'),
            'wheelied at': ('wheelie at', 'wheelied at', 'wheelying at'),
            'wheelied': ('wheelie', 'wheelied', 'wheelying'),
            'air-dropped on': ('air-drop on', 'air-dropped on', 'air-dropping on'),
            'strafed': ('strafe', 'strafed', 'strafing'),
            'taxied': ('taxi', 'taxied', 'taxying'),
            'pedaled past': ('pedal past', 'pedaled past', 'pedaling past'),
            'rang bell at': ('ring bell at', 'rang bell at', 'ringing bell at'),
            'cycled around': ('cycle around', 'cycled around', 'cycling around'),
            'pedaled': ('pedal', 'pedaled', 'pedaling'),
            'sailed past': ('sail past', 'sailed past', 'sailing past'),
            'anchored near': ('anchor near', 'anchored near', 'anchoring near'),
            'moored to': ('moor to', 'moored to', 'mooring to'),
            'sailed': ('sail', 'sailed', 'sailing'),
            'anchored': ('anchor', 'anchored', 'anchoring'),
            'docked': ('dock', 'docked', 'docking'),
            'whistled at': ('whistle at', 'whistled at', 'whistling at'),
            'whistled': ('whistle', 'whistled', 'whistling'),
            'steamed': ('steam', 'steamed', 'steaming'),
            'orbited': ('orbit', 'orbited', 'orbiting'),
            'launched': ('launch', 'launched', 'launching'),
            'blasted off into the sky': (
            'blast off into the sky', 'blasted off into the sky', 'blasting off into the sky'),
            'sirened at': ('siren at', 'sirened at', 'sirening at'),
            'sirened': ('siren', 'sirened', 'sirening'),
        }

        # Update verb particles to include common particles from the new verbs
        self.verb_particles = {
            'away', 'off', 'up', 'down', 'in', 'out', 'on', 'over', 'back',
            'around', 'through', 'along', 'across', 'by', 'forward', 'ahead',
            'aside', 'together', 'apart', 'at', 'towards', 'toward', 'into', 'onto',
            'past', 'beyond', 'under', 'beneath', 'above', 'below', 'behind',
            'beside', 'near', 'against', 'underneath', 'inside', 'outside', 'from',
            'with', 'to'
        }

        # Common adverbs that aren't actions
        self.adverbs = {
            'anxiously', 'quickly', 'slowly', 'carefully', 'loudly', 'quietly',
            'suddenly', 'gradually', 'immediately', 'eventually', 'finally',
            'desperately', 'frantically', 'calmly', 'nervously', 'happily',
            'sadly', 'angrily', 'peacefully', 'aggressively', 'gently'
        }

    def _identify_entities_in_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Identify and locate entities within the given text.

        Args:
            text (str): The input text to analyze

        Returns:
            List[Tuple[str, int, int]]: List of (entity, start_pos, end_pos) tuples
            sorted by position in text, with overlapping entities resolved
        """
        text_lower = text.lower()
        entities_found = []

        # Sort entities by length (longest first) to match longer entities first
        for entity in self.entity_list:
            entity_lower = entity.lower()
            start = 0
            while True:
                pos = text_lower.find(entity_lower, start)
                if pos == -1:
                    break

                # Check word boundaries
                if pos > 0 and text_lower[pos - 1].isalnum():
                    start = pos + 1
                    continue

                end_pos = pos + len(entity_lower)
                if end_pos < len(text_lower) and text_lower[end_pos].isalnum():
                    start = pos + 1
                    continue

                # Check if this overlaps with existing entities
                overlaps = False
                for existing_entity, existing_start, existing_end in entities_found:
                    if not (end_pos <= existing_start or pos >= existing_end):
                        overlaps = True
                        break

                if not overlaps:
                    entities_found.append((entity, pos, end_pos))

                start = pos + 1

        # Sort by position in text
        entities_found.sort(key=lambda x: x[1])
        return entities_found

    def _get_verb_forms(self, verb_text: str) -> Tuple[str, str, str]:
        """
        Get base, past participle, and gerund forms of a verb.

        Args:
            verb_text (str): The verb in any form, potentially with particles

        Returns:
            Tuple[str, str, str]: (base_form, past_participle, gerund)
        """
        # Clean up the verb text
        verb_text = verb_text.strip()

        # Check if it's in our irregular verbs dictionary
        if verb_text in self.irregular_verbs:
            return self.irregular_verbs[verb_text]

        # If not found, try to handle it as a regular verb
        # For multi-word verbs, work with the first word
        words = verb_text.split()
        main_verb = words[0]
        particles = words[1:] if len(words) > 1 else []

        # Handle the main verb
        if main_verb in self.irregular_verbs:
            base, participle, gerund = self.irregular_verbs[main_verb]
        else:
            base, participle, gerund = self._handle_regular_verb(main_verb)

        # Add particles back
        if particles:
            particle_str = ' '.join(particles)
            base = f"{base} {particle_str}"
            participle = f"{participle} {particle_str}"
            gerund = f"{gerund} {particle_str}"

        return base, participle, gerund

    def _handle_regular_verb(self, verb: str) -> Tuple[str, str, str]:
        """
        Handle conjugation patterns for regular verbs.

        Args:
            verb (str): A regular verb in any form

        Returns:
            Tuple[str, str, str]: (base_form, past_participle, gerund)
        """
        if verb.endswith('ed'):
            # Handle different regular patterns
            if verb.endswith('ied'):
                # studied -> study
                base = verb[:-3] + 'y'
                participle = verb
                gerund = base[:-1] + 'ying'
            elif verb.endswith('eed'):
                # agreed -> agree
                base = verb[:-1]
                participle = verb
                gerund = base + 'ing'
            elif len(verb) > 4 and verb[-4] == verb[-3] and verb[-4] not in 'aeiou':
                # stopped -> stop (doubled consonant)
                base = verb[:-3]
                participle = verb
                gerund = base + 'ping'
            else:
                # General -ed removal
                if verb.endswith('sed'):
                    base = verb[:-1]  # chased -> chase
                elif verb.endswith('ded'):
                    base = verb[:-2]  # added -> add
                elif verb.endswith('ted'):
                    if len(verb) > 5 and verb[-4] == verb[-3]:
                        base = verb[:-3]  # spotted -> spot
                    else:
                        base = verb[:-2]  # wanted -> want
                else:
                    base = verb[:-2]

                participle = verb

                # Handle gerund formation
                if base.endswith('e') and not base.endswith('ee'):
                    gerund = base[:-1] + 'ing'
                elif len(base) > 2 and base[-1] not in 'aeiouyw' and base[-2] in 'aeiou':
                    gerund = base + base[-1] + 'ing'
                else:
                    gerund = base + 'ing'
        else:
            # Assume it's already base form
            base = verb
            participle = verb + 'ed'
            gerund = verb + 'ing'

        return base, participle, gerund

    def _parse_center_embedded_sentence(self, sentence: str, complexity_level: int) -> Tuple[List[Dict], str]:
        """
        Parse a center-embedded sentence into its structural components.

        Identifies entities, determines the middle entity, and extracts the
        grammatical relationships between entities and their actions.

        Args:
            sentence (str): The center-embedded sentence to parse
            complexity_level (int): Expected nesting level of the sentence

        Returns:
            Tuple[List[Dict], str]: (sentence_structure, middle_entity)
        """
        # Remove period and clean
        sentence = sentence.rstrip('.')

        # Find entities in the sentence
        entities_found = self._identify_entities_in_text(sentence)

        if not entities_found:
            return [], ""

        # Get the order of entities as they appear
        entities_in_order = [entity for entity, _, _ in entities_found]

        # Calculate middle entity using proper median
        middle_entity = self._calculate_middle_entity(entities_in_order)

        # Parse the sentence structure
        structure = self._parse_sentence_structure(sentence, entities_in_order, complexity_level)

        return structure, middle_entity

    def _calculate_middle_entity(self, entities: List[str]) -> str:
        """
        Calculate the middle entity using median logic.

        For odd-length lists, returns the center element. For even-length lists,
        returns the left-center element.

        Args:
            entities (List[str]): List of entities in order of appearance

        Returns:
            str: The middle entity identifier
        """
        if not entities:
            return ""

        n = len(entities)
        if n % 2 == 1:
            # Odd number: return middle element
            return entities[n // 2]
        else:
            # Even number: return left-center element
            return entities[n // 2 - 1]

    def _parse_sentence_structure(self, sentence: str, entities: List[str], complexity_level: int) -> List[Dict]:
        """
        Parse sentence structure to extract subject-action-object relationships.

        Identifies "that" clause boundaries and maps actions to entity pairs,
        handling both transitive and intransitive verb constructions.

        Args:
            sentence (str): The complete sentence
            entities (List[str]): Ordered list of entities found
            complexity_level (int): Expected number of embedded clauses

        Returns:
            List[Dict]: List of relationship dictionaries with 'subject', 'action', 'object'
        """
        words = sentence.split()

        # Find "that" positions
        that_positions = [i for i, word in enumerate(words) if word == "that"]

        if len(that_positions) < complexity_level:
            return []

        # Find where the verb section starts
        last_that_pos = that_positions[-1]
        verb_section_start = last_that_pos + 1

        # Skip "the" if it follows
        if (verb_section_start < len(words) and
                words[verb_section_start] == "the"):
            verb_section_start += 1

        # Skip the last entity
        if verb_section_start < len(words):
            # Find the end of the last entity
            last_entity = entities[-1]
            last_entity_words = last_entity.split()
            verb_section_start += len(last_entity_words)

        # Extract verb section
        if verb_section_start >= len(words):
            return []

        verb_words = words[verb_section_start:]

        # Parse actions from verb section
        actions = self._extract_actions_from_verb_section(verb_words, complexity_level)

        # Build structure
        structure = []

        # First entity performs the intransitive action (last action)
        if actions:
            intransitive_action = actions[-1]
            structure.append({
                "subject": entities[0],
                "action": intransitive_action,
                "object": None
            })

        # Map transitive actions (in reverse order) to entity pairs
        transitive_actions = actions[:-1] if len(actions) > 1 else []

        for i, action in enumerate(reversed(transitive_actions)):
            if i + 1 < len(entities):
                structure.append({
                    "subject": entities[i + 1],
                    "action": action,
                    "object": entities[i]
                })

        return structure

    def _extract_actions_from_verb_section(self, verb_words: List[str], complexity_level: int) -> List[str]:
        """
        Extract action verbs from the verb section of the sentence.

        Handles phrasal verbs, prepositional phrases, and complex verb constructions
        while avoiding non-action words.

        Args:
            verb_words (List[str]): Words from the verb section
            complexity_level (int): Expected number of actions to find

        Returns:
            List[str]: Extracted action phrases
        """
        actions = []
        i = 0

        # Join all words to search for multi-word verbs
        verb_text = ' '.join(verb_words)

        while i < len(verb_words):
            action, consumed = self._extract_single_action(verb_words, i, verb_text)
            if action:
                actions.append(action)
                i += consumed
            else:
                i += 1

        return actions

    def _extract_single_action(self, words: List[str], start: int, full_text: str = None) -> Tuple[str, int]:

        """
        Extract a single action phrase starting at the given position.

        Recognizes various verb patterns including phrasal verbs, "tried to" constructions,
        and verbs with spatial/directional particles.

        Args:
            words (List[str]): List of words to analyze
            start (int): Starting position in the word list

        Returns:
            Tuple[str, int]: (extracted_action, words_consumed)
        """
        if start >= len(words):
            return "", 0

        # Get remaining text from this position
        remaining_text = ' '.join(words[start:])

        # Sort verb phrases by length (longest first) to match longest phrases first
        verb_phrases = sorted(self.irregular_verbs.keys(), key=len, reverse=True)

        # Look for multi-word verbs first
        for verb_phrase in verb_phrases:
            if remaining_text.startswith(verb_phrase):
                # Check if this is a complete word match (not partial)
                end_pos = len(verb_phrase)
                if (end_pos == len(remaining_text) or
                        remaining_text[end_pos].isspace() or
                        remaining_text[end_pos] in '.,!?'):
                    # Count how many words this verb phrase consumes
                    words_consumed = len(verb_phrase.split())
                    return verb_phrase, words_consumed

        # If no multi-word verb found, try single word
        word = words[start]

        # Skip articles and determiners
        if word in ['a', 'an', 'the']:
            return "", 1

        # Skip pure adverbs when they're standalone
        if word in self.adverbs:
            return "", 1

        # Check if single word is a known verb
        if word in self.irregular_verbs:
            return word, 1

        # Pattern for phrasal verbs (verb + particle)
        if start + 1 < len(words):
            next_word = words[start + 1]
            if next_word in self.verb_particles:
                phrasal_verb = f"{word} {next_word}"
                if phrasal_verb in self.irregular_verbs:
                    return phrasal_verb, 2

        # If it looks like a verb but isn't in our dictionary, return it anyway
        if word not in ['a', 'an', 'the'] and word not in self.adverbs:
            return word, 1

        return "", 1



    def _describe_causal_chain(self, causal_chain: List[Dict]) -> str:
        """
            Generate a natural language description of a causal chain.

            Converts the causal relationship structure into grammatically correct
            English using appropriate verb forms and connecting phrases.

            Args:
                causal_chain (List[Dict]): List of causal relationships

            Returns:
                str: Natural language description of the causal sequence
        """
        if not causal_chain:
            return "no prior events"

        # Reverse the chain to get chronological order (earliest first)
        chronological_chain = list(reversed(causal_chain))

        descriptions = []
        for i, relation in enumerate(chronological_chain):
            if i == 0:
                # First event uses gerund form
                _, _, gerund = self._get_verb_forms(relation['action'])
                descriptions.append(f"the {relation['subject']} {gerund} the {relation['object']}")
            else:
                # Subsequent events use gerund form after "which led to"
                _, _, gerund = self._get_verb_forms(relation['action'])
                descriptions.append(f"the {relation['subject']} {gerund} the {relation['object']}")

        return " which led to ".join(descriptions)


    def _generate_questions_for_entity(self, entity: str, structure: List[Dict],
                                       middle_entity: str, complexity_level: int) -> List[Dict]:
        """
        Generate exactly 6 comprehension questions for a specific entity.

        Creates 2 questions each at easy, medium, and hard difficulty levels,
        covering different aspects of sentence comprehension and reasoning.

        Args:
            entity (str): The entity to generate questions for
            structure (List[Dict]): Parsed sentence structure
            middle_entity (str): The identified middle entity
            complexity_level (int): Sentence complexity level

        Returns:
            List[Dict]: List of 6 question dictionaries with metadata
        """
        questions = []

        # Find relations involving this entity
        as_subject = [r for r in structure if r['subject'] == entity]
        as_object = [r for r in structure if r['object'] == entity]

        # Get all entities for counting
        all_entities = set()
        for relation in structure:
            all_entities.add(relation['subject'])
            if relation['object']:
                all_entities.add(relation['object'])

        entity_count = len(all_entities)
        is_middle_entity = (entity == middle_entity)

        # EASY QUESTIONS (2) - Keep existing logic

        # Question 1: What did the entity do?
        if as_subject:
            relation = as_subject[0]
            base_verb, _, _ = self._get_verb_forms(relation['action'])

            if relation['object'] is None:
                # Intransitive: "What did the dog do?"
                questions.append({
                    'question': f"What did the {entity} do?",
                    'answer': base_verb,
                    'type': 'action_performed',
                    'difficulty': 'easy',
                    'entity': entity,
                    'is_middle_entity': is_middle_entity
                })
            else:
                # Transitive: "What did the wolf do?"
                questions.append({
                    'question': f"What did the {entity} do?",
                    'answer': f"{base_verb} the {relation['object']}",
                    'type': 'action_performed',
                    'difficulty': 'easy',
                    'entity': entity,
                    'is_middle_entity': is_middle_entity
                })
        else:
            # Entity only receives action
            questions.append({
                'question': f"What happened to the {entity}?",
                'answer': f"was {as_object[0]['action']}" if as_object else "nothing happened",
                'type': 'action_performed',
                'difficulty': 'easy',
                'entity': entity,
                'is_middle_entity': is_middle_entity
            })

        # Question 2: What/Who acted on the entity?
        if as_object:
            relation = as_object[0]
            questions.append({
                'question': f"What {relation['action']} the {entity}?",
                'answer': f"the {relation['subject']}",
                'type': 'agent_identification',
                'difficulty': 'easy',
                'entity': entity,
                'is_middle_entity': is_middle_entity
            })
        else:
            # Entity acts but nothing acts on it
            if as_subject and as_subject[0]['object']:
                questions.append({
                    'question': f"What was affected by the {entity}?",
                    'answer': f"the {as_subject[0]['object']}",
                    'type': 'agent_identification',
                    'difficulty': 'easy',
                    'entity': entity,
                    'is_middle_entity': is_middle_entity
                })
            else:
                questions.append({
                    'question': f"What acted on the {entity}?",
                    'answer': "nothing",
                    'type': 'agent_identification',
                    'difficulty': 'easy',
                    'entity': entity,
                    'is_middle_entity': is_middle_entity
                })

        # MEDIUM QUESTIONS (2)

        # Type 3: Entity Count (NEW - replaces role_disambiguation)
        questions.append({
            'question': f"How many distinct entities are in the sentence?",
            'answer': str(entity_count),
            'type': 'entity_count',
            'difficulty': 'medium',
            'entity': entity,
            'is_middle_entity': is_middle_entity
        })

        # Type 4: Nested Dependency (UNCHANGED)
        if as_object:
            relation = as_object[0]
            # Find what this entity did after being acted upon
            entity_actions = [r for r in structure if r['subject'] == entity]
            if entity_actions:
                verb_parts = relation['action'].split()
                main_verb = verb_parts[0]
                _, participle, _ = self._get_verb_forms(main_verb)

                if len(verb_parts) > 1:
                    participle_phrase = participle + ' ' + ' '.join(verb_parts[1:])
                else:
                    participle_phrase = participle

                target_action = entity_actions[0]
                if target_action['object']:
                    answer = f"{target_action['action']} the {target_action['object']}"
                else:
                    answer = target_action['action']

                questions.append({
                    'question': f"What did the entity that was {participle_phrase} do?",
                    'answer': answer,
                    'type': 'nested_dependency',
                    'difficulty': 'medium',
                    'entity': entity,
                    'is_middle_entity': is_middle_entity
                })
            else:
                # Entity was acted upon but didn't do anything
                verb_parts = relation['action'].split()
                main_verb = verb_parts[0]
                _, participle, _ = self._get_verb_forms(main_verb)

                questions.append({
                    'question': f"What happened after the {entity} was {participle}?",
                    'answer': "no subsequent action by this entity",
                    'type': 'nested_dependency',
                    'difficulty': 'medium',
                    'entity': entity,
                    'is_middle_entity': is_middle_entity
                })
        else:
            # Fallback for subjects only
            if as_subject and as_subject[0]['object']:
                obj = as_subject[0]['object']
                obj_actions = [r for r in structure if r['subject'] == obj]
                if obj_actions:
                    obj_action = obj_actions[0]
                    if obj_action['object']:
                        answer = f"{obj_action['action']} the {obj_action['object']}"
                    else:
                        answer = obj_action['action']

                    questions.append({
                        'question': f"What did the entity acted upon by the {entity} do?",
                        'answer': answer,
                        'type': 'nested_dependency',
                        'difficulty': 'medium',
                        'entity': entity,
                        'is_middle_entity': is_middle_entity
                    })
                else:
                    questions.append({
                        'question': f"What is the dependency relationship involving the {entity}?",
                        'answer': f"the {entity} acts on the {obj}",
                        'type': 'nested_dependency',
                        'difficulty': 'medium',
                        'entity': entity,
                        'is_middle_entity': is_middle_entity
                    })
            else:
                questions.append({
                    'question': f"What dependencies involve the {entity}?",
                    'answer': f"the {entity} is at the root of the dependency chain",
                    'type': 'nested_dependency',
                    'difficulty': 'medium',
                    'entity': entity,
                    'is_middle_entity': is_middle_entity
                })

        # HARD QUESTIONS (2) - Keep existing logic

        # Question 5: Causal sequence - FIXED
        causal_chain = self._build_causal_chain(entity, structure)
        chain_desc = self._describe_causal_chain(causal_chain)

        questions.append({
            'question': f"What series of events led to the {entity}'s action?",
            'answer': chain_desc,
            'type': 'causal_sequence',
            'difficulty': 'hard',
            'entity': entity,
            'is_middle_entity': is_middle_entity
        })

        # Question 6: Chain consequence
        consequence = self._find_chain_consequence(entity, structure)
        questions.append({
            'question': f"What is the consequence of the {entity}'s involvement?",
            'answer': consequence,
            'type': 'chain_consequence',
            'difficulty': 'hard',
            'entity': entity,
            'is_middle_entity': is_middle_entity
        })

        return questions



    def _build_causal_chain(self, entity: str, structure: List[Dict]) -> List[Dict]:
        """
        Build the causal chain of events leading to an entity's action.

        Traces the sequence of transitive relationships that precede or involve
        the specified entity's participation in the sentence.

        Args:
            entity (str): The entity to trace causality for
            structure (List[Dict]): Complete sentence structure

        Returns:
            List[Dict]: Ordered list of causal relationships
        """

        # Get all transitive relations
        transitive_relations = [r for r in structure if r['object'] is not None]

        if not transitive_relations:
            return []

        # Find where this entity appears as a SUBJECT performing a transitive action
        entity_subject_position = None
        for i, relation in enumerate(transitive_relations):
            if relation['subject'] == entity:
                entity_subject_position = i
                break

        # Find where this entity appears as an OBJECT being acted upon
        entity_object_position = None
        for i, relation in enumerate(transitive_relations):
            if relation['object'] == entity:
                entity_object_position = i
                break

        # Determine which position to use for finding prior events
        if entity_subject_position is not None:
            # Entity performs a transitive action - use that position
            reference_position = entity_subject_position
            # Get all relations that come after this reference position
            prior_events = transitive_relations[reference_position + 1:]
        elif entity_object_position is not None:
            # Entity only receives action - include that action in the chain
            reference_position = entity_object_position
            # Get all relations from this position onwards (including the action on this entity)
            prior_events = transitive_relations[reference_position:]
        else:
            # Entity has no transitive involvement
            return []

        return prior_events




    def _find_chain_consequence(self, entity: str, structure: List[Dict]) -> str:
        """
        Identify the consequences of an entity's involvement in the action chain.

        Determines what happens as a result of the entity's actions or
        what the entity's participation leads to in the overall sequence.

        Args:
            entity (str): The entity to analyze consequences for
            structure (List[Dict]): Complete sentence structure

        Returns:
            str: Description of the consequences
        """
        as_subject = [r for r in structure if r['subject'] == entity]

        if as_subject:
            relation = as_subject[0]
            if relation['object']:
                # Find what happens to the object
                object_actions = [r for r in structure if r['subject'] == relation['object']]
                if object_actions:
                    obj_action = object_actions[0]
                    if obj_action['object']:
                        return f"the {relation['object']} {obj_action['action']} the {obj_action['object']}"
                    else:
                        return f"the {relation['object']} {obj_action['action']}"
                else:
                    return f"the {relation['object']} is affected"
            else:
                return "the action sequence completes"
        else:
            return "no direct consequence"

    def process_sentences(self, data: Dict) -> Dict:
        """
        Process all sentences in the dataset and generate comprehensive questions.

        Main processing method that handles multiple complexity levels,
        parses each sentence, and generates questions for all entities.

        Args:
            data (Dict): Input data organized by complexity levels

        Returns:
            Dict: Complete results with questions, statistics, and metadata
        """
        results = {}

        for complexity_level, level_data in data.items():
            if complexity_level.startswith("complexity_"):
                level_num = int(complexity_level.split("_")[1])
                results[complexity_level] = {
                    "sentences": [],
                    "total_questions": 0,
                    "entities_processed": 0,
                    "middle_entities_flagged": 0
                }

                print(f"\nProcessing {complexity_level}...")

                for i, sentence in enumerate(level_data["sentences"]):
                    # Parse sentence
                    structure, middle_entity = self._parse_center_embedded_sentence(sentence, level_num)

                    if not structure:
                        print(f"  Warning: Could not parse sentence {i + 1}: {sentence}")
                        continue

                    # Get all unique entities
                    all_entities = set()
                    for relation in structure:
                        all_entities.add(relation['subject'])
                        if relation['object']:
                            all_entities.add(relation['object'])

                    # Generate questions for each entity
                    entity_questions = {}
                    middle_entity_count = 0

                    for entity in all_entities:
                        questions = self._generate_questions_for_entity(entity, structure, middle_entity, level_num)
                        entity_questions[entity] = questions

                        # Count questions and middle entities
                        results[complexity_level]["total_questions"] += len(questions)
                        results[complexity_level]["entities_processed"] += 1

                        if entity == middle_entity:
                            middle_entity_count += 1

                    results[complexity_level]["middle_entities_flagged"] += middle_entity_count

                    # Store result
                    results[complexity_level]["sentences"].append({
                        "sentence": sentence,
                        "structure": structure,
                        "middle_entity": middle_entity,
                        "all_entities": list(all_entities),
                        "questions_by_entity": entity_questions,
                        "total_questions": sum(len(q) for q in entity_questions.values())
                    })

                    if (i + 1) % 10 == 0:
                        print(f"  Processed {i + 1}/{len(level_data['sentences'])} sentences")

        return results

    def save_results(self, results: Dict, output_filename: str = "improved_center_embedding_questions.json"):
        """
        Save processing results to a JSON file.

        Args:
            results (Dict): Complete processing results
            output_filename (str): Output file path
        """
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_filename}")

    def print_sample_questions(self, results: Dict, samples_per_level: int = 2):
        """
        Print sample questions for verification and review purposes.

        Args:
            results (Dict): Processing results to sample from
            samples_per_level (int): Number of sample sentences per complexity level
        """
        print("\n" + "=" * 80)
        print("SAMPLE QUESTIONS - IMPROVED VERSION")
        print("=" * 80)

        for complexity_level, level_results in results.items():
            if complexity_level.startswith("complexity_"):
                print(f"\n{complexity_level.upper()}:")

                for i, sentence_data in enumerate(level_results["sentences"][:samples_per_level]):
                    print(f"\nSentence: {sentence_data['sentence']}")
                    print(f"Middle Entity: {sentence_data['middle_entity']}")
                    print(f"Structure:")
                    for rel in sentence_data['structure']:
                        if rel['object']:
                            print(f"  - {rel['subject']} {rel['action']} {rel['object']}")
                        else:
                            print(f"  - {rel['subject']} {rel['action']} (intransitive)")

                    # Show questions by entity
                    for entity, questions in sentence_data['questions_by_entity'].items():
                        print(f"\n  Entity: {entity}")

                        if questions and questions[0].get('is_middle_entity', False):
                            print(f"    *** MIDDLE ENTITY ***")

                        for q in questions:
                            print(f"    {q['difficulty'].upper()} ({q['type']}): {q['question']}")
                            print(f"      Answer: {q['answer']}")


def main():
    """
        Main execution function for the center-embedded question generator.

        Loads input data, processes sentences through the improved generator,
        saves results, and displays summary statistics and sample questions.
    """
    input_filename = "input_filename.json"
    output_filename = "output_filename.json"

    try:
        with open(input_filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {input_filename}")
        return

    # Initialize generator
    generator = CenterEmbeddedQuestionGenerator()

    print("Processing center-embedded sentences...")
    print("- Enhanced entity recognition")
    print("- Fixed grammar issues")
    print("- Corrected middle entity calculation")
    print("- Improved question differentiation")
    print("- Fixed causal chain logic")

    # Process sentences
    results = generator.process_sentences(data)

    # Save results
    generator.save_results(results, output_filename)

    # Print summary
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)

    total_sentences = 0
    total_questions = 0
    total_entities = 0
    total_middle_entities = 0

    for complexity_level, level_results in results.items():
        if complexity_level.startswith("complexity_"):
            sentences = len(level_results['sentences'])
            questions = level_results['total_questions']
            entities = level_results['entities_processed']
            middle_entities = level_results['middle_entities_flagged']

            total_sentences += sentences
            total_questions += questions
            total_entities += entities
            total_middle_entities += middle_entities

            print(f"{complexity_level}: {sentences} sentences, "
                  f"{questions} questions, {entities} entities, "
                  f"{middle_entities} middle entities")

    print(f"\nTOTAL: {total_sentences} sentences, {total_questions} questions, "
          f"{total_entities} entities, {total_middle_entities} middle entities")
    print(f"Average questions per entity: {total_questions / total_entities:.1f}")

    # Print samples
    generator.print_sample_questions(results, samples_per_level=2)


if __name__ == "__main__":
    main()