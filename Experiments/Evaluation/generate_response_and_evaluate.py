"""
This script calls models using the Together AI, Gemini/Google Vertex AI, DeepSeek, and Anthropic platforms.
It generates responses to questions about center-embedding sentences using zero-shot prompting with a predefined system prompt.
It evaluates the responses against a set of predefined questions and answers,
Then normalizes and compares them to determine correctness.
To generate the responses.  GEMINI_API_KEY, DEEPSEEK_API_KEY, TOGETHER_API_KEY, ANTHROPIC_API_KEY
should to be set in the environment variables or in a .env file.
Vertx AI project parameters: project="project_name",location="project_location"
Must install google gcloud cli and authenticate with `gcloud auth login`
"""

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
from google import genai
from google.genai import types
from tqdm import tqdm  # Progress bar

# Environment variable loading
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda *_, **__: None

# Together AI Python SDK
try:
    from together import Together
except ImportError as exc:
    raise SystemExit("Please install the Together AI SDK:  pip install together-ai") from exc

# Anthropic SDK
try:
    from anthropic import Anthropic
    from anthropic._exceptions import OverloadedError
except ImportError as exc:
    Anthropic = None
    OverloadedError = None
    raise SystemExit("Please install the anthropic SDK:  pip install anthropic") from exc

# DeepSeek SDK (using OpenAI SDK)
try:
    from openai import OpenAI
except ImportError as exc:
    OpenAI = None
    raise SystemExit("Please install the OpenAI SDK:  pip install openai") from exc

# For lemmatisation
try:
    import spacy
except ImportError as exc:
    raise SystemExit("Please install spaCy:  pip install spacy && python -m spacy download en_core_web_sm") from exc

# For Semantic Fallback
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError as exc:
    SentenceTransformer = None
    util = None
    raise SystemExit("Please install the sentence-transformers library:  pip install sentence-transformers") from exc


### CONSTANTS & HELPERS
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"


# Verb tense variations
answer_variations = {
    "freeze": ["froze", "frozen", "freezing", "freezes"],
    "flee": ["fled", "fleeing", "flees"],
    "spot": ["spotted", "spotting", "spots"],
    "see": ["saw", "seen", "seeing", "sees"],
    "catch": ["caught", "catching", "catches"],
    "hide": ["hid", "hiding", "hides"],
    "swim": ["swam", "swiming", "swims"],
    "pounce": ["pounced", "pouncing", "pounces"],
    "cross-examine": ["cross-examined", "cross-examines", "cross-examining"],
    "crawl": ["crawled", "crawls", "crawling"],
    "baa": ["baaed", "baaes", "baaing"],
    "gavel": ["gavelled", "gavels", "gavelling"],
    "neigh": ["neighed", "neighs", "neighing"],
    "yip": ["yipped", "yips", "yipping"],
    "wheelie": ["wheelied", "wheelies", "wheelying"],
    "snort": ["snorted", "snorts", "snorting"],
    "croak": ["croaked", "croaks", "croaking"],
    "sting": ["stung", "stings", "stinging"],
    "siren": ["sirened", "sirens", "sirening"],
    "anchor": ["anchored", "anchors", "anchoring"],
    "whistle": ["whistled", "whistles", "whistling"],
    "pedal": ["pedaled", "pedals", "pedaling"],
    "trumpet": ["trumpeted", "trumpets", "trumpeting"],
    "flutter": ["fluttered", "flutters", "fluttering"],
    "strafe": ["strafed", "strafes", "strafing"],
    "moor": ["moored", "moors", "mooring"],
    "photograph": ["photographed", "photographs", "photographing"],
    "shoot": ["shot", "shoots", "shooting"],
    "orbit": ["orbited", "orbits", "orbiting"],
    "chatter": ["chattered", "chatters", "chattering"],
    "school": ["schooled", "schools", "schooling"],
    "jackknife": ["jackknifed", "jackknifes", "jackknifing"],
    "bask": ["basked", "basks", "basking"],
    "rear": ["reared", "rears", "rearing"],
    "meow": ["meowed", "meows", "meowing"],
    "gnaw": ["gnawed", "gnaws", "gnawing"],
    "lane-split": ["lane-split", "lane-splits", "lane-splitting"],
    "subpoena": ["subpoenaed", "subpoenas", "subpoenaing"],
    "handcuff": ["handcuffed", "handcuffs", "handcuffing"],
    "prescribe": ["prescribed", "prescribes", "prescribing"],
    "tow": ["towed", "tows", "towing"],
    "blare": ["blared", "blares", "blaring"],
    "blast": ["blasted", "blasts", "blasting"],
    "air-drop": ["air-dropped", "air-drops", "air-dropping"],
    "steam": ["steamed", "steams", "steaming"],
    "seat": ["seated", "seats", "seating"],
    "discipline": ["disciplined", "disciplines", "disciplining"],
    "ring": ["rang", "rings", "ringing"],
    
    # Common determiners and articles to ignore
    "articles": ["the", "a", "an"],
    
    # Equivalent phrases
    "no prior events": ["none", "nothing", "no events", "nothing happened before"],
    "the action sequence completes": ["sequence completes", "completion", "end of sequence"],
    "none": ["nothing", "no consequences", "end of sequence"]
}

ZERO_SHOT_SYSTEM = (
    "You are a precise question-answering assistant tasked to answer questions on center-embedding sentences.\n"
    "The following are the rules you have to follow to answer the questions you will encounter.\n"
    "• For *action_performed*, *nested_dependency*, and *causal_sequence* question types,"
    " respond using **only** exact word forms that appear in the provided sentence; do not substitute synonyms or paraphrases.\n"
    "• For *agent_identification* questions, respond **only** the exact agent entity from the provided sentence,"
    " do not attach any verbs or verb phrases to the entity.\n"
    "• For *action_performed* questions, respond **only** the exact 'verb' or verb + object entity' phrase using the identical wording found in the provided sentence,"
    " do not replace the object entity related to the verbs into pronouns (e.g. 'it').\n"
    "• For *entity_count* questions output a numeric answer only (e.g. '2').\n"
    "• For *nested_dependency* questions, respond **only** the exact 'verb + object entity' phrase using the identical wording found in the provided sentence,"
    " do not replace the object entity related to the verbs into pronouns (e.g. 'it').\n"
    "• For *causal_sequence* questions answer exactly 'no prior events' when no causal"
    " chain exists, answer **exactly** in 'subject + verb phrase + object' phrase using the wording found in the provided sentence otherwise.\n"
    "• For *chain_consequence* questions answer exactly 'none' when no chained subsequent consequence related to the entity exists.\n"
    "Respond with the short answer only: no explanations, no extra punctuation,"
    " and no leading labels such as 'Answer:'."
)

ANSWER_PREFIX_RE = re.compile(r"^\s*(?:answer:|Answer:|\*\*Answer:\*\*)\s*", re.IGNORECASE)
ARTICLE_RE = re.compile(r"\b(the|a|an)\s+", re.IGNORECASE)
THINK_START_RE = re.compile(r"<\s*think\s*>", flags=re.I)
THINK_END_RE = re.compile(r"</think>\s*", flags=re.I)
THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", flags=re.I)
HIDDEN_CHARS_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")
THINK_OPEN_RE = re.compile(r"<think>", flags=re.I)

nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

if SentenceTransformer is not None:
    _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("SentenceTransformer found: _sbert_model all-MiniLM-L6-v2 loaded successfully! \n")
else:
    _sbert_model = None  # type: ignore
    print("SentenceTransformer not found: _sbert_model not loaded! \n")


def _lemmatize(text: str) -> List[str]:
    """
    This function lemmatizes the input text, converting verbs to their base form.
    Args:
        text (str): The input text to lemmatize.
    Returns:
        List[str]: A list of lemmatized words, with verbs converted to their base form.
    Raises:
        None
    """

    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_stop or token.pos_ == "VERB"]


def normalize_answer(answer: str) -> str:
    """
    This function normalizes the answer by converting it to lowercase.
    Args:
        answer (str): The answer to normalize.
    Returns:
        answer (str): The normalized answer, with leading/trailing whitespace removed.
    Raises:
        None.
    """

    answer = answer.lower().strip()
    answer = re.sub(r'[^\w\s]', '', answer)
    answer = ' '.join(answer.split())
    
    return answer


def extract_key_components(answer: str, question_type: str) -> Dict[str, List[str]]:
    """
    This function extracts verbs, nouns, and other key components based on question type, with lemmatization.
    Args:
        answer (str): The answer to process.
        question_type (str): The type of question being answered.
    Returns:
        components (Dict[str, List[str]]): A dictionary containing lists of verbs, nouns, and the full answer.
    Raises:
        None
    """
    doc = nlp(answer.lower())
    
    components = {
        "verbs": [],
        "nouns": [],
        "full_answer": answer.lower(),
        "lemmas": []
    }
    
    for token in doc:
        if token.pos_ == "VERB":
            components["verbs"].append(token.text)
            components["lemmas"].append(token.lemma_)
        elif token.pos_ in ["NOUN", "PROPN"]:
            components["nouns"].append(token.text)
    
    return components


def check_verb_match(model_verb: str, gold_verb: str) -> bool:
    """
    This function checks if the model's verb matches the gold standard verb, accounting for lemmatization and known variations.
    Args:
        model_verb (str): The verb predicted by the model.
        gold_verb (str): The verb from the gold standard answer.
    Returns:
        bool: True if the verbs match, False otherwise.
    Raises:
        None
    """
    # Direct match
    if model_verb == gold_verb:
        return True
    
    # Check lemma match
    model_lemma = nlp(model_verb)[0].lemma_
    gold_lemma = nlp(gold_verb)[0].lemma_
    
    if model_lemma == gold_lemma:
        return True
    
    # Check known variations
    for base_form, variations in answer_variations.items():
        if model_verb in variations and gold_verb in variations:
            return True
        if model_verb == base_form and gold_verb in variations:
            return True
        if gold_verb == base_form and model_verb in variations:
            return True
    
    return False


def evaluate_action_performed(model_answer: str, gold_answer: str, question_type: str) -> bool:
    """
    This function evaluates if the model's answer for an action_performed question matches the gold standard answer.
    Args:
        model_answer (str): The answer predicted by the model.
        gold_answer (str): The gold standard answer.
        question_type (str): The type of question being answered.
    Returns:
        bool: True if the model's answer matches the gold standard answer, False otherwise.
    Raises:
        None
    """
    model_components = extract_key_components(model_answer, "action_performed")
    gold_components = extract_key_components(gold_answer, "action_performed")
    
    # Check if any verb in model answer matches gold answer verb
    for model_verb in model_components["verbs"] + model_components["lemmas"]:
        for gold_verb in gold_components["verbs"] + gold_components["lemmas"]:
            if check_verb_match(model_verb, gold_verb):
                return True
    
    # Check if gold verb appears anywhere in model answer (for verb+object cases)
    gold_verbs = gold_components["verbs"] + gold_components["lemmas"]
    for gold_verb in gold_verbs:
        if gold_verb in model_answer.lower():
            return True
    
    return False


def _sanitize(text: str) -> str:
    """
    This function sanitizes the input text by removing line breaks and hidden zero-width characters.
    Args:
        text (str): The input text to sanitize.
    Returns:
        str: The sanitized text, with line breaks and hidden characters removed.
    Raises:
        None
    """
    text = text.replace("\n", "").replace("\r", "")
    return HIDDEN_CHARS_RE.sub("", text)


def extract_prediction_claude(raw: str) -> str:
    """
    This function extracts the prediction from a Claude response.
    It returns the first non-empty line before the </think> tag if present.
    Args:
        raw (str): The raw response from Claude.
    Returns:
        str: The extracted prediction, sanitized and stripped of leading/trailing whitespace.
    Raises:
        None
    """
    before_think = THINK_OPEN_RE.split(raw, maxsplit=1)[0]

    for line in before_think.splitlines():
        stripped = line.strip()
        if stripped:
            return _sanitize(stripped)

    return _sanitize(before_think.strip())


def extract_prediction(raw: str) -> str:
    """
    This function extracts the prediction from a model response.
    It handles both Claude and Gemini responses, extracting the content before the <think> tag.
    Args:
        raw (str): The raw response from the model.
    Returns:
        str: The extracted prediction, sanitized and stripped of leading/trailing whitespace.
    Raises:
        None
    """
    cleaned = THINK_BLOCK_RE.sub("", raw).strip()

    after_tag = THINK_END_RE.split(raw)
    if len(after_tag) > 1:
        candidate = after_tag[-1].strip()
    else:
        candidate = cleaned

    # Take first non‑empty line
    for line in candidate.splitlines():
        line = line.strip()
        if line:
            return line
    return candidate

def _strip_articles(text: str) -> str:
    return ARTICLE_RE.sub("", text.strip().lower()).strip()


def answers_match(pred: str, gold: str, q_type: str, *, sim_threshold: float = 0.9) -> bool:
    """
    This function checks if the predicted answer matches the gold standard answer.
    It performs exact match checks, lemmatization, and semantic similarity using SBERT embeddings.
    Args:
        pred (str): The predicted answer.
        gold (str): The gold standard answer.
        q_type (str): The question type.
        sim_threshold (float, optional): The similarity threshold for semantic matching. Defaults to 0.9.
    Returns:
        bool: True if the answers match, False otherwise.
    Raises:
        None
    """
    # Exact match first (case/space insensitive)
    if pred.strip().lower() == gold.strip().lower():
        return True

    # Special handling per type
    if q_type == "agent_identification":
        if _strip_articles(pred) == _strip_articles(gold):
            return True
    elif q_type in {"action_performed", "nested_dependency"}:
        if _lemmatize(pred) == _lemmatize(gold):
            return True
    else:
        # Generic lemmatized match for any verbs present
        if _lemmatize(pred) == _lemmatize(gold):
            return True

    if _sbert_model is not None and util is not None:
        emb = _sbert_model.encode([pred, gold], normalize_embeddings=True)
        if util.cos_sim(emb[0], emb[1]).item() >= sim_threshold:
            return True

    return False


def claude_api_call_with_retry(client, messages, model, extended_thinking, max_retries=5):
    """
    This function makes Claude API calls with retry logic for errors.
    Args:
        client: The Anthropic client instance.
        messages (List[Dict[str, str]]): The messages to send to the Claude API.
        model (str): The model to use for the API call.
        extended_thinking (bool): Whether to enable extended thinking mode.
        max_retries (int, optional): The maximum number of retries for overloaded errors. Defaults to 5.
    Returns:
        Tuple: (response object, retry_count)
    Raises:
        OverloadedError: If the API is overloaded and retries are exhausted.
        RuntimeError: If the API is overloaded after max retries.
    """
    for attempt in range(max_retries):
        try:
            extra: Dict = {}
            if extended_thinking:
                extra["thinking"] = {"type": "enabled", "budget_tokens": 16000}
                resp = client.messages.create(
                    model=model,
                    max_tokens=20000,
                    temperature=1,
                    system=ZERO_SHOT_SYSTEM,
                    messages=messages,
                    **extra,
                )
            else:
                resp = client.messages.create(
                    model=model,
                    max_tokens=16000,
                    temperature=0,
                    system=ZERO_SHOT_SYSTEM,
                    messages=messages,
                    **extra,
                )
            return resp, attempt
            
        except OverloadedError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Claude API overloaded, retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"Claude API overloaded after {max_retries} attempts, giving up.")
                raise
    
    raise RuntimeError("Unexpected error in retry logic")


def deepseek_api_call_with_retry(client, messages, model, max_retries=5):
    """
    This function makes DeepSeek API calls with retry logic for errors.
    Args:
        client: The OpenAI client instance.
        messages (List[Dict[str, str]]): The messages to send to the DeepSeek API.
        model (str): The model to use for the API call.
        max_retries (int, optional): The maximum number of retries for errors. Defaults to 5.   
    Returns:
        Tuple: (response object, retry_count)
    Raises:
        RuntimeError: If the API is overloaded after max retries.
        Exception: If an unexpected error occurs.
    """
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=16000
            )
            return resp, attempt
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"DeepSeek API error: {e}, retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"DeepSeek API failed after {max_retries} attempts, giving up.")
                raise
    
    raise RuntimeError("Unexpected error in retry logic")


def chat_completion(
    client: Together,
    messages: List[Dict[str, str]],
    model: str,
    extended_thinking: bool = False,
    if_gemini: bool = False,
    gemini_thinking_disabled: bool = False,
    gemini_dynamic_thinking_enabled: bool = False,
    if_deepseek: bool = False,
) -> Tuple[str, int]:
    """
    This function generates a chat completion using the specified model and client.
    Args:
        client: The client instance for the model API.
        messages (List[Dict[str, str]]): The messages to send to the model.
        model (str): The model to use for the API call.
        extended_thinking (bool, optional): Whether to enable extended thinking mode. Defaults to False.
        if_gemini (bool, optional): Whether to use Gemini/Google Vertex AI. Defaults to False.
        gemini_thinking_disabled (bool, optional): Whether to disable thinking in Gemini. Defaults to False.
        gemini_dynamic_thinking_enabled (bool, optional): Whether to enable dynamic thinking in Gemini. Defaults to False.
        if_deepseek (bool, optional): Whether to use DeepSeek. Defaults to False.
    Returns:
        Tuple[str, int, int, int]: The generated content, prompt token count, completion token count, and total token count.
    Raises:
        RuntimeError: If the API call fails or if an unexpected error occurs.
        OverloadedError: If the API is overloaded and retries are exhausted.
        Exception: If an unexpected error occurs during the API call.
    """
    
    # >>> CHANGE: Use retry wrapper for Claude API calls <<<
    if Anthropic is not None and isinstance(client, Anthropic):
        resp, retry_count = claude_api_call_with_retry(client, messages, model, extended_thinking)
        
        if extended_thinking:
            content = resp.content[1].text.strip() + "<think>" + resp.content[0].thinking.strip() + "<think>"
        else:
            content = resp.content[0].text.strip() if resp.content else ""
        
        usage = resp.usage
        prompt_tok = getattr(usage, "input_tokens", 0)
        completion_tok = getattr(usage, "output_tokens", 0)
        total_tok = getattr(usage, "total_tokens", prompt_tok + completion_tok)

        return content, prompt_tok, completion_tok, total_tok


    if OpenAI is not None and if_deepseek:
        resp, retry_count = deepseek_api_call_with_retry(client, messages, model)
        
        content = resp.choices[0].message.content.strip()
        content = content + "<think>" + resp.choices[0].message.reasoning_content.strip() + "<think>"        
        # Extract token counts
        usage = getattr(resp, "usage", None)
        if usage is not None:
            prompt_tok = getattr(usage, "prompt_tokens", 0)
            completion_tok = getattr(usage, "completion_tokens", 0)
            total_tok = getattr(usage, "total_tokens", prompt_tok + completion_tok)
        else:
            prompt_tok = completion_tok = total_tok = 0
        
        return content, prompt_tok, completion_tok, total_tok


    # Gemini branch
    if genai is not None and if_gemini:
        system_content_parts = []
        user_content_parts = []

        for msg in messages:
            if msg.role == "system": # Accessing role attribute
                for part in msg.parts:
                    if part.text:
                        system_content_parts.append(part.text)
            elif msg.role == "user": # Accessing role attribute
                for part in msg.parts:
                    if part.text:
                        user_content_parts.append(part.text)
        
        # Join parts to form content strings
        system_content = "\n".join(system_content_parts)
        user_content = "\n".join(user_content_parts)
        full_prompt = system_content + user_content


        if gemini_thinking_disabled:
            generation_config = types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=16000,
                system_instruction=[types.Part.from_text(text=ZERO_SHOT_SYSTEM)],
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0
                )
            )
        elif gemini_dynamic_thinking_enabled:
            generation_config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=16000,
                system_instruction=[types.Part.from_text(text=ZERO_SHOT_SYSTEM)],
                thinking_config=types.ThinkingConfig(
                    thinking_budget=-1,
                    include_thoughts=True
                )
            )
        else:
            generation_config = types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=16000,
                system_instruction=[types.Part.from_text(text=ZERO_SHOT_SYSTEM)],
                thinkingConfig=types.ThinkingConfig(
                    thinkingBudget=16000,
                    include_thoughts=True
                )
            )

        try:
            response = client.models.generate_content(
                model=model,
                contents=messages,
                config=generation_config,
            )

            thoughts_accumulator = []
            answer_accumulator = []

            all_parts = response.candidates[0].content.parts
            if all_parts:
                for i, part in enumerate(all_parts):
                    if not part.text:
                        continue

                    if i < len(all_parts) - 1 or (hasattr(part, 'metadata') and part.metadata.get('is_thinking', False)):
                        thoughts_accumulator.append(part.text)
                    else:
                        answer_accumulator.append(part.text)
            
            thoughts = "".join(thoughts_accumulator)
            answer = "".join(answer_accumulator)

            if not thoughts:
                cleaned_answer_lines = [line.strip() for line in answer.split('\n') if line.strip()]
                if cleaned_answer_lines:
                    raw = answer
                    answer = cleaned_answer_lines[-1]
                else:
                    answer = answer
            else:
                raw = thoughts
            
            # Extract token counts from usage_metadata
            if hasattr(response, 'usage_metadata') and response.usage_metadata is not None:
                prompt_tok = (getattr(response.usage_metadata, 'prompt_token_count', 0) or 0)
                completion_tok = (getattr(response.usage_metadata, 'candidates_token_count', 0) or 0)
                total_tok = getattr(response.usage_metadata, 'total_token_count', prompt_tok + completion_tok)
            else:
                print("Counting in else. \n")
                user_content_for_token_count = ""
                for msg in messages:
                    if msg.role == "user":
                        for part in msg.parts:
                            if part.text:
                                user_content_for_token_count += part.text
                
                prompt_tok = len(user_content_for_token_count.split())
                completion_tok = len(answer.split()) + len(raw.split())
                total_tok = prompt_tok + completion_tok

            content = answer + "<think>" + raw.strip() + "<think>"
            return content, prompt_tok, completion_tok, total_tok
            
        except Exception as e:
            print(f"Error in Gemini API call: {e}")
            raise

    # Together AI branch
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=16000
        # stream=True
    )

    content = resp.choices[0].message.content
    usage = getattr(resp, "usage", None)
    if usage is not None:
        prompt_tok = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0)
        completion_tok = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0)
        total_tok = getattr(usage, "total_tokens", None) or (prompt_tok + completion_tok)
    else:
        prompt_tok = completion_tok = total_tok = 0


    return content, prompt_tok, completion_tok, total_tok


### QUESTION SAMPLING & FILTERING
def gather_questions(record: Dict, *, only_middle: bool, difficulty: Optional[str]) -> List[Dict]:
    """
    This function gathers questions from a sentence record.
    It filters questions based on whether they are middle entity questions and optional difficulty level.
    Args:
        record (Dict): The sentence record containing questions.
        only_middle (bool): Whether to include only middle entity questions.
        difficulty (Optional[str]): The difficulty level to filter questions by, if specified.
    Returns:
        List[Dict]: A list of filtered questions.
    Raises:
        None
    """
    questions: List[Dict] = []
    for ent_questions in record["questions_by_entity"].values():
        for q in ent_questions:
            if only_middle and not q.get("is_middle_entity", False):
                continue
            if difficulty and q.get("difficulty") != difficulty:
                continue
            questions.append(q)
    return questions


def sample_one_per_type(questions: List[Dict]) -> List[Dict]:
    """
    This function samples one question per unique type from a list of questions.
    Args:
        questions (List[Dict]): The list of questions to sample from.
    Returns:
        List[Dict]: A list of sampled questions, one per unique type.
    Raises:
        None
    """
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for q in questions:
        by_type[q["type"].lower()].append(q)
    return [random.choice(lst) for lst in by_type.values()]


### CORE EVALUATION LOGIC
def evaluate_sentence(
    client: Together,
    sentence_rec: Dict,
    model: str,
    *,
    only_middle: bool,
    claude_extended_thinking: bool,
    qwen_thinking_disabled: bool,
    gemini_thinking_disabled: bool,
    gemini_dynamic_thinking_enabled: bool,
    if_claude: bool,
    if_gemini: bool,
    if_deepseek: bool,
    difficulty: Optional[str],
    one_per_type: bool,
) -> Dict:
    """
    This function evaluates a sentence record by generating answers to its questions using the specified model.
    It gathers questions, generates answers, and checks correctness against the gold standard answers. 
    Args:
        client: The client instance for the model API.
        sentence_rec (Dict): The sentence record containing the sentence and questions.
        model (str): The model to use for generating answers.
        only_middle (bool): Whether to include only middle entity questions.
        claude_extended_thinking (bool): Whether to enable extended thinking mode for Claude.
        qwen_thinking_disabled (bool): Whether to disable thinking in Qwen.
        gemini_thinking_disabled (bool): Whether to disable thinking in Gemini.
        gemini_dynamic_thinking_enabled (bool): Whether to enable dynamic thinking in Gemini.
        if_claude (bool): Whether to use Claude for evaluation.
        if_gemini (bool): Whether to use Gemini for evaluation.
        if_deepseek (bool): Whether to use DeepSeek for evaluation.
        difficulty (Optional[str]): The difficulty level to filter questions by, if specified.
        one_per_type (bool): Whether to sample one question per unique type.
    Returns:
        Dict: A dictionary containing the sentence ID, sentence text, complexity level, results for each
              question, and token usage statistics.
    Raises:
        None
    """
    q_pool = gather_questions(sentence_rec, only_middle=only_middle, difficulty=difficulty)
    if one_per_type:
        q_pool = sample_one_per_type(q_pool)

    qwen_thinking_disable_token = "/no_think\n"
    results = []
    in_tok_total = out_tok_total = all_tok_total = 0

    for q in q_pool:
        if qwen_thinking_disabled:
            messages = [
                {"role": "system", "content": qwen_thinking_disable_token + ZERO_SHOT_SYSTEM},
                {"role": "user", "content": f"Sentence: {sentence_rec['sentence']}\nQuestion: {q['question']}"},
            ]
        elif if_claude:
            messages = [
                {"role": "user", "content": f"Sentence: {sentence_rec['sentence']}\nQuestion: {q['question']}"},
            ]
        elif if_gemini:
                messages = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=f"Sentence: {sentence_rec['sentence']}\nQuestion: {q['question']}")
                        ]
                    ),
                ]
        elif if_deepseek:
            messages = [
                {"role": "system", "content": ZERO_SHOT_SYSTEM},
                {"role": "user", "content": f"Sentence: {sentence_rec['sentence']}\nQuestion: {q['question']}"},
            ]
        else:
            messages = [
                {"role": "system", "content": ZERO_SHOT_SYSTEM},
                {"role": "user", "content": f"Sentence: {sentence_rec['sentence']}\nQuestion: {q['question']}"},
            ]

        raw, in_tok, out_tok, tot_tok = chat_completion(client, messages, model, claude_extended_thinking, if_gemini, gemini_thinking_disabled, gemini_dynamic_thinking_enabled, if_deepseek)
        in_tok_total += in_tok
        out_tok_total += out_tok
        all_tok_total += tot_tok

        if if_claude:
            print("Splitting claude results to get pred.\n")
            pred = extract_prediction_claude(raw)
            pred = re.split(r'(?i)<\s*think\s*>', raw, maxsplit=1)[0].strip()
            pred = pred.split("<think>")[0]
            print(f"pred is {pred}.\n")
        elif if_gemini:
            pred = raw.split("<think>")[0]
        elif if_deepseek:
            print("Processing DeepSeek results to get pred.\n")

            pred = extract_prediction_claude(raw)
            pred = re.split(r'(?i)<\s*think\s*>', raw, maxsplit=1)[0].strip()
            pred = pred.split("<think>")[0]
            
            print(f"pred is {pred}.\n")
        else:
            print("Splitting DeepSeek results to get pred.\n")
            pred = ANSWER_PREFIX_RE.sub("", raw).strip()
            pred = extract_prediction(pred)
            print(f"pred is {pred}.\n")
        
        print(f"Before answer_match, raw is: {raw}.\n")
        print(f"Before answer_match, pred is: {pred}.\n")
        correct = answers_match(pred, q["answer"], q["type"])
        print(f"After answer_match, pred is: {pred}.\n")
        print(f"After answer_match, correct is: {correct}.\n")


        if q["type"] == "action_performed":
            if len(pred.strip().split()) <= 2 and len(q["answer"].strip().split()) >= 2:
                model_norm = normalize_answer(pred)
                gold_norm = normalize_answer(q["answer"])
            
                if model_norm == gold_norm:
                    correct = True

                correct = evaluate_action_performed(model_norm, gold_norm, q["type"])
            elif correct == False and len(pred.strip().split()) == 1 and len(q["answer"].strip().split()) == 1:
                if check_verb_match(pred, q["answer"]):
                    correct = True
            
        q_result = q.copy()

        print(f"Before q_result.update, pred is: {pred}.\n")
        q_result.update(
            {
                "prediction": pred,
                "correct": correct,
                "raw": raw,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": tot_tok,
            }
        )

        results.append(q_result)

    return {
        "id": sentence_rec["id"],
        "sentence": sentence_rec["sentence"],
        "complexity_level": sentence_rec.get("complexity_level"),
        "results": results,
        "input_token_total": in_tok_total,
        "output_token_total": out_tok_total,
        "token_total": all_tok_total,
    }


### DATA LOADING / FLATTENING
def load_dataset(path: str) -> List[Dict]:
    """
    This function loads and flattens the dataset from a JSON file.
    It converts the nested structure into a flat list of records, each containing the sentence, complexity
    level, and question details.
    Args:
        path (str): The path to the JSON file containing the dataset.
    Returns:
        List[Dict]: A flat list of records, each containing the sentence, complexity level, and question details.
    Raises:
        FileNotFoundError: If the specified JSON file does not exist.
        json.JSONDecodeError: If the JSON file is not properly formatted.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat_records = []
    for level, block in data.items():
        for idx, sent_rec in enumerate(block["sentences"], 1):
            rec = sent_rec.copy()
            rec["complexity_level"] = level
            rec["id"] = f"{level}_{idx}"
            flat_records.append(rec)
    return flat_records


def main() -> None:
    """
    Main function to evaluate QA on center-embedded sentences.
    It parses command-line arguments, loads the dataset, and evaluates each sentence using the specified model.
    It aggregates results and prints accuracy statistics.
    """
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate QA on centre-embedded sentences (zero-shot only)")
    parser.add_argument("json_path", help="Path to edited_questions.json")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3", help="Together AI chat model name")
    parser.add_argument("--output", default="center_embedding_eval_results_v2.json", help="Output JSON path")
    parser.add_argument("--limit", type=int, help="Optional limit on number of sentences")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests")
    parser.add_argument("--middle-entity", action="store_true", help="Evaluate only questions where is_middle_entity = true")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], help="Only evaluate questions of this difficulty level")
    parser.add_argument("--extended-thinking", action="store_true", help="Enable Claude extended thinking block")
    parser.add_argument("--qwen-disable-thinking", action="store_true", help="Disable thinking mode for qwen from together AI.")
    parser.add_argument("--gemini-disable-thinking", action="store_true", help="Disable thinking mode for Gemini models")
    parser.add_argument("--gemini-enable-dynamic-thinking", action="store_true", help="Enable dynamic thinking mode for Gemini models")
    parser.add_argument("--one-per-type", action="store_true", help="Sample exactly one question per type per sentence")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    args = parser.parse_args()

    if_claude = args.model.startswith("claude")
    if_gemini = args.model.startswith("gemini")
    if_deepseek = args.model.startswith("deepseek") or args.model.startswith("DeepSeek")

    if if_claude:
        if Anthropic is None:
            sys.exit("anthropic package not installed.  pip install anthropic")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("ANTHROPIC_API_KEY environment variable is not set.")
        client = Anthropic(api_key=api_key)
    elif if_gemini:
        if genai is None:
            sys.exit("google-generativeai package not installed.  pip install google-generativeai")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            sys.exit("GEMINI_API_KEY environment variable is not set.")

        model_name = args.model
        if not model_name.startswith('models/'):
            model_name = f'models/{model_name}'

        client = genai.Client(vertexai=True,project="project_name",location="project_location")
    elif if_deepseek:
        if OpenAI is None:
            sys.exit("openai package not installed.  pip install openai")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            sys.exit("DEEPSEEK_API_KEY not found in environment variables or .env file.")
        
        client = OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_API_BASE
        )
    else:
        api_key = os.getenv("TOGETHER_API_KEY")

        if not api_key:
            sys.exit("TOGETHER_API_KEY environment variable is not set.")

        if args.seed is not None:
            random.seed(args.seed)

        client = Together(api_key=api_key)

    # Load dataset
    sentences = load_dataset(args.json_path)
    if args.limit:
        sentences = sentences[: args.limit]

    # Evaluation loop
    aggregate_stats = defaultdict(lambda: [0, 0])  # correct, total per level
    grand_correct = grand_total = 0
    token_grand_total = 0

    all_results = []
    g_correct = g_total = 0
    g_in_tok = g_out_tok = 0

    detailed_results = []

    prog = tqdm(sentences, desc="Evaluating", unit="sent")
    for sent_rec in prog:
        eval_res = evaluate_sentence(
            client,
            sent_rec,
            model=args.model,
            only_middle=args.middle_entity,
            claude_extended_thinking = args.extended_thinking,
            qwen_thinking_disabled = args.qwen_disable_thinking,
            gemini_thinking_disabled = args.gemini_disable_thinking,
            gemini_dynamic_thinking_enabled = args.gemini_enable_dynamic_thinking,
            if_claude = if_claude,
            if_gemini = if_gemini, 
            if_deepseek = if_deepseek,
            difficulty=args.difficulty,
            one_per_type=args.one_per_type,
        )
        detailed_results.append(eval_res)
        # Update stats
        correct_here = sum(q["correct"] for q in eval_res["results"])
        total_here = len(eval_res["results"])
        lvl = eval_res["complexity_level"]
        aggregate_stats[lvl][0] += correct_here
        aggregate_stats[lvl][1] += total_here
        grand_correct += correct_here
        grand_total += total_here

        g_in_tok += eval_res["input_token_total"]
        g_out_tok += eval_res["output_token_total"]
        prog.set_postfix(correct=f"{grand_correct}/{grand_total}", in_tok=g_in_tok, out_tok=g_out_tok)

        if args.sleep:
            time.sleep(args.sleep)


    print("\n=================  Accuracy  =================")

    header = f"{'Level':<10}{'Correct':>8}{'Total':>8}{'Accuracy':>10}"
    print(header)
    print("-" * len(header))
    for lvl in sorted(aggregate_stats):
        corr, tot = aggregate_stats[lvl]
        acc = 100 * corr / tot if tot else 0.0
        print(f"{lvl:<10}{corr:>8}{tot:>8}{acc:9.1f}%")
    print("-" * len(header))
    overall_acc = 100 * grand_correct / grand_total if grand_total else 0.0
    print(f"Total correct: {grand_correct}, Total questions: {grand_total}, Total Overall: {overall_acc:.1f}%\n")
    
    print(f"Total input tokens : {g_in_tok}")
    print(f"Total output tokens: {g_out_tok}")
    print(f"Grand total tokens : {g_in_tok + g_out_tok}\n")

    # Save results with appended predictions
    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"Detailed results written to {output_path.resolve()}")


if __name__ == "__main__":
    main()