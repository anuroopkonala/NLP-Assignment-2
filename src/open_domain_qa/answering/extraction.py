"""
Task 1 classical extractive answerer:
- Selects the sentence having highest lexical unigram overlap with question words.
- Extracts the longest noun phrase chunk or numeric span using NLTK POS parsing and regex.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import nltk


def ensure_nltk_resources() -> None:
    """Downloads required NLTK tokenizers and taggers quietly if not present."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


# Shallow chunk grammar for noun phrases: optional determiner/possessive, modifiers, and nouns
NP_GRAMMAR = r"NP: {<DT|PRP\$>?<JJ.*|VBG|VBN>*<NN.*>+}"
NP_PARSER = nltk.RegexpParser(NP_GRAMMAR)

# Regex to match cardinal numbers, years, percentages, decimals, and ordinals
NUM_REGEX = re.compile(r"\b(?:\d{1,4}(?:,\d{3})*(?:\.\d+)?|\d+(?:st|nd|rd|th)?)\b")


def extract_longest_candidate(sentence: str, question: str = "") -> str:
    """
    Extracts the longest noun phrase chunk or numeric span from a sentence.
    Applies clean determiner stripping, stopword filtering, and question-type heuristics.
    Falls back to the full sentence if neither an NP nor a number span is found.
    """
    if not sentence or not sentence.strip():
        return ""

    candidates: List[str] = []

    # 1. Parse noun phrases using chunk parsing on POS tags
    try:
        words = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        tree = NP_PARSER.parse(pos_tags)
        for subtree in tree.subtrees():
            if subtree.label() == "NP":
                np_text = " ".join(leaf[0] for leaf in subtree.leaves())
                if np_text.strip():
                    candidates.append(np_text.strip())
    except Exception:
        pass

    # 2. Extract number spans via regex
    number_spans = NUM_REGEX.findall(sentence)
    for num in number_spans:
        if num.strip():
            candidates.append(num.strip())

    if not candidates:
        return sentence.strip()

    # Clean and filter candidate spans
    q_words = set(re.findall(r"\b\w+\b", question.lower())) if question else set()
    stopwords = {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "is", "are", "was", "were", "the", "a", "an", "in", "on", "at", "of",
        "for", "to", "and", "do", "does", "did"
    }

    cleaned_cands = []
    for c in candidates:
        c_str = re.sub(r"^(?:the|a|an)\s+", "", c.strip(), flags=re.IGNORECASE)
        c_str = re.sub(r"[\"\'\.,;:!?\(\)]", "", c_str).strip()
        c_tokens = set(re.findall(r"\b\w+\b", c_str.lower()))
        if c_tokens and c_tokens.issubset(q_words | stopwords):
            continue
        if len(c_str) > 0:
            cleaned_cands.append(c_str)

    # Question-type specific preferences
    q_lower = question.lower()
    is_numeric_q = any(q_lower.startswith(p) for p in [
        "when", "what year", "what date", "how many", "how much", "how long", "how old"
    ])
    if is_numeric_q and number_spans:
        valid_nums = [n for n in number_spans if not set(re.findall(r"\b\w+\b", n.lower())).issubset(q_words)]
        if valid_nums:
            return valid_nums[0].strip()

    if cleaned_cands:
        proper_cands = [c for c in cleaned_cands if any(w[0].isupper() for w in c.split() if len(w) > 0)]
        if proper_cands and any(q_lower.startswith(p) for p in ["who", "which", "where"]):
            return max(proper_cands, key=len).strip()
        return max(cleaned_cands, key=len).strip()

    return max(candidates, key=len).strip()


def select_best_sentence(question: str, passage_text: str) -> Tuple[str, int]:
    """
    Finds the sentence in passage_text with highest lexical overlap with question words.
    Returns a tuple of (best_sentence, overlap_count).
    """
    try:
        sentences = nltk.sent_tokenize(passage_text)
    except Exception:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage_text) if s.strip()]

    if not sentences:
        return passage_text.strip(), 0

    q_words = set(re.findall(r"\b\w+\b", question.lower()))

    best_sent = sentences[0]
    max_overlap = -1

    for sent in sentences:
        sent_words = set(re.findall(r"\b\w+\b", sent.lower()))
        overlap = len(q_words & sent_words)
        if overlap > max_overlap:
            max_overlap = overlap
            best_sent = sent

    return best_sent.strip(), max(max_overlap, 0)


class ExtractiveAnswerer:
    """Extracts answer spans from top retrieved passages using lexical heuristics."""

    def __init__(self, top_passages_to_consider: int = 5) -> None:
        self.top_passages_to_consider = top_passages_to_consider
        ensure_nltk_resources()

    def extract_answer(
        self,
        question: str,
        retrieved_passages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Iterates over top passages to find the sentence with highest question overlap,
        then extracts the longest noun phrase chunk or numeric span.
        """
        passages_to_search = retrieved_passages[:self.top_passages_to_consider]
        if not passages_to_search:
            return {
                "prediction": "",
                "selected_sentence": "",
                "passage_rank": -1,
                "overlap_count": 0,
            }

        best_sentence = ""
        best_passage_rank = 1
        highest_overlap = -1

        for rank_idx, passage in enumerate(passages_to_search, start=1):
            text = passage.get("text", "")
            sent, overlap = select_best_sentence(question, text)
            if overlap > highest_overlap:
                highest_overlap = overlap
                best_sentence = sent
                best_passage_rank = rank_idx

        # If zero lexical overlap across all passages, fall back to top sentence of passage 1
        if not best_sentence and passages_to_search:
            best_sentence, _ = select_best_sentence(question, passages_to_search[0].get("text", ""))
            best_passage_rank = 1

        extracted_span = extract_longest_candidate(best_sentence, question)

        return {
            "prediction": extracted_span,
            "selected_sentence": best_sentence,
            "passage_rank": best_passage_rank,
            "overlap_count": highest_overlap,
        }
