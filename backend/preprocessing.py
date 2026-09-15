import re
import ast
import pandas as pd
import numpy as np
import torch
from backend.config import MAX_DISTANCE, MAX_LEN

# Lexicon for robust inference language detection on unseen Hinglish inputs
ENGLISH_STOPWORDS_AND_VOCAB = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
    # Domain words common in DimABSA
    "acting", "story", "camera", "quality", "battery", "food", "service", "movie",
    "phone", "display", "price", "college", "faculty", "placement", "teacher",
    "supportive", "awesome", "amazing", "bad", "good", "slow", "weak", "strong",
    "clean", "screen", "charging", "hotel", "room", "government", "scheme",
    "implementation", "match", "batting", "bowling", "climax", "direction",
    "sound", "music", "picture", "performance", "delivery", "design", "look",
    "build", "processor", "speed", "lag", "hanging", "issue", "problem", "review",
    "hospital", "doctor", "staff", "management", "experience", "budget", "value",
    "money", "worth", "waste", "horrible", "terrible", "excellent", "superb",
    "fantastic", "average", "disappointing", "lightweight", "heavy", "heating",
    "cooling", "wifi", "network", "signal", "bluetooth", "audio", "video", "link"
}

HINDI_STOPWORDS_AND_VOCAB = {
    "hai", "hain", "tha", "thi", "the", "hu", "hoon", "ho", "ka", "ki", "ke",
    "ko", "se", "me", "mein", "par", "pe", "ne", "aur", "ya", "lekin", "parantu",
    "magar", "to", "toh", "bhi", "hi", "kya", "kyu", "kyun", "kaise", "kaha",
    "kahan", "kab", "kaun", "kitna", "kitni", "kitne", "yeh", "ye", "voh", "woh",
    "wo", "apna", "apne", "apni", "mera", "meri", "mere", "tera", "teri", "tere",
    "uska", "uski", "uske", "unka", "unki", "unke", "hum", "ham", "hume", "humko",
    "tum", "tumhe", "aap", "aapko", "sab", "kuch", "koi", "kisi", "bohot", "bahut",
    "jyada", "kam", "mast", "achi", "achha", "achhi", "acha", "bekar", "kharab",
    "ganda", "bura", "buri", "sahi", "galat", "badhiya", "zabardast", "shandar",
    "bakwas", "laga", "lagi", "lage", "kar", "karo", "karta", "karti", "karte",
    "kiya", "kiye", "hoga", "hogi", "honge", "chahiye", "diya", "diye", "liya",
    "raha", "rahi", "rahe", "wala", "wali", "wale", "dekh", "dekha", "dekho",
    "sun", "suna", "padhaate", "samajh", "bhai", "yaar", "saab", "sir", "madam"
}


def parse_code_switch(value: str):
    """
    Parses 'word:LANG word2:LANG2' format into separate words and languages lists.
    """
    if pd.isna(value):
        return [], []
    words = []
    languages = []
    tokens = str(value).strip().split()
    for token in tokens:
        if ":" in token:
            word, lang = token.rsplit(":", 1)
        else:
            word, lang = token, "EN"
        words.append(word)
        languages.append(lang.upper())
    return words, languages


def parse_set_string(value):
    """
    Parses set string '{aspect1; aspect2}' or '[aspect1, aspect2]' into a list of strings.
    """
    if pd.isna(value):
        return []
    value = str(value).strip()
    value = value.replace("{", "").replace("}", "")
    if value == "":
        return []
    if value.startswith("[") and value.endswith("]"):
        try:
            result = ast.literal_eval(value)
            if isinstance(result, list):
                return [str(item).strip() for item in result if str(item).strip() != ""]
        except Exception:
            pass
    return [item.strip() for item in value.split(";") if item.strip() != ""]


def parse_float_string(value):
    """
    Parses float list string '{0.592; 0.475}' or '[0.592, 0.475]' into list of floats.
    """
    if pd.isna(value):
        return []
    str_list = parse_set_string(value)
    result = []
    for item in str_list:
        try:
            result.append(float(item))
        except ValueError:
            pass
    return result


def expand_aspect_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands a sentence-level DataFrame into an individual aspect-level DataFrame (1 aspect = 1 sample).
    Guarantees exact 1:1 alignment between aspect, opinion, valence, and arousal.
    """
    records = []
    for idx, row in df.iterrows():
        aspects = parse_set_string(row.get("all_aspects", ""))
        opinions = parse_set_string(row.get("all_opinions", ""))
        vals = parse_float_string(row.get("valence_scores", ""))
        aros = parse_float_string(row.get("arousal_scores", ""))

        # Match lengths safely
        k = min(len(aspects), len(opinions), len(vals), len(aros))
        for i in range(k):
            records.append({
                "sample_id": f"{row.get('sentence_id', idx)}_{i}",
                "sentence_id": row.get("sentence_id", idx),
                "sentence": str(row.get("sentence", "")),
                "code_switch": str(row.get("code_switch", "")),
                "aspect": str(aspects[i]).strip(),
                "opinion": str(opinions[i]).strip(),
                "valence": float(vals[i]),
                "arousal": float(aros[i]),
            })

    expanded_df = pd.DataFrame(records)
    return expanded_df


def find_switch_positions(languages: list):
    """
    Finds index positions where language switch occurs between adjacent words.
    """
    switch_positions = []
    for i in range(1, len(languages)):
        if languages[i] != languages[i - 1]:
            switch_positions.append(i)
    return switch_positions


def calculate_switch_distance(languages: list, max_distance: int = MAX_DISTANCE):
    """
    Calculates signed distance of each word position to the nearest language switch point.
    """
    n = len(languages)
    if n == 0:
        return []
    switch_positions = find_switch_positions(languages)
    if len(switch_positions) == 0:
        return [0] * n

    distances = []
    for i in range(n):
        nearest_switch = switch_positions[0]
        smallest_distance = abs(i - nearest_switch)
        for sp in switch_positions:
            curr_dist = abs(i - sp)
            if curr_dist < smallest_distance:
                smallest_distance = curr_dist
                nearest_switch = sp
        distance = i - nearest_switch
        distances.append(distance)
    return distances


def distance_to_index(distance: int, max_distance: int = MAX_DISTANCE):
    """
    Clamps distance to [-max_distance, max_distance] and converts to positive index [0, 2*max_distance].
    """
    clamped = max(-max_distance, min(max_distance, distance))
    return clamped + max_distance


def align_distances_to_tokens(word_distances: list, word_ids: list, max_distance: int = MAX_DISTANCE):
    """
    Aligns word-level switch distances to subword tokens using tokenizer word_ids.
    """
    token_distances = []
    for word_id in word_ids:
        if word_id is None or word_id >= len(word_distances):
            token_distances.append(0)
        else:
            token_distances.append(word_distances[word_id])
    return token_distances


def find_span(sentence_words: list, target_words):
    """
    Finds word-level span [start_idx, end_idx] of target_words in sentence_words.
    """
    if isinstance(target_words, str):
        target_words = target_words.split()
    sentence_lower = [str(w).lower() for w in sentence_words]
    target_lower = [str(w).lower() for w in target_words]
    m = len(target_lower)
    n = len(sentence_lower)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        if sentence_lower[i : i + m] == target_lower:
            return i, i + m - 1
    return None


def word_span_to_token_span(word_ids: list, word_start: int, word_end: int):
    """
    Maps word-level span [word_start, word_end] to token-level span [token_start, token_end].
    """
    token_start = None
    token_end = None
    for idx, w in enumerate(word_ids):
        if w == word_start and token_start is None:
            token_start = idx
        if w == word_end:
            token_end = idx
    if token_start is None or token_end is None:
        return None
    return token_start, token_end


def infer_word_languages(words: list):
    """
    Infers language tag ('EN' vs 'HI') for each word in arbitrary unseen Hinglish text.
    Uses vocabulary lexicon and phonetic heuristics.
    """
    languages = []
    for word in words:
        w_clean = re.sub(r"[^\w\s]", "", word.lower())
        if not w_clean:
            languages.append("EN")
            continue
        if w_clean in HINDI_STOPWORDS_AND_VOCAB:
            languages.append("HI")
        elif w_clean in ENGLISH_STOPWORDS_AND_VOCAB:
            languages.append("EN")
        elif any(w_clean.endswith(suf) for suf in ["aa", "iya", "iye", "kar", "wali", "wala", "wale", "ta", "ti", "te", "unga", "ungi", "enge"]):
            languages.append("HI")
        elif any(w_clean.startswith(pref) for pref in ["bhe", "kho", "chho", "dho", "bha", "kha"]):
            languages.append("HI")
        else:
            languages.append("EN")
    return languages


def clean_tokens(token_list: list):
    """
    Converts list of subword tokens back to a clean readable string.
    """
    words = []
    current = ""
    for tok in token_list:
        if tok in ["<s>", "</s>", "<pad>", "<unk>", "<mask>", "[CLS]", "[SEP]"]:
            continue
        tok = tok.replace("▁", " ")
        if tok.startswith(" "):
            if current != "":
                words.append(current)
            current = tok.strip()
        else:
            current += tok
    if current != "":
        words.append(current)
    return " ".join(words)


def token_span_to_words(start: int, end: int, word_ids: list, words: list):
    """
    Maps token span [start, end] accurately back to slice of original words list.
    """
    valid_word_ids = []
    for idx in range(start, end + 1):
        if idx >= len(word_ids):
            continue
        wid = word_ids[idx]
        if wid is not None and wid not in valid_word_ids:
            valid_word_ids.append(wid)
    if not valid_word_ids:
        return ""
    extracted = [words[wid] for wid in valid_word_ids if wid < len(words)]
    return " ".join(extracted)


def extract_aspect_opinion_candidates(sentence: str):
    """
    Rule-based and linguistic candidate extraction for Hinglish reviews.
    Identifies conjunctive clauses (e.g., 'but', 'aur', 'lekin', 'pr') and extracts aspect-opinion pairs.
    """
    clauses = re.split(r"\b(but|lekin|magar|parantu|aur|and|pr|or)\b", sentence, flags=re.IGNORECASE)
    pairs = []

    # Clean clauses
    current_clause = ""
    for segment in clauses:
        segment = segment.strip()
        if not segment:
            continue
        if segment.lower() in ["but", "lekin", "magar", "parantu", "aur", "and", "pr", "or"]:
            continue
        words = segment.split()
        if len(words) >= 2:
            # Simple heuristic: look for nouns/aspect words and opinion words
            asp_candidate = words[0] if len(words) < 4 else " ".join(words[:2])
            op_candidate = " ".join(words[1:]) if len(words) < 4 else " ".join(words[2:])
            pairs.append((asp_candidate, op_candidate))

    if not pairs and sentence.strip():
        words = sentence.strip().split()
        if len(words) >= 2:
            pairs.append((words[0], " ".join(words[1:])))
        else:
            pairs.append((sentence.strip(), sentence.strip()))

    return pairs
