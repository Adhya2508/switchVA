import os
import re
import sys
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Set CPU threading
torch.set_num_threads(12)

# Add workspace to path
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from backend.config import (
    MAX_LEN,
    MAX_DISTANCE,
    MODEL_SAVE_PATH,
    PRETRAINED_MODEL_PATH,
    DEVICE,
)
from backend.model import (
    NSSGDimNet,
    HINGLISH_AFFECT_LEXICON,
    extract_token_lexicon_features,
    construct_heterogeneous_adj,
)
from backend.preprocessing import (
    infer_word_languages,
    calculate_switch_distance,
    distance_to_index,
)


def get_affect_quadrant(valence: float, arousal: float):
    """
    Maps continuous Valence and Arousal scores (0-1) to Russell's Circumplex Affect Quadrants.
    """
    if valence >= 0.5 and arousal >= 0.5:
        quadrant = "Q1: High Valence, High Arousal"
        emotion = "Excited / Joyful / Enthusiastic 😊⚡"
        color = "#10b981"  # Emerald Green
        polarity = "Positive"
        intensity = "High Arousal (Intense)"
    elif valence >= 0.5 and arousal < 0.5:
        quadrant = "Q4: High Valence, Low Arousal"
        emotion = "Pleasant / Content / Relaxed 😌🍃"
        color = "#06b6d4"  # Cyan
        polarity = "Positive"
        intensity = "Low Arousal (Subdued)"
    elif valence < 0.5 and arousal >= 0.5:
        quadrant = "Q2: Low Valence, High Arousal"
        emotion = "Frustrated / Annoyed / Angry 😡⚡"
        color = "#ef4444"  # Red
        polarity = "Negative"
        intensity = "High Arousal (Intense)"
    else:
        quadrant = "Q3: Low Valence, Low Arousal"
        emotion = "Disappointed / Sad / Dull 😞🌧️"
        color = "#8b5cf6"  # Purple
        polarity = "Negative"
        intensity = "Low Arousal (Subdued)"

    # Confidence score based on distance from neutral (0.5, 0.5)
    dist_from_center = np.sqrt((valence - 0.5) ** 2 + (arousal - 0.5) ** 2)
    max_dist = np.sqrt(0.5**2 + 0.5**2)
    confidence = min(0.99, max(0.50, 0.50 + (dist_from_center / max_dist) * 0.49))

    return {
        "quadrant": quadrant,
        "emotion": emotion,
        "polarity": polarity,
        "intensity": intensity,
        "color": color,
        "confidence": round(float(confidence), 3),
    }


def split_hinglish_clauses(sentence: str):
    """
    Splits Hinglish sentences into constituent semantic clauses based on:
    1. Punctuation (, . ; ! -)
    2. Explicit conjunctions (but, lekin, magar, par, parantu, aur, and, etc.)
    3. Tense/auxiliary verb boundaries (tha, thi, the, hai, hain, etc.) when followed by words
    """
    s = sentence.strip()
    raw_segments = re.split(r'[,;!?\.\n]+', s)
    
    clauses = []
    conjunction_pattern = r'\b(but|lekin|magar|parantu|par|kintu|aur|and|or|phir|waise|halanki|while|whereas)\b'
    aux_boundary_pattern = r'(\b(?:tha|thi|the|hai|hain|hoga|hogi|gaya|gayi|raha|rahi)\b)(?=\s+[a-zA-Z0-9])'
    
    for seg in raw_segments:
        seg = seg.strip()
        if not seg:
            continue
        conj_parts = re.split(conjunction_pattern, seg, flags=re.IGNORECASE)
        current = ""
        for p in conj_parts:
            p = p.strip()
            if not p:
                continue
            if re.fullmatch(conjunction_pattern, p, flags=re.IGNORECASE):
                if current.strip():
                    clauses.append(current.strip())
                    current = ""
            else:
                sub_parts = re.split(aux_boundary_pattern, p, flags=re.IGNORECASE)
                accum = ""
                for sp in sub_parts:
                    sp = sp.strip()
                    if not sp:
                        continue
                    if re.fullmatch(r'\b(tha|thi|the|hai|hain|hoga|hogi|gaya|gayi|raha|rahi)\b', sp, flags=re.IGNORECASE):
                        accum = (accum + " " + sp).strip()
                        clauses.append(accum)
                        accum = ""
                    else:
                        if accum:
                            clauses.append(accum)
                            accum = sp
                        else:
                            accum = sp
                if accum.strip():
                    current = accum.strip()
        if current.strip():
            clauses.append(current.strip())
            
    return clauses if clauses else [s]


class DimABSAInferenceEngine:
    """
    Inference Engine powered by NSSG-DimNet (Neuro-Symbolic Switch-Gated Dual-Graph Network).
    Performs precise Aspect-Opinion span extraction and continuous (Valence, Arousal) regression.
    """

    def __init__(self, model_path: str = None, device: torch.device = DEVICE):
        self.device = device
        self.model_path = model_path if model_path else MODEL_SAVE_PATH
        self.tokenizer = None
        self.transformer = None
        self.model = None
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        model_source = PRETRAINED_MODEL_PATH if os.path.exists(PRETRAINED_MODEL_PATH) else "l3cube-pune/hing-roberta"
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.transformer = AutoModel.from_pretrained(model_source).to(self.device)
        self.transformer.eval()

        self.model = NSSGDimNet(
            hidden_dim=768,
            switch_dim=64,
            span_dim=128,
            graph_dim=128,
            lex_dim=3,
            max_dist=MAX_DISTANCE,
        ).to(self.device)

        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded fine-tuned NSSG-DimNet from {self.model_path}")
        else:
            print("Notice: No saved checkpoint found. Using initialized weights.")

        self.model.eval()

    def _extract_opinion_from_clause(self, clause: str, aspect: str):
        """
        Extracts the opinion phrase associated with the aspect in the given clause.
        """
        clause_words = clause.split()
        asp_words = [w.lower() for w in aspect.split()]

        # Remove aspect words
        op_words = [w for w in clause_words if w.lower() not in asp_words]
        
        # Strip leading prepositional/auxiliary markers (ka, ki, ke, me, mein, par, the, a, etc.)
        leading_stop = {"ka", "ki", "ke", "ko", "se", "me", "mein", "par", "the", "a", "an", "ye", "yeh", "wo", "woh", "is", "us", "movie", "match", "film", "game"}
        while op_words and op_words[0].lower() in leading_stop:
            op_words = op_words[1:]

        op_str = " ".join(op_words).strip()
        return op_str if op_str else clause

    def extract_aspects_and_opinions(self, sentence: str):
        sentence = sentence.strip()
        if not sentence:
            return []

        # Domain aspect dictionary
        domain_aspects = [
            "battery life", "battery backup", "battery", "camera quality", "camera", "display", "screen",
            "food", "service", "acting", "story", "price", "cost", "climax", "direction",
            "faculty", "placement", "teacher", "batting", "bowling", "sound", "performance",
            "room", "speed", "build", "quality", "design", "hotel", "kafil",
            "haram ka paisa", "halal ki roti", "traders", "desh ke business", "justice",
            "govt", "business", "maal", "development", "guruji", "sarkar", "parda",
            "fashion", "squad", "sid", "asim", "tweets", "pyaar", "alt", "party", "insaan",
            "vacancy", "teachers", "emotional intelligence", "fear", "father", "approach"
        ]

        sent_lower = sentence.lower()
        clauses = split_hinglish_clauses(sentence)

        extracted_pairs = []

        for clause in clauses:
            cl_lower = clause.lower()
            matched_aspects = [da for da in domain_aspects if re.search(rf"\b{re.escape(da)}\b", cl_lower)]
            
            # Filter substring overlaps (e.g. keep 'battery backup' over 'battery')
            if matched_aspects:
                matched_aspects = [a for a in matched_aspects if not any(a != o and a in o for o in matched_aspects)]
                for asp in matched_aspects:
                    op = self._extract_opinion_from_clause(clause, asp)
                    extracted_pairs.append((asp, op, clause))
            else:
                # Fallback heuristic for clauses without exact domain dictionary match
                words = clause.split()
                if len(words) >= 2:
                    # Filter leading articles/demonstratives
                    if words[0].lower() in ["the", "a", "an", "ye", "yeh", "apne", "is"]:
                        words = words[1:]
                    if len(words) >= 3:
                        asp = words[0]
                        op = " ".join(words[1:])
                    elif len(words) == 2:
                        asp = words[0]
                        op = words[1]
                    else:
                        asp = clause
                        op = clause
                    extracted_pairs.append((asp, op, clause))

        return extracted_pairs if extracted_pairs else [(sentence, sentence, sentence)]

    def predict_single_aspect(self, sentence: str, aspect: str, opinion: str, clause: str = None):
        """
        Runs NSSG-DimNet inference for an aspect-opinion pair in context.
        """
        eval_context = clause if clause else sentence
        words = eval_context.split()
        if not words:
            words = ["sentence"]

        langs = infer_word_languages(words)
        sw_dists = calculate_switch_distance(langs, MAX_DISTANCE)
        sw_idx_list = [distance_to_index(d, MAX_DISTANCE) for d in sw_dists]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        word_ids = encoding.word_ids(batch_index=0)

        # Token switch ids
        token_sw = []
        for wid in word_ids:
            if wid is None or wid >= len(sw_idx_list):
                token_sw.append(MAX_DISTANCE)
            else:
                token_sw.append(sw_idx_list[wid])
        if len(token_sw) < MAX_LEN:
            token_sw.extend([MAX_DISTANCE] * (MAX_LEN - len(token_sw)))
        else:
            token_sw = token_sw[:MAX_LEN]
        switch_ids = torch.tensor([token_sw], dtype=torch.long).to(self.device)

        # Aspect & Opinion subword masks
        asp_words = [w.lower() for w in aspect.split()]
        op_words = [w.lower() for w in opinion.split()]

        asp_mask = np.zeros(MAX_LEN, dtype=np.float32)
        op_mask = np.zeros(MAX_LEN, dtype=np.float32)

        for ti, wid in enumerate(word_ids[:MAX_LEN]):
            if wid is not None and wid < len(words):
                w = words[wid].lower()
                if w in asp_words:
                    asp_mask[ti] = 1.0
                if w in op_words:
                    op_mask[ti] = 1.0
        if asp_mask.sum() == 0:
            asp_mask[0] = 1.0
        if op_mask.sum() == 0:
            op_mask[0] = 1.0

        asp_mask_t = torch.from_numpy(asp_mask).unsqueeze(0).float().to(self.device)
        op_mask_t = torch.from_numpy(op_mask).unsqueeze(0).float().to(self.device)

        # Adjacency tensor [1, 3, MAX_LEN, MAX_LEN]
        adj = construct_heterogeneous_adj(words, langs, asp_words, op_words, max_len=MAX_LEN).unsqueeze(0).to(self.device)

        # Token Lexicon features [1, MAX_LEN, 3]
        lex_tokens = extract_token_lexicon_features(words, max_len=MAX_LEN).unsqueeze(0).to(self.device)

        # Opinion Lexicon Prior [1, 3]
        op_v, op_a, op_hit = 0.5, 0.5, 0.0
        op_clean_words = [re.sub(r'[^\w]', '', w.lower()) for w in opinion.split()]
        hits = [HINGLISH_AFFECT_LEXICON[w] for w in op_clean_words if w in HINGLISH_AFFECT_LEXICON]
        if hits:
            op_v = float(np.mean([h[0] for h in hits]))
            op_a = float(np.mean([h[1] for h in hits]))
            op_hit = 1.0
        op_prior_t = torch.tensor([[op_v, op_a, op_hit]], dtype=torch.float).to(self.device)

        with torch.inference_mode():
            out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            token_embs = out.last_hidden_state  # [1, MAX_LEN, 768]

            outputs = self.model(
                token_embs=token_embs,
                switch_ids=switch_ids,
                adj_matrices=adj,
                lex_token_feats=lex_tokens,
                asp_mask=asp_mask_t,
                op_mask=op_mask_t,
                attention_mask=attention_mask,
                opinion_lex_prior=op_prior_t,
            )

            val_score = float(outputs["valence"][0].cpu().item())
            aro_score = float(outputs["arousal"][0].cpu().item())

        affect_info = get_affect_quadrant(val_score, aro_score)

        return {
            "aspect": aspect,
            "opinion": opinion,
            "valence": round(val_score, 4),
            "arousal": round(aro_score, 4),
            "polarity": affect_info["polarity"],
            "intensity": affect_info["intensity"],
            "emotion": affect_info["emotion"],
            "quadrant": affect_info["quadrant"],
            "color": affect_info["color"],
            "confidence": affect_info["confidence"],
        }

    def predict(self, sentence: str, custom_aspects: list = None):
        sentence = sentence.strip()
        if not sentence:
            return {
                "sentence": "",
                "aspects": [],
                "words_info": [],
            }

        words = sentence.split()
        languages = infer_word_languages(words)
        switch_distances = calculate_switch_distance(languages, MAX_DISTANCE)

        if custom_aspects and len(custom_aspects) > 0:
            pairs = []
            clauses = split_hinglish_clauses(sentence)
            for asp in custom_aspects:
                matched_clause = sentence
                for c in clauses:
                    if re.search(rf"\b{re.escape(asp)}\b", c, flags=re.IGNORECASE):
                        matched_clause = c
                        break
                op = self._extract_opinion_from_clause(matched_clause, asp)
                pairs.append((asp, op, matched_clause))
        else:
            pairs = self.extract_aspects_and_opinions(sentence)

        aspect_results = []
        for item in pairs:
            asp, op = item[0], item[1]
            clause = item[2] if len(item) > 2 else sentence
            res = self.predict_single_aspect(sentence, asp, op, clause=clause)
            aspect_results.append(res)

        words_info = []
        for idx, (w, lang, dist) in enumerate(zip(words, languages, switch_distances)):
            is_aspect = any(w.lower() in a["aspect"].lower().split() for a in aspect_results)
            is_opinion = any(w.lower() in a["opinion"].lower().split() for a in aspect_results)
            words_info.append({
                "index": idx,
                "word": w,
                "language": lang,
                "switch_distance": dist,
                "is_aspect": is_aspect,
                "is_opinion": is_opinion,
            })

        return {
            "sentence": sentence,
            "words": words,
            "aspects": aspect_results,
            "words_info": words_info,
        }
