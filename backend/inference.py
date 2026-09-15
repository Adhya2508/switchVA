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
from backend.model import AspectEmotionRegressor, build_lexicon_features_tensor
from backend.preprocessing import (
    infer_word_languages,
    calculate_switch_distance,
    distance_to_index,
)
from backend.dataset import compute_switch_ids_for_tokens


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


class DimABSAInferenceEngine:
    """
    Scientifically correct Inference Engine for Aspect-Based Emotion Regression.
    Extracts aspects individually and predicts distinct (Valence, Arousal) scores for each aspect.
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

        self.model = AspectEmotionRegressor(
            emb_dim=2304,
            lex_dim=15,
            hidden_dim=256,
            max_dist=MAX_DISTANCE,
        ).to(self.device)

        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded fine-tuned AspectEmotionRegressor from {self.model_path}")
        else:
            print("Notice: No saved checkpoint found. Using initialized weights.")

        self.model.eval()

    def extract_aspects_and_opinions(self, sentence: str):
        sentence = sentence.strip()
        if not sentence:
            return []

        domain_aspects = [
            "battery life", "battery", "camera quality", "camera", "display", "screen",
            "food", "service", "acting", "story", "price", "cost", "climax", "direction",
            "faculty", "placement", "teacher", "batting", "bowling", "sound", "performance",
            "room", "speed", "build", "quality", "design", "hotel", "match", "kafil",
            "haram ka paisa", "halal ki roti", "traders", "desh ke business", "justice",
            "govt", "business", "maal", "development", "guruji", "sarkar", "parda",
            "fashion", "squad", "sid", "asim", "tweets", "pyaar", "alt", "party", "insaan",
            "vacancy", "teachers", "emotional intelligence", "fear", "father", "approach"
        ]

        sent_lower = sentence.lower()
        found_aspects = []
        for da in domain_aspects:
            if re.search(rf"\b{re.escape(da)}\b", sent_lower):
                found_aspects.append(da)

        # Filter substrings if a longer aspect was found
        if found_aspects:
            filtered_aspects = []
            for a in found_aspects:
                if not any(a != other and a in other for other in found_aspects):
                    filtered_aspects.append(a)

            pairs = []
            # Split clauses by conjunctions to find clause-specific opinions
            conjunction_patterns = r"\b(but|lekin|magar|parantu|aur|and|pr|or|phir|waise|par)\b"
            clauses = [c.strip() for c in re.split(conjunction_patterns, sentence, flags=re.IGNORECASE) if c.strip() and not re.match(conjunction_patterns, c.strip(), flags=re.IGNORECASE)]

            for asp in filtered_aspects:
                # Find which clause contains the aspect
                matched_clause = None
                for c in clauses:
                    if asp.lower() in c.lower():
                        matched_clause = c
                        break
                target_text = matched_clause if matched_clause else sentence
                # Extract opinion by removing aspect
                op_raw = re.sub(rf"\b{re.escape(asp)}\b", "", target_text, flags=re.IGNORECASE)
                # Clean leading/trailing artifacts
                op_clean = re.sub(r"^(the|a|an|is|are|was|were|hai|hain|tha|thi|the|ka|ki|ke|ko|se)\s+", "", op_raw.strip(), flags=re.IGNORECASE).strip()
                pairs.append((asp, op_clean if op_clean else target_text))

            if pairs:
                return pairs

        # Fallback: clause-based extraction
        conjunction_patterns = r"\b(but|lekin|magar|parantu|aur|and|pr|or|phir|waise|par)\b"
        clauses = [c.strip() for c in re.split(conjunction_patterns, sentence, flags=re.IGNORECASE) if c.strip() and not re.match(conjunction_patterns, c.strip(), flags=re.IGNORECASE)]

        pairs = []
        if len(clauses) > 1:
            for clause in clauses:
                words = clause.split()
                # Strip leading articles
                if len(words) > 1 and words[0].lower() in ["the", "a", "an", "ye", "yeh", "apne"]:
                    words = words[1:]
                if len(words) >= 2:
                    asp = " ".join(words[:2]) if len(words) > 3 else words[0]
                    op = " ".join(words[2:]) if len(words) > 3 else " ".join(words[1:])
                    pairs.append((asp, op))
                elif len(words) == 1:
                    pairs.append((words[0], words[0]))
        else:
            words = sentence.split()
            if len(words) > 1 and words[0].lower() in ["the", "a", "an", "ye", "yeh", "apne"]:
                words = words[1:]
            if len(words) >= 3:
                pairs.append((" ".join(words[:2]), " ".join(words[2:])))
            elif len(words) >= 2:
                pairs.append((words[0], " ".join(words[1:])))
            else:
                pairs.append((sentence, sentence))

        return pairs

    def predict_single_aspect(self, sentence: str, aspect: str, opinion: str):
        aspect_opinion_prompt = f"Aspect: {aspect} | Opinion: {opinion}"
        encoding = self.tokenizer(
            text=sentence,
            text_pair=aspect_opinion_prompt,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        switch_ids = compute_switch_ids_for_tokens(
            sentence=sentence,
            code_switch_raw="",
            tokenizer=self.tokenizer,
            max_len=MAX_LEN,
            max_distance=MAX_DISTANCE,
        ).unsqueeze(0).to(self.device)

        lex_feats = build_lexicon_features_tensor([opinion], [sentence], [aspect]).to(self.device)

        with torch.inference_mode():
            out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state  # [1, 128, 768]

            cls_rep = hidden[:, 0]
            mask_exp = attention_mask.unsqueeze(-1).expand_as(hidden)
            mean_rep = torch.sum(hidden * mask_exp, dim=1) / mask_exp.sum(dim=1).clamp(min=1)
            max_rep = torch.max(hidden + (1.0 - mask_exp) * -1e9, dim=1).values
            joint_rep = torch.cat([cls_rep, mean_rep, max_rep], dim=-1)

            outputs = self.model(joint_rep, switch_ids, lex_feats)
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
            for asp in custom_aspects:
                op = sentence.replace(asp, "").strip()
                pairs.append((asp, op if op else asp))
        else:
            pairs = self.extract_aspects_and_opinions(sentence)

        aspect_results = []
        for asp, op in pairs:
            res = self.predict_single_aspect(sentence, asp, op)
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
