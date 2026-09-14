import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from backend.config import (
    MAX_LEN,
    MAX_DISTANCE,
    HIDDEN_SIZE,
    MODEL_NAME,
    MODEL_SAVE_PATH,
    PRETRAINED_MODEL_PATH,
    SPAN_THRESHOLD,
    TOP_K_SPANS,
    DEVICE,
)
from backend.model import DimABSAModel
from backend.preprocessing import (
    infer_word_languages,
    calculate_switch_distance,
    distance_to_index,
    clean_tokens,
    token_span_to_words,
)


def decode_spans(score_matrix: torch.Tensor, tokens: list, threshold: float = SPAN_THRESHOLD, top_k: int = TOP_K_SPANS):
    """
    Decodes span candidates from 2D upper-triangular score matrix using Non-Maximum Suppression (NMS).
    """
    spans = []
    N = score_matrix.shape[0]

    for i in range(N):
        for j in range(i, N):
            score = score_matrix[i, j].item()
            if score >= threshold:
                spans.append((i, j, score))

    # Sort descending by confidence score
    spans = sorted(spans, key=lambda x: x[2], reverse=True)

    filtered = []
    used = []
    for s, e, sc in spans:
        overlap = False
        for us, ue in used:
            # Overlap check
            if not (e < us or s > ue):
                overlap = True
                break
        if not overlap:
            filtered.append((s, e, sc))
            used.append((s, e))
            if len(filtered) >= top_k:
                break

    return filtered


def get_affect_quadrant(valence: float, arousal: float):
    """
    Maps continuous Valence and Arousal scores (0-1) to Russell's Circumplex Affect Quadrants.
    """
    if valence >= 0.5 and arousal >= 0.5:
        quadrant = "Q1: High Valence, High Arousal"
        emotion = "Excited / Joyful / Enthusiastic 😊⚡"
        color = "#10b981"  # Emerald Green
    elif valence >= 0.5 and arousal < 0.5:
        quadrant = "Q4: High Valence, Low Arousal"
        emotion = "Pleasant / Content / Relaxed 😌🍃"
        color = "#06b6d4"  # Cyan
    elif valence < 0.5 and arousal >= 0.5:
        quadrant = "Q2: Low Valence, High Arousal"
        emotion = "Frustrated / Annoyed / Angry 😡⚡"
        color = "#ef4444"  # Red
    else:
        quadrant = "Q3: Low Valence, Low Arousal"
        emotion = "Disappointed / Sad / Dull 😞🌧️"
        color = "#8b5cf6"  # Purple

    polarity = "Positive" if valence >= 0.5 else "Negative"
    intensity = "High Arousal (Intense)" if arousal >= 0.5 else "Low Arousal (Subdued)"

    return {
        "quadrant": quadrant,
        "emotion": emotion,
        "polarity": polarity,
        "intensity": intensity,
        "color": color,
    }


class DimABSAInferenceEngine:
    """
    Inference Engine for Dimensional Aspect-Based Sentiment Analysis.
    """

    def __init__(self, model_path: str = None, device: torch.device = DEVICE):
        self.device = device
        self.model_path = model_path if model_path else MODEL_SAVE_PATH
        self.tokenizer = None
        self.model = None
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        # Determine model path / identifier
        model_source = PRETRAINED_MODEL_PATH if os.path.exists(PRETRAINED_MODEL_PATH) else "l3cube-pune/hing-roberta"
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        transformer = AutoModel.from_pretrained(model_source)

        self.model = DimABSAModel(
            transformer_model=transformer,
            hidden_size=transformer.config.hidden_size,
            max_distance=MAX_DISTANCE,
        ).to(self.device)

        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded trained DimABSA weights from {self.model_path}")
        else:
            print("No saved fine-tuned checkpoint found; initialized base HingRoBERTa DimABSA model.")

        self.model.eval()

    def prepare_input(self, sentence: str):
        words = sentence.strip().split()
        if not words:
            words = ["sentence"]

        languages = infer_word_languages(words)
        switch_distances = calculate_switch_distance(languages, MAX_DISTANCE)
        switch_indexes = [distance_to_index(d, MAX_DISTANCE) for d in switch_distances]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        token_switch = []
        for wid in word_ids:
            if wid is None or wid >= len(switch_indexes):
                token_switch.append(MAX_DISTANCE)
            else:
                token_switch.append(switch_indexes[wid])

        token_switch_tensor = torch.tensor([token_switch], dtype=torch.long, device=self.device)

        batch = {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device),
            "switch_ids": token_switch_tensor,
            "words": [words],
            "aspect_matrix": torch.zeros(1, MAX_LEN, MAX_LEN, device=self.device),
            "opinion_matrix": torch.zeros(1, MAX_LEN, MAX_LEN, device=self.device),
            "valence": [torch.tensor([])],
            "arousal": [torch.tensor([])],
        }

        return batch, words, languages, switch_distances, word_ids

    def predict(self, sentence: str, threshold: float = SPAN_THRESHOLD, top_k: int = TOP_K_SPANS):
        """
        Executes end-to-end inference on a single sentence.
        """
        if not sentence or not sentence.strip():
            return {
                "sentence": "",
                "aspects": [],
                "opinions": [],
                "pairs": [],
                "valence": 0.5,
                "arousal": 0.5,
                "affect": get_affect_quadrant(0.5, 0.5),
                "words_info": [],
            }

        batch, words, languages, switch_distances, word_ids = self.prepare_input(sentence)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch)

        aspect_probs = torch.sigmoid(outputs["aspect_scores"][0])
        opinion_probs = torch.sigmoid(outputs["opinion_scores"][0])
        val_score = float(outputs["valence"][0].cpu().item())
        aro_score = float(outputs["arousal"][0].cpu().item())

        tokens = self.tokenizer.convert_ids_to_tokens(batch["input_ids"][0])

        aspect_spans = decode_spans(aspect_probs, tokens, threshold=threshold, top_k=top_k)
        opinion_spans = decode_spans(opinion_probs, tokens, threshold=threshold, top_k=top_k)

        # Convert token spans back to original words
        aspect_list = []
        for s, e, score in aspect_spans:
            txt = token_span_to_words(s, e, word_ids, words)
            if txt:
                aspect_list.append({"text": txt, "score": round(score, 4), "token_range": (s, e)})

        opinion_list = []
        for s, e, score in opinion_spans:
            txt = token_span_to_words(s, e, word_ids, words)
            if txt:
                opinion_list.append({"text": txt, "score": round(score, 4), "token_range": (s, e)})

        # Deduplicate while preserving order
        def dedup(lst):
            seen = set()
            out = []
            for item in lst:
                if item["text"].lower() not in seen:
                    out.append(item)
                    seen.add(item["text"].lower())
            return out

        aspect_list = dedup(aspect_list)
        opinion_list = dedup(opinion_list)

        # Pair aspects and opinions
        pairs = []
        num_pairs = max(len(aspect_list), len(opinion_list))
        for i in range(num_pairs):
            asp = aspect_list[i]["text"] if i < len(aspect_list) else "-"
            asp_score = aspect_list[i]["score"] if i < len(aspect_list) else 0.0
            op = opinion_list[i]["text"] if i < len(opinion_list) else "-"
            op_score = opinion_list[i]["score"] if i < len(opinion_list) else 0.0

            pairs.append({
                "Aspect": asp,
                "Opinion": op,
                "Aspect Score": asp_score,
                "Opinion Score": op_score,
                "Valence": round(val_score, 3),
                "Arousal": round(aro_score, 3),
            })

        affect_info = get_affect_quadrant(val_score, aro_score)

        # Build word metadata for visual highlighting
        words_info = []
        for idx, (w, lang, dist) in enumerate(zip(words, languages, switch_distances)):
            is_aspect = any(w.lower() in asp["text"].lower().split() for asp in aspect_list)
            is_opinion = any(w.lower() in op["text"].lower().split() for op in opinion_list)
            words_info.append({
                "index": idx,
                "word": w,
                "language": lang,
                "switch_distance": dist,
                "is_aspect": is_aspect,
                "is_opinion": is_opinion,
            })

        gate_weights = outputs["gate"][0].mean(dim=-1).cpu().numpy()

        return {
            "sentence": sentence,
            "words": words,
            "tokens": tokens,
            "aspects": aspect_list,
            "opinions": opinion_list,
            "pairs": pairs,
            "valence": round(val_score, 4),
            "arousal": round(aro_score, 4),
            "affect": affect_info,
            "words_info": words_info,
            "gate_weights": gate_weights,
        }
