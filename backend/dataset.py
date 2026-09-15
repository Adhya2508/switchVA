import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from backend.config import MAX_LEN, MAX_DISTANCE
from backend.preprocessing import (
    parse_set_string,
    parse_float_string,
    expand_aspect_dataset,
    find_span,
    word_span_to_token_span,
    parse_code_switch,
    calculate_switch_distance,
    distance_to_index,
    infer_word_languages,
)


def compute_switch_ids_for_tokens(
    sentence: str,
    code_switch_raw: str,
    tokenizer,
    max_len: int = MAX_LEN,
    max_distance: int = MAX_DISTANCE,
):
    """
    Computes token-aligned switch distance indices for SP-GSA attention.
    """
    words = sentence.strip().split()
    if not words:
        words = ["sentence"]

    if code_switch_raw and not pd.isna(code_switch_raw) and ":" in str(code_switch_raw):
        cs_words, languages = parse_code_switch(str(code_switch_raw))
        if len(cs_words) == len(words):
            distances = calculate_switch_distance(languages, max_distance)
        else:
            inferred_lang = infer_word_languages(words)
            distances = calculate_switch_distance(inferred_lang, max_distance)
    else:
        inferred_lang = infer_word_languages(words)
        distances = calculate_switch_distance(inferred_lang, max_distance)

    switch_indexes = [distance_to_index(d, max_distance) for d in distances]

    # Align to subwords
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)
    token_switch = []
    for wid in word_ids:
        if wid is None or wid >= len(switch_indexes):
            token_switch.append(max_distance)
        else:
            token_switch.append(switch_indexes[wid])

    # Pad or truncate to max_len
    if len(token_switch) < max_len:
        token_switch.extend([max_distance] * (max_len - len(token_switch)))
    else:
        token_switch = token_switch[:max_len]

    return torch.tensor(token_switch, dtype=torch.long)


class AspectEmotionDataset(Dataset):
    """
    Aspect-Level PyTorch Dataset for Dimensional Aspect-Based Sentiment Analysis.
    Every sample represents ONE aspect-opinion pair in context with individual (V, A) targets.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_len: int = MAX_LEN,
        max_distance: int = MAX_DISTANCE,
    ):
        if "aspect" not in dataframe.columns:
            self.df = expand_aspect_dataset(dataframe).reset_index(drop=True)
        else:
            self.df = dataframe.copy().reset_index(drop=True)

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_distance = max_distance

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sentence = str(row["sentence"])
        aspect = str(row["aspect"])
        opinion = str(row["opinion"])
        val_target = float(row["valence"])
        aro_target = float(row["arousal"])

        # Construct aspect-conditioned cross-encoding input:
        # text_a: Sentence | text_b: Aspect [SEP] Opinion
        aspect_opinion_prompt = f"Aspect: {aspect} | Opinion: {opinion}"
        
        encoding = self.tokenizer(
            text=sentence,
            text_pair=aspect_opinion_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Compute SP-GSA switch distance representation
        code_switch_raw = row.get("code_switch", "")
        switch_ids = compute_switch_ids_for_tokens(
            sentence=sentence,
            code_switch_raw=code_switch_raw,
            tokenizer=self.tokenizer,
            max_len=self.max_len,
            max_distance=self.max_distance,
        )

        targets = torch.tensor([val_target, aro_target], dtype=torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "switch_ids": switch_ids,
            "target": targets,
            "valence": val_target,
            "arousal": aro_target,
            "sentence": sentence,
            "aspect": aspect,
            "opinion": opinion,
            "sample_id": row.get("sample_id", f"{idx}"),
        }


def aspect_collate_fn(batch: list):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "switch_ids": torch.stack([b["switch_ids"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "valence": torch.tensor([b["valence"] for b in batch], dtype=torch.float),
        "arousal": torch.tensor([b["arousal"] for b in batch], dtype=torch.float),
        "sentence": [b["sentence"] for b in batch],
        "aspect": [b["aspect"] for b in batch],
        "opinion": [b["opinion"] for b in batch],
        "sample_id": [b["sample_id"] for b in batch],
    }


class SentenceSpanDataset(Dataset):
    """
    Sentence-level Dataset for training and evaluating Biaffine Aspect and Opinion span extractors.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_len: int = MAX_LEN,
        max_distance: int = MAX_DISTANCE,
    ):
        self.df = dataframe.copy().reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_distance = max_distance

        if "aspect_list" not in self.df.columns and "all_aspects" in self.df.columns:
            self.df["aspect_list"] = self.df["all_aspects"].apply(parse_set_string)
        if "opinion_list" not in self.df.columns and "all_opinions" in self.df.columns:
            self.df["opinion_list"] = self.df["all_opinions"].apply(parse_set_string)
        if "valence_list" not in self.df.columns and "valence_scores" in self.df.columns:
            self.df["valence_list"] = self.df["valence_scores"].apply(parse_float_string)
        if "arousal_list" not in self.df.columns and "arousal_scores" in self.df.columns:
            self.df["arousal_list"] = self.df["arousal_scores"].apply(parse_float_string)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sentence = str(row["sentence"])
        words = sentence.split()

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        word_ids = encoding.word_ids(batch_index=0)

        code_switch_raw = row.get("code_switch", "")
        switch_ids = compute_switch_ids_for_tokens(
            sentence=sentence,
            code_switch_raw=code_switch_raw,
            tokenizer=self.tokenizer,
            max_len=self.max_len,
            max_distance=self.max_distance,
        )

        aspect_matrix = torch.zeros(self.max_len, self.max_len, dtype=torch.float)
        opinion_matrix = torch.zeros(self.max_len, self.max_len, dtype=torch.float)

        for aspect in row.get("aspect_list", []):
            span = find_span(words, aspect)
            if span is not None:
                token_span = word_span_to_token_span(word_ids, span[0], span[1])
                if token_span is not None:
                    s, e = token_span
                    if s < self.max_len and e < self.max_len:
                        aspect_matrix[s, e] = 1.0

        for opinion in row.get("opinion_list", []):
            span = find_span(words, opinion)
            if span is not None:
                token_span = word_span_to_token_span(word_ids, span[0], span[1])
                if token_span is not None:
                    s, e = token_span
                    if s < self.max_len and e < self.max_len:
                        opinion_matrix[s, e] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "switch_ids": switch_ids,
            "aspect_matrix": aspect_matrix,
            "opinion_matrix": opinion_matrix,
            "sentence": sentence,
            "words": words,
        }


def sentence_collate_fn(batch: list):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "switch_ids": torch.stack([b["switch_ids"] for b in batch]),
        "aspect_matrix": torch.stack([b["aspect_matrix"] for b in batch]),
        "opinion_matrix": torch.stack([b["opinion_matrix"] for b in batch]),
        "sentence": [b["sentence"] for b in batch],
        "words": [b["words"] for b in batch],
    }
