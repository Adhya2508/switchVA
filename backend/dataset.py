import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from backend.config import MAX_LEN, MAX_DISTANCE
from backend.preprocessing import (
    parse_set_string,
    parse_float_string,
    expand_aspect_dataset,
    parse_code_switch,
    calculate_switch_distance,
    distance_to_index,
    infer_word_languages,
)
from backend.model import (
    HINGLISH_AFFECT_LEXICON,
    extract_token_lexicon_features,
    construct_heterogeneous_adj,
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


class NSSGAspectDataset(Dataset):
    """
    Aspect-Level PyTorch Dataset for NSSG-DimNet.
    Prepares token inputs, switch positional indices, heterogeneous adjacency matrices,
    and affective priors for each sample.
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

        words = sentence.split()
        if not words:
            words = ["sentence"]

        langs = infer_word_languages(words)
        sw_dists = calculate_switch_distance(langs, self.max_distance)
        sw_idx_list = [distance_to_index(d, self.max_distance) for d in sw_dists]

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

        # Switch distance to tokens
        token_sw = []
        for wid in word_ids:
            if wid is None or wid >= len(sw_idx_list):
                token_sw.append(self.max_distance)
            else:
                token_sw.append(sw_idx_list[wid])
        if len(token_sw) < self.max_len:
            token_sw.extend([self.max_distance] * (self.max_len - len(token_sw)))
        else:
            token_sw = token_sw[:self.max_len]
        token_sw_t = torch.tensor(token_sw, dtype=torch.long)

        # Subword masks for aspect and opinion
        asp_words = [w.lower() for w in aspect.split()]
        op_words = [w.lower() for w in opinion.split()]

        asp_mask = np.zeros(self.max_len, dtype=np.float32)
        op_mask = np.zeros(self.max_len, dtype=np.float32)

        for ti, wid in enumerate(word_ids[:self.max_len]):
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

        # Adjacency tensor [3, max_len, max_len]
        adj = construct_heterogeneous_adj(words, langs, asp_words, op_words, max_len=self.max_len)

        # Token Lexicon Features [max_len, 3]
        lex_tokens = extract_token_lexicon_features(words, max_len=self.max_len)

        # Opinion Lexicon Prior [3] (Valence, Arousal, IsHit)
        op_v, op_a, op_hit = 0.5, 0.5, 0.0
        op_clean_words = [re.sub(r'[^\w]', '', w.lower()) for w in opinion.split()]
        hits = [HINGLISH_AFFECT_LEXICON[w] for w in op_clean_words if w in HINGLISH_AFFECT_LEXICON]
        if hits:
            op_v = float(np.mean([h[0] for h in hits]))
            op_a = float(np.mean([h[1] for h in hits]))
            op_hit = 1.0
        op_prior_t = torch.tensor([op_v, op_a, op_hit], dtype=torch.float)

        targets = torch.tensor([val_target, aro_target], dtype=torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "switch_ids": token_sw_t,
            "adj": adj,
            "lex_tokens": lex_tokens,
            "asp_mask": torch.tensor(asp_mask, dtype=torch.float),
            "op_mask": torch.tensor(op_mask, dtype=torch.float),
            "op_prior": op_prior_t,
            "target": targets,
            "valence": val_target,
            "arousal": aro_target,
            "sentence": sentence,
            "aspect": aspect,
            "opinion": opinion,
            "sample_id": row.get("sample_id", f"{idx}"),
        }


def nssg_collate_fn(batch: list):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "switch_ids": torch.stack([b["switch_ids"] for b in batch]),
        "adj": torch.stack([b["adj"] for b in batch]),
        "lex_tokens": torch.stack([b["lex_tokens"] for b in batch]),
        "asp_mask": torch.stack([b["asp_mask"] for b in batch]),
        "op_mask": torch.stack([b["op_mask"] for b in batch]),
        "op_prior": torch.stack([b["op_prior"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "valence": torch.tensor([b["valence"] for b in batch], dtype=torch.float),
        "arousal": torch.tensor([b["arousal"] for b in batch], dtype=torch.float),
        "sentence": [b["sentence"] for b in batch],
        "aspect": [b["aspect"] for b in batch],
        "opinion": [b["opinion"] for b in batch],
        "sample_id": [b["sample_id"] for b in batch],
    }


# Backwards compatibility alias
AspectEmotionDataset = NSSGAspectDataset
aspect_collate_fn = nssg_collate_fn
