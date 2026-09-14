import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from backend.config import MAX_LEN, MAX_DISTANCE
from backend.preprocessing import (
    parse_set_string,
    parse_float_string,
    find_span,
    word_span_to_token_span,
    parse_code_switch,
    calculate_switch_distance,
    distance_to_index,
)


def compute_switch_ids_from_code_switch_string(code_switch_str: str, max_distance: int = MAX_DISTANCE):
    """
    Computes token switch distance indices from raw code_switch column.
    """
    words, languages = parse_code_switch(code_switch_str)
    distances = calculate_switch_distance(languages, max_distance)
    return [distance_to_index(d, max_distance) for d in distances]


class DimABSADataset(Dataset):
    """
    PyTorch Dataset for Dimensional Aspect-Based Sentiment Analysis.
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

        # Parse string representations if not already lists
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

        # Compute token-aligned switch distance indices
        code_switch_raw = row.get("code_switch", "")
        if code_switch_raw and not pd.isna(code_switch_raw):
            switch_ids = compute_switch_ids_from_code_switch_string(
                str(code_switch_raw), self.max_distance
            )
        else:
            switch_ids = [self.max_distance] * len(words)

        token_switch = []
        for w in word_ids:
            if w is None or w >= len(switch_ids):
                token_switch.append(self.max_distance)
            else:
                token_switch.append(switch_ids[w])

        token_switch = torch.tensor(token_switch, dtype=torch.long)

        # Construct target aspect & opinion span matrices [MAX_LEN, MAX_LEN]
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

        valence = torch.tensor(row.get("valence_list", []), dtype=torch.float)
        arousal = torch.tensor(row.get("arousal_list", []), dtype=torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "switch_ids": token_switch,
            "aspect_matrix": aspect_matrix,
            "opinion_matrix": opinion_matrix,
            "valence": valence,
            "arousal": arousal,
            "sentence": sentence,
            "words": words,
        }


def collate_fn(batch: list):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "switch_ids": torch.stack([b["switch_ids"] for b in batch]),
        "aspect_matrix": torch.stack([b["aspect_matrix"] for b in batch]),
        "opinion_matrix": torch.stack([b["opinion_matrix"] for b in batch]),
        "valence": [b["valence"] for b in batch],
        "arousal": [b["arousal"] for b in batch],
        "sentence": [b["sentence"] for b in batch],
        "words": [b["words"] for b in batch],
    }
