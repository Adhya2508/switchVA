import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from backend.config import (
    MAX_LEN,
    MAX_DISTANCE,
    HIDDEN_SIZE,
    DROPOUT,
    DEVICE,
)

# Hinglish Polarity & Affect Lexicon
HINGLISH_AFFECT_LEXICON = {
    # High Valence / Positive
    "mast": (0.85, 0.65), "zabardast": (0.90, 0.75), "shandar": (0.88, 0.70), "awesome": (0.88, 0.75),
    "amazing": (0.89, 0.72), "excellent": (0.90, 0.68), "superb": (0.88, 0.70), "fantastic": (0.88, 0.72),
    "badhiya": (0.82, 0.60), "achi": (0.80, 0.55), "achha": (0.80, 0.55), "achhi": (0.80, 0.55),
    "acha": (0.78, 0.55), "good": (0.75, 0.52), "support": (0.78, 0.58), "supportive": (0.82, 0.55),
    "pyar": (0.85, 0.60), "love": (0.88, 0.65), "win": (0.85, 0.70), "gold": (0.88, 0.68),
    "changa": (0.80, 0.50), "changi": (0.80, 0.50), "best": (0.90, 0.65), "sahi": (0.70, 0.50),
    "right": (0.70, 0.50), "honestly": (0.75, 0.48), "protect": (0.80, 0.60), "save": (0.78, 0.58),
    "like": (0.72, 0.50), "khush": (0.85, 0.62), "enjoy": (0.85, 0.65), "sundar": (0.82, 0.55),
    "badiya": (0.82, 0.60), "great": (0.85, 0.65), "clean": (0.75, 0.50), "fast": (0.75, 0.60),
    "strong": (0.75, 0.60),
    
    # Low Valence / Negative
    "bakwas": (0.12, 0.85), "bekar": (0.15, 0.75), "kharab": (0.15, 0.80), "ganda": (0.12, 0.82),
    "bura": (0.18, 0.75), "puri": (0.15, 0.70), "fail": (0.12, 0.80), "scam": (0.18, 0.82),
    "dhoka": (0.10, 0.88), "fraud": (0.10, 0.88), "terrorist": (0.05, 0.92), "bhikari": (0.10, 0.85),
    "haram": (0.08, 0.90), "galat": (0.22, 0.70), "laprvahi": (0.12, 0.80), "fuck": (0.08, 0.90),
    "sad": (0.25, 0.45), "disappointing": (0.18, 0.68), "horrible": (0.10, 0.85), "terrible": (0.10, 0.85),
    "waste": (0.15, 0.75), "issue": (0.25, 0.65), "problem": (0.22, 0.68), "weak": (0.28, 0.55),
    "slow": (0.30, 0.50), "hanging": (0.18, 0.75), "lag": (0.20, 0.72), "heating": (0.20, 0.75),
    "lannat": (0.12, 0.85), "hate": (0.10, 0.88), "mulle": (0.15, 0.80), "marne": (0.12, 0.90),
    "gira": (0.18, 0.75), "mushkil": (0.25, 0.65), "chhed": (0.18, 0.78), "chori": (0.10, 0.85),
    "danga": (0.12, 0.88), "dango": (0.12, 0.88), "mahangai": (0.22, 0.75), "chamcho": (0.15, 0.78),
    "kaid": (0.15, 0.75), "badh": (0.18, 0.75), "moun": (0.25, 0.60), "threat": (0.10, 0.90),
    "bad": (0.18, 0.70), "worst": (0.10, 0.85), "cheat": (0.12, 0.85),
}


def extract_affect_lexicon_vector(text: str):
    """
    Extracts continuous statistical affect summary features from Hinglish text.
    """
    words = re.findall(r"\w+", str(text).lower())
    v_scores = []
    a_scores = []
    for w in words:
        if w in HINGLISH_AFFECT_LEXICON:
            v_scores.append(HINGLISH_AFFECT_LEXICON[w][0])
            a_scores.append(HINGLISH_AFFECT_LEXICON[w][1])
    if v_scores:
        return [
            float(np.mean(v_scores)),
            float(np.mean(a_scores)),
            float(np.min(v_scores)),
            float(np.max(v_scores)),
            float(len(v_scores)),
        ]
    return [0.411, 0.605, 0.411, 0.605, 0.0]


def build_lexicon_features_tensor(opinions: list, sentences: list, aspects: list):
    """
    Constructs 15-dimensional lexicon prior feature matrix for a batch of aspect samples.
    """
    feats = []
    for op, sent, asp in zip(opinions, sentences, aspects):
        op_l = extract_affect_lexicon_vector(op)
        sent_l = extract_affect_lexicon_vector(sent)
        asp_l = extract_affect_lexicon_vector(asp)
        feats.append(op_l + sent_l + asp_l)
    return torch.tensor(feats, dtype=torch.float)


class SwitchGatedSelfAttention(nn.Module):
    """
    SP-GSA: Switch-Gated Self-Attention Layer.
    Embeds signed distance to the nearest code-switch boundary and modulates
    Transformer token representations via a learned gating mechanism.
    """

    def __init__(self, hidden_size: int = HIDDEN_SIZE, max_distance: int = MAX_DISTANCE):
        super().__init__()
        self.max_distance = max_distance
        self.hidden_size = hidden_size
        self.switch_embedding = nn.Embedding(2 * max_distance + 1, hidden_size)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor, switch_ids: torch.Tensor):
        switch_embed = self.switch_embedding(switch_ids)
        concat = torch.cat([hidden_states, switch_embed], dim=-1)
        gate = torch.sigmoid(self.gate(concat))
        output = hidden_states + gate * hidden_states
        return output, gate


class AspectEmotionRegressor(nn.Module):
    """
    Scientifically correct Aspect-Level Emotion Regressor for continuous Valence & Arousal.
    Fuses:
      1. 2304-D Contextual embeddings from fine-tuned HingRoBERTa (CLS + Mean + Max)
      2. SP-GSA Code-switch boundary distance embeddings
      3. 15-D Continuous Affect Lexicon priors
    Through a balanced projection layer and gated cross-feature fusion.
    """

    def __init__(
        self,
        emb_dim: int = 2304,
        lex_dim: int = 15,
        hidden_dim: int = 256,
        max_dist: int = MAX_DISTANCE,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        # Project high-dimensional transformer representations
        self.emb_proj = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.sw_emb = nn.Embedding(2 * max_dist + 1, 16)
        self.sw_mlp = nn.Sequential(
            nn.Linear(128 * 16, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.lex_proj = nn.Sequential(
            nn.Linear(lex_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # Cross-feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64 + 128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # Regressor head
        self.head = nn.Linear(64, 2)
        # Learnable gating for direct lexicon prior — higher initial values give
        # the opinion-specific lexicon prior stronger influence at the start of training
        # so the model learns to respect "awesome" vs "slow" distinction quickly.
        self.lex_gate = nn.Parameter(torch.tensor([0.85, 0.70]))

    def forward(self, embs: torch.Tensor, switch_ids: torch.Tensor, lex_feats: torch.Tensor):
        B = embs.shape[0]
        e = self.emb_proj(embs)
        s = self.sw_mlp(self.sw_emb(switch_ids).view(B, -1))
        l = self.lex_proj(lex_feats)

        feat = self.fusion(torch.cat([e, s, l], dim=-1))
        delta = self.head(feat)

        # Opinion affect prior (cols 0, 1 are opinion valence and arousal)
        prior = lex_feats[:, :2]
        gate = torch.sigmoid(self.lex_gate)

        # The prior has a stronger pull (gate * 4.0 -> 6.0) so the model respects
        # the opinion lexicon polarity more aggressively
        preds = torch.sigmoid(delta + (prior - 0.5) * gate * 6.0)

        return {
            "valence": preds[:, 0],
            "arousal": preds[:, 1],
            "predictions": preds,
        }
