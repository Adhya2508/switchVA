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

# ==============================================================================
# Hinglish Polarity & Affect Lexicon (NRC-VAD + Domain Calibrated)
# ==============================================================================
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
    "strong": (0.75, 0.60), "perfect": (0.90, 0.70), "solid": (0.80, 0.60), "top": (0.85, 0.65),
    "helpful": (0.78, 0.52), "cooperative": (0.80, 0.50), "positive": (0.80, 0.55), "sweet": (0.82, 0.50),
    "smooth": (0.80, 0.48), "crisp": (0.80, 0.55), "fresh": (0.82, 0.58),
    
    # Low Valence / Negative
    "bakwas": (0.12, 0.85), "bekar": (0.15, 0.75), "kharab": (0.15, 0.80), "ganda": (0.12, 0.82),
    "bura": (0.18, 0.75), "buri": (0.18, 0.75), "puri": (0.15, 0.70), "fail": (0.12, 0.80),
    "scam": (0.18, 0.82), "dhoka": (0.10, 0.88), "fraud": (0.10, 0.88), "terrorist": (0.05, 0.92),
    "bhikari": (0.10, 0.85), "haram": (0.08, 0.90), "galat": (0.22, 0.70), "laprvahi": (0.12, 0.80),
    "fuck": (0.08, 0.90), "sad": (0.25, 0.45), "disappointing": (0.18, 0.68), "horrible": (0.10, 0.85),
    "terrible": (0.10, 0.85), "waste": (0.15, 0.75), "issue": (0.25, 0.65), "problem": (0.22, 0.68),
    "weak": (0.28, 0.55), "slow": (0.30, 0.50), "hanging": (0.18, 0.75), "lag": (0.20, 0.72),
    "heating": (0.20, 0.75), "lannat": (0.12, 0.85), "hate": (0.10, 0.88), "mulle": (0.15, 0.80),
    "marne": (0.12, 0.90), "gira": (0.18, 0.75), "mushkil": (0.25, 0.65), "chhed": (0.18, 0.78),
    "chori": (0.10, 0.85), "danga": (0.12, 0.88), "dango": (0.12, 0.88), "mahangai": (0.22, 0.75),
    "chamcho": (0.15, 0.78), "kaid": (0.15, 0.75), "badh": (0.18, 0.75), "moun": (0.25, 0.60),
    "threat": (0.10, 0.90), "bad": (0.18, 0.70), "worst": (0.10, 0.85), "cheat": (0.12, 0.85),
    "high": (0.35, 0.60), "poor": (0.18, 0.65), "tough": (0.35, 0.65), "expensive": (0.28, 0.60),
    "boring": (0.20, 0.35), "rude": (0.15, 0.75), "drain": (0.22, 0.65),
}


def extract_token_lexicon_features(words, max_len=MAX_LEN):
    """
    Constructs [max_len, 3] matrix of (Valence, Arousal, IsLexiconHit) for each token node.
    """
    feats = np.zeros((max_len, 3), dtype=np.float32)
    feats[:, 0] = 0.5  # Neutral default V
    feats[:, 1] = 0.5  # Neutral default A
    feats[:, 2] = 0.0  # Not a lexicon hit
    
    for i, w in enumerate(words[:max_len]):
        w_clean = re.sub(r'[^\w]', '', w.lower())
        if w_clean in HINGLISH_AFFECT_LEXICON:
            v, a = HINGLISH_AFFECT_LEXICON[w_clean]
            feats[i, 0] = v
            feats[i, 1] = a
            feats[i, 2] = 1.0
    return torch.tensor(feats, dtype=torch.float)


def construct_heterogeneous_adj(words, languages, aspect_words, opinion_words, max_len=MAX_LEN):
    """
    Constructs [3, max_len, max_len] multi-relational adjacency tensor:
      Relation 0: Syntactic Dependency Edges (Window adjacency k=2 + Self-loops)
      Relation 1: Code-Switch Transition Bridges (Edges between tokens at language switch points)
      Relation 2: Semantic Aspect-Opinion Links (Bipartite edges between aspect & opinion tokens)
    """
    N = min(len(words), max_len)
    adj = np.zeros((3, max_len, max_len), dtype=np.float32)
    
    # 0. Syntactic / Local Dependency (window size k=2 + self-loops)
    for i in range(N):
        adj[0, i, i] = 1.0  # self-loop
        for j in range(max(0, i - 2), min(N, i + 3)):
            adj[0, i, j] = 1.0
            adj[0, j, i] = 1.0
            
    # 1. Code-Switch Transition Bridges
    for i in range(N):
        for j in range(i + 1, min(N, i + 4)):
            if i < len(languages) and j < len(languages):
                if languages[i] != languages[j]:
                    adj[1, i, j] = 1.0
                    adj[1, j, i] = 1.0
                    
    # 2. Semantic Aspect-Opinion Links
    asp_indices = [i for i, w in enumerate(words[:N]) if w.lower() in [aw.lower() for aw in aspect_words]]
    op_indices = [i for i, w in enumerate(words[:N]) if w.lower() in [ow.lower() for ow in opinion_words]]
    for ai in asp_indices:
        for oi in op_indices:
            adj[2, ai, oi] = 1.0
            adj[2, oi, ai] = 1.0
            
    return torch.tensor(adj, dtype=torch.float)


# ==============================================================================
# NSSG-DimNet Architecture Modules (Vectorized & Highly Optimized)
# ==============================================================================

class SwitchEmbeddingBlock(nn.Module):
    """
    Switch Embedding Block: Embeds token language switch distances into continuous dense vectors.
    """
    def __init__(self, max_distance: int = MAX_DISTANCE, embed_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(2 * max_distance + 1, embed_dim)

    def forward(self, switch_ids: torch.Tensor):
        return self.embed(switch_ids)


class SwitchPointGatedAttentionUnit(nn.Module):
    """
    Switch-Point Gated Attention Unit:
    Modulates multilingual contextual token representations at language switch points.
    """
    def __init__(self, hidden_dim: int = HIDDEN_SIZE, switch_dim: int = 64):
        super().__init__()
        self.proj_sw = nn.Linear(switch_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, H: torch.Tensor, switch_embed: torch.Tensor):
        sw_p = self.proj_sw(switch_embed)
        g = self.gate(torch.cat([H, sw_p], dim=-1))
        return self.norm(H + g * H)


class BiaffineSpanExtractor(nn.Module):
    """
    Branch 1: Biaffine Span-Pair Extractor
    Scores candidate aspect and opinion spans and pools targeted aspect-opinion pair representations.
    """
    def __init__(self, hidden_dim: int = HIDDEN_SIZE, span_dim: int = 128):
        super().__init__()
        self.mlp_start = nn.Sequential(nn.Linear(hidden_dim, span_dim), nn.GELU())
        self.mlp_end = nn.Sequential(nn.Linear(hidden_dim, span_dim), nn.GELU())
        self.U_asp = nn.Linear(span_dim, span_dim, bias=False)
        self.U_op = nn.Linear(span_dim, span_dim, bias=False)
        self.span_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, span_dim),
            nn.LayerNorm(span_dim),
            nn.GELU(),
        )

    def forward_span_scores(self, H: torch.Tensor):
        hs = self.mlp_start(H)
        he = self.mlp_end(H)
        asp_scores = torch.bmm(self.U_asp(hs), he.transpose(1, 2))
        op_scores = torch.bmm(self.U_op(hs), he.transpose(1, 2))
        return asp_scores, op_scores

    def pool_aspect_opinion_pair(self, H: torch.Tensor, asp_mask: torch.Tensor, op_mask: torch.Tensor):
        B, N, D = H.shape
        asp_exp = asp_mask.unsqueeze(-1)
        op_exp = op_mask.unsqueeze(-1)
        h_asp = (H * asp_exp).sum(dim=1) / asp_exp.sum(dim=1).clamp(min=1)
        h_op = (H * op_exp).sum(dim=1) / op_exp.sum(dim=1).clamp(min=1)
        h_pair = torch.cat([h_asp, h_op, h_asp * h_op], dim=-1)
        return self.span_fuse(h_pair)  # [B, span_dim]


class HeterogeneousRelationalGAT(nn.Module):
    """
    Branch 2: Heterogeneous Neuro-Symbolic Graph Module (Relational GAT)
    Aggregates messages across 3 edge types:
      - Syntactic Dependency Edges
      - Code-Switch Transition Bridges
      - Semantic Aspect-Opinion Links
    Combined with NRC-VAD / Hinglish Affect Lexicon Emotional Priors.
    """
    def __init__(self, node_dim: int = HIDDEN_SIZE, out_dim: int = 128, num_relations: int = 3, lex_dim: int = 3):
        super().__init__()
        self.num_relations = num_relations
        self.W_r = nn.ModuleList([nn.Linear(node_dim, out_dim, bias=False) for _ in range(num_relations)])
        self.a_src = nn.ParameterList([nn.Parameter(torch.randn(out_dim, 1) * 0.02) for _ in range(num_relations)])
        self.a_dst = nn.ParameterList([nn.Parameter(torch.randn(out_dim, 1) * 0.02) for _ in range(num_relations)])
        self.lex_proj = nn.Linear(lex_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()

    def forward(self, H: torch.Tensor, adj_matrices: torch.Tensor, lex_features: torch.Tensor):
        B, N, D = H.shape
        rel_outs = []
        for r in range(self.num_relations):
            Wr_H = self.W_r[r](H)  # [B, N, out_dim]
            score_src = torch.matmul(Wr_H, self.a_src[r])  # [B, N, 1]
            score_dst = torch.matmul(Wr_H, self.a_dst[r])  # [B, N, 1]
            attn = F.leaky_relu(score_src + score_dst.transpose(1, 2), negative_slope=0.2)
            
            adj = adj_matrices[:, r]
            attn = attn.masked_fill(adj == 0, -1e9)
            alpha = F.softmax(attn, dim=-1)
            alpha = torch.nan_to_num(alpha, nan=0.0)
            out_r = torch.bmm(alpha, Wr_H)
            rel_outs.append(out_r)

        aggregated = sum(rel_outs) + self.lex_proj(lex_features)
        return self.norm(self.act(aggregated))  # [B, N, out_dim]


class AspectGuidedMutualCrossAttention(nn.Module):
    """
    Fusion Layer: Aspect-Guided Mutual Cross-Attention & Span Fusion
    Cross-attends Biaffine Aspect-Opinion Span representations with Heterogeneous Graph representations.
    """
    def __init__(self, span_dim: int = 128, graph_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.q_proj = nn.Linear(span_dim, out_dim)
        self.k_proj = nn.Linear(graph_dim, out_dim)
        self.v_proj = nn.Linear(graph_dim, out_dim)
        self.scale = np.sqrt(out_dim)
        self.fuse_mlp = nn.Sequential(
            nn.Linear(span_dim + out_dim + span_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, H_span: torch.Tensor, H_graph: torch.Tensor, node_mask: torch.Tensor):
        Q = self.q_proj(H_span).unsqueeze(1)  # [B, 1, out_dim]
        K = self.k_proj(H_graph)              # [B, N, out_dim]
        V = self.v_proj(H_graph)              # [B, N, out_dim]

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        scores = scores.masked_fill(node_mask.unsqueeze(1) == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        A_cross = torch.bmm(attn, V).squeeze(1)  # [B, out_dim]

        fused = self.fuse_mlp(torch.cat([H_span, A_cross, H_span * A_cross], dim=-1))
        return fused


class NSSGDimNet(nn.Module):
    """
    Complete NSSG-DimNet:
    Neuro-Symbolic Switch-Gated Dual-Graph Network for Hinglish DimABSA
    """
    def __init__(
        self,
        hidden_dim: int = HIDDEN_SIZE,
        switch_dim: int = 64,
        span_dim: int = 128,
        graph_dim: int = 128,
        lex_dim: int = 3,
        max_dist: int = MAX_DISTANCE,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        # Input & Encoding Layer
        self.switch_embed_block = SwitchEmbeddingBlock(max_distance=max_dist, embed_dim=switch_dim)
        self.switch_point_gated_attention = SwitchPointGatedAttentionUnit(hidden_dim=hidden_dim, switch_dim=switch_dim)

        # Dual-Branch Processing Layer
        self.biaffine_span_extractor = BiaffineSpanExtractor(hidden_dim=hidden_dim, span_dim=span_dim)
        self.hetero_graph_module = HeterogeneousRelationalGAT(node_dim=hidden_dim, out_dim=graph_dim, num_relations=3, lex_dim=lex_dim)

        # Fusion Layer
        self.mutual_cross_attention = AspectGuidedMutualCrossAttention(span_dim=span_dim, graph_dim=graph_dim, out_dim=span_dim)

        # Continuous Output Heads
        self.valence_head = nn.Sequential(
            nn.Linear(span_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.arousal_head = nn.Sequential(
            nn.Linear(span_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Neuro-symbolic emotional prior weighting gate
        self.lex_gate_v = nn.Parameter(torch.tensor([1.2]))
        self.lex_gate_a = nn.Parameter(torch.tensor([0.8]))

    def forward(
        self,
        token_embs: torch.Tensor,
        switch_ids: torch.Tensor,
        adj_matrices: torch.Tensor,
        lex_token_feats: torch.Tensor,
        asp_mask: torch.Tensor,
        op_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        opinion_lex_prior: torch.Tensor,
    ):
        # 1. Switch Positional Embedding & Switch-Point Gated Attention
        sw_emb = self.switch_embed_block(switch_ids)
        H_gated = self.switch_point_gated_attention(token_embs, sw_emb)

        # 2. Dual-Branch Processing
        # Branch 1: Biaffine Span-Pair Extractor
        H_span = self.biaffine_span_extractor.pool_aspect_opinion_pair(H_gated, asp_mask, op_mask)

        # Branch 2: Heterogeneous Relational Graph Module
        H_graph = self.hetero_graph_module(H_gated, adj_matrices, lex_token_feats)

        # 3. Fusion Layer: Aspect-Guided Mutual Cross-Attention
        H_fused = self.mutual_cross_attention(H_span, H_graph, attention_mask)

        # 4. Continuous Output Heads
        raw_v = self.valence_head(H_fused).squeeze(-1)
        raw_a = self.arousal_head(H_fused).squeeze(-1)

        # Direct Neuro-Symbolic Affect Lexicon Integration
        op_v_prior = opinion_lex_prior[:, 0]
        op_a_prior = opinion_lex_prior[:, 1]
        is_hit = opinion_lex_prior[:, 2]

        v_shift = (op_v_prior - 0.5) * torch.sigmoid(self.lex_gate_v) * 4.0 * is_hit
        a_shift = (op_a_prior - 0.5) * torch.sigmoid(self.lex_gate_a) * 3.0 * is_hit

        pred_val = torch.sigmoid(raw_v + v_shift)
        pred_aro = torch.sigmoid(raw_a + a_shift)

        return {
            "valence": pred_val,
            "arousal": pred_aro,
            "predictions": torch.stack([pred_val, pred_aro], dim=-1),
        }


# Alias for backward compatibility
AspectEmotionRegressor = NSSGDimNet
