import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from backend.config import (
    MAX_LEN,
    MAX_DISTANCE,
    HIDDEN_SIZE,
    NUM_HEADS,
    NUM_RELATIONS,
    RGAT_LAYERS,
    MODEL_NAME,
    DEVICE,
)


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
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor, switch_ids: torch.Tensor):
        # hidden_states: [B, N, H]
        # switch_ids: [B, N]
        switch_embed = self.switch_embedding(switch_ids)  # [B, N, H]
        concat = torch.cat([hidden_states, switch_embed], dim=-1)  # [B, N, 2H]
        gate = torch.sigmoid(self.gate(concat))  # [B, N, H]
        output = hidden_states + gate * hidden_states
        return output, gate


class BiaffineSpanExtractor(nn.Module):
    """
    Biaffine Span Extractor for Aspect & Opinion Span scoring.
    Computes upper-triangular span representation matrix [B, N, N]
    using bilinear parameter tensor U.
    """

    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.start = nn.Linear(hidden_size, hidden_size)
        self.end = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.U)

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: [B, N, H]
        start = self.start(hidden_states)  # [B, N, H]
        end = self.end(hidden_states)  # [B, N, H]
        # scores: [B, N, N] where scores[b, i, j] is confidence of span from token i to token j
        scores = torch.einsum("bih,hk,bjk->bij", start, self.U, end)
        return scores


def build_batch_graph(
    words_batch: list,
    switch_ids_batch: torch.Tensor,
    aspect_matrix: torch.Tensor = None,
    opinion_matrix: torch.Tensor = None,
    max_len: int = MAX_LEN,
    device: torch.device = DEVICE,
):
    """
    Constructs 4-Relational Graph Adjacency matrix and Relation Type tensor:
      Relation 0: Sequential edges (i <-> i+1)
      Relation 1: Self loop (i <-> i)
      Relation 2: Language Switch edges (i <-> i+1 where switch_ids change)
      Relation 3: Aspect -> Opinion cross-edges
    """
    B = len(words_batch)
    adjacency = torch.zeros(B, max_len, max_len, device=device)
    relation = torch.zeros(B, max_len, max_len, dtype=torch.long, device=device)

    for b in range(B):
        words = words_batch[b]
        n = min(len(words), max_len)

        # Relation 0: Sequential edges
        for i in range(n - 1):
            adjacency[b, i, i + 1] = 1
            adjacency[b, i + 1, i] = 1
            relation[b, i, i + 1] = 0
            relation[b, i + 1, i] = 0

        # Relation 1: Self loop
        for i in range(n):
            adjacency[b, i, i] = 1
            relation[b, i, i] = 1

        # Relation 2: Switch edges
        switch = switch_ids_batch[b].cpu().tolist()
        for i in range(n - 1):
            if i < len(switch) - 1 and switch[i] != switch[i + 1]:
                adjacency[b, i, i + 1] = 1
                adjacency[b, i + 1, i] = 1
                relation[b, i, i + 1] = 2
                relation[b, i + 1, i] = 2

        # Relation 3: Aspect -> Opinion cross edges
        if aspect_matrix is not None and opinion_matrix is not None:
            aspect = aspect_matrix[b]
            opinion = opinion_matrix[b]
            aspect_pos = torch.nonzero(aspect)
            opinion_pos = torch.nonzero(opinion)
            k = min(len(aspect_pos), len(opinion_pos))
            for i in range(k):
                a_start = aspect_pos[i][0].item()
                o_start = opinion_pos[i][0].item()
                if a_start < max_len and o_start < max_len:
                    adjacency[b, a_start, o_start] = 1
                    adjacency[b, o_start, a_start] = 1
                    relation[b, a_start, o_start] = 3
                    relation[b, o_start, a_start] = 3

    return adjacency, relation


class RGATLayer(nn.Module):
    """
    Relational Graph Attention Layer parameterized by relation type embeddings.
    """

    def __init__(self, hidden_size: int = HIDDEN_SIZE, num_relations: int = NUM_RELATIONS):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.rel_embedding = nn.Embedding(num_relations, hidden_size)
        self.attn = nn.Linear(hidden_size * 3, 1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, relation: torch.Tensor):
        B, N, H = x.shape
        h = self.linear(x)
        output = torch.zeros_like(h)

        for i in range(N):
            hi = h[:, i].unsqueeze(1).expand(-1, N, -1)  # [B, N, H]
            hj = h  # [B, N, H]
            rel = self.rel_embedding(relation[:, i])  # [B, N, H]
            pair = torch.cat([hi, hj, rel], dim=-1)  # [B, N, 3H]
            score = self.attn(pair).squeeze(-1)  # [B, N]
            score = score.masked_fill(adjacency[:, i] == 0, -1e9)
            alpha = torch.softmax(score, dim=-1)  # [B, N]
            output[:, i] = torch.bmm(alpha.unsqueeze(1), hj).squeeze(1)

        return F.relu(output)


class RGATNetwork(nn.Module):
    """
    Multi-layer Relational Graph Attention Network.
    """

    def __init__(self, hidden_size: int = HIDDEN_SIZE, layers: int = RGAT_LAYERS, num_relations: int = NUM_RELATIONS):
        super().__init__()
        self.layers = nn.ModuleList(
            [RGATLayer(hidden_size, num_relations) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, relation: torch.Tensor):
        for layer in self.layers:
            x = layer(x, adjacency, relation)
        return x


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Layer between Transformer representations and Relational Graph representations.
    """

    def __init__(self, hidden_size: int = HIDDEN_SIZE, heads: int = NUM_HEADS):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, transformer_features: torch.Tensor, graph_features: torch.Tensor):
        fused, _ = self.attention(
            query=transformer_features, key=graph_features, value=graph_features
        )
        fused = self.norm(fused + transformer_features)
        return fused


class RegressionHead(nn.Module):
    """
    Continuous Multi-Layer Perceptron Regression Head for Valence and Arousal (range [0, 1]).
    """

    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class DimABSAModel(nn.Module):
    """
    Full End-to-End DimABSA Model Architecture.
    Combines HingRoBERTa + SP-GSA + Biaffine Span Extractors + 4-Relational RGAT + Cross Attention + Valence/Arousal Regressors.
    """

    def __init__(
        self,
        transformer_model: AutoModel = None,
        hidden_size: int = HIDDEN_SIZE,
        max_distance: int = MAX_DISTANCE,
        num_relations: int = NUM_RELATIONS,
        rgat_layers: int = RGAT_LAYERS,
        num_heads: int = NUM_HEADS,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.transformer = transformer_model

        # SP-GSA
        self.spgsa = SwitchGatedSelfAttention(hidden_size, max_distance)

        # Span Heads
        self.aspect_head = BiaffineSpanExtractor(hidden_size)
        self.opinion_head = BiaffineSpanExtractor(hidden_size)

        # RGAT Network
        self.rgat = RGATNetwork(hidden_size, layers=rgat_layers, num_relations=num_relations)

        # Cross-Attention Fusion
        self.fusion = CrossAttentionFusion(hidden_size, heads=num_heads)

        # Regression Heads
        self.valence_head = RegressionHead(hidden_size)
        self.arousal_head = RegressionHead(hidden_size)

    def forward(self, batch: dict):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        switch_ids = batch["switch_ids"]
        words = batch["words"]
        aspect_matrix = batch.get("aspect_matrix", None)
        opinion_matrix = batch.get("opinion_matrix", None)

        device = input_ids.device

        # 1. Contextual Backbone
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, N, H]

        # 2. Switch-Gated Self Attention (SP-GSA)
        hidden, gate = self.spgsa(hidden, switch_ids)

        # 3. Span Predictions (Biaffine)
        aspect_scores = self.aspect_head(hidden)  # [B, N, N]
        opinion_scores = self.opinion_head(hidden)  # [B, N, N]

        # 4. 4-Relational Graph Construction
        adjacency, relation = build_batch_graph(
            words_batch=words,
            switch_ids_batch=switch_ids,
            aspect_matrix=aspect_matrix,
            opinion_matrix=opinion_matrix,
            max_len=hidden.shape[1],
            device=device,
        )

        # 5. Relational Graph Attention Network (RGAT)
        graph_embeddings = self.rgat(hidden, adjacency, relation)

        # 6. Cross-Attention Fusion
        fused = self.fusion(hidden, graph_embeddings)

        # 7. Sentence CLS Representation
        cls_rep = fused[:, 0]

        # 8. Continuous Valence & Arousal Predictions
        valence = self.valence_head(cls_rep).squeeze(-1)
        arousal = self.arousal_head(cls_rep).squeeze(-1)

        return {
            "aspect_scores": aspect_scores,
            "opinion_scores": opinion_scores,
            "graph_embeddings": graph_embeddings,
            "fused_embeddings": fused,
            "valence": valence,
            "arousal": arousal,
            "gate": gate,
            "adjacency": adjacency,
            "relation": relation,
        }
