"""
Deep Cognitive Graph Topology Refiner (Graph Neural Network).

Performs relational message-passing across incident road segments:
- Evaluates topological flow conservation (roads that enter an intersection must continue);
- Eliminates isolated diagonal chords and spurious cross-block webbings;
- Reinforces cyclic roundabouts, grid corridors, and serpentine mountain curves;
- Replaces rigid manual heuristics with pure learnable graph attention.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphTopologyRefiner(nn.Module):
    """
    Relational Graph Neural Network refining candidate road edges via message passing.
    """

    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim

        # Edge feature transformation
        self.edge_proj = nn.Linear(embed_dim, embed_dim)

        # Multi-Head Graph Attention between incident edges
        self.gat1 = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.gat2 = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm4 = nn.LayerNorm(embed_dim)

        # Final refined probability head
        self.prob_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(
        self,
        edge_tokens: torch.Tensor,
        candidate_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        edge_tokens : torch.Tensor
            (N, embed_dim) pooled contextual embeddings for each candidate edge from Transformer.
        candidate_indices : torch.Tensor
            (N, 2) integer pairs (u, v) indicating endpoint node indices.

        Returns
        -------
        refined_probs : torch.Tensor
            (N, 1) topologically refined edge existence probabilities in [0, 1].
        """
        N = edge_tokens.shape[0]
        if N <= 1:
            logits = self.prob_head(edge_tokens)
            return torch.sigmoid(logits)

        # Construct Line-Graph Adjacency Mask (incident edges sharing at least one node)
        u = candidate_indices[:, 0].unsqueeze(1)  # (N, 1)
        v = candidate_indices[:, 1].unsqueeze(1)  # (N, 1)

        # Two edges i and j share a node if u_i == u_j, u_i == v_j, v_i == u_j, or v_i == v_j
        u_eq_u = (u == u.T)
        u_eq_v = (u == v.T)
        v_eq_u = (v == u.T)
        v_eq_v = (v == v.T)
        adj_mask = (u_eq_u | u_eq_v | v_eq_u | v_eq_v)  # (N, N) bool

        # Convert to additive attention mask (0 for connected, -inf for non-incident)
        # We allow self-attention plus incident edges
        attn_mask = torch.where(adj_mask, 0.0, float("-1e4"))

        # Layer 1 Message Passing
        h = self.edge_proj(edge_tokens).unsqueeze(0)  # (1, N, embed_dim)
        attn_out, _ = self.gat1(h, h, h, attn_mask=attn_mask)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ffn1(h))

        # Layer 2 Message Passing
        attn_out2, _ = self.gat2(h, h, h, attn_mask=attn_mask)
        h = self.norm3(h + attn_out2)
        h = self.norm4(h + self.ffn2(h))

        h = h.squeeze(0)  # (N, embed_dim)
        logits = self.prob_head(h)
        return torch.sigmoid(logits)
