"""Gated-attention multiple-instance-learning head (Ilse et al., 2018).

A patient is a "bag" of slice vectors h_1..h_S (from the frozen encoder):

    g_i = tanh(V h_i) * sigmoid(U h_i)        gated slice representation
    a   = softmax_i(w^T g_i)                  one attention weight per slice (padding masked out)
    z   = sum_i a_i h_i                       patient vector
    p   = sigmoid(classifier(z))              IDH-mutant score

Only this head is trained (~100k parameters); DINOv2 stays frozen.
"""

import torch
import torch.nn as nn


class GatedAttentionMIL(nn.Module):
    def __init__(self, dimension=384, attention_dim=128, dropout=0.3):
        super().__init__()
        self.feature_norm = nn.LayerNorm(dimension)
        self.attention_v = nn.Linear(dimension, attention_dim)
        self.attention_u = nn.Linear(dimension, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(dimension, 1))

    def forward(self, features, mask, return_attention=False):
        """features: (B, S, D) zero-padded bags; mask: (B, S) True for real slices. Returns logits (B,)."""
        h = self.feature_norm(features)
        gates = torch.tanh(self.attention_v(h)) * torch.sigmoid(self.attention_u(h))
        scores = self.attention_w(gates).squeeze(-1).masked_fill(~mask, -torch.inf)
        attention = torch.softmax(scores, dim=1)
        pooled = (h * attention.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        return (logits, attention) if return_attention else logits
