from re import X
import torch
import torch.nn as nn
import numpy as np
from .utils.loss import LabelSmoothCE


class VisualHead(torch.nn.Module):
    def __init__(self, num_class, input_size=256, word_emb_tab=None, head_args=dict()):

        super().__init__()
        self.input_size = input_size
        self.gloss_output_layer = nn.Linear(input_size, num_class)
        self.word_emb_tab = word_emb_tab
        self.word_emb_dim = 300
        self.word_emb_mapper = nn.Linear(self.word_emb_dim, input_size)
        self.label_smooth = LabelSmoothCE(
            **head_args, word_emb_tab=word_emb_tab)

    def forward(self, x, labels):

        # interact with word_embs
        logits = self.gloss_output_layer(x)  # [B,N]
        output = self.label_smooth(logits, labels)
        return output
