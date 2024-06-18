from re import X
import torch
import torch.nn as nn


class VisualHead(torch.nn.Module):
    def __init__(self, num_class, input_size=256, word_emb_tab=None, top_k=5):

        super().__init__()
        self.input_size = input_size
        self.gloss_output_layer = nn.Linear(input_size, num_class)
        self.word_emb_tab = word_emb_tab.float()
        self.word_emb_dim = 300
        self.word_emb_mapper = nn.Linear(self.word_emb_dim, input_size)
        self.top_k = top_k
        self.word_fused_gloss_output_layer = nn.Linear(input_size, num_class)

    def forward(self, x, label=None, is_training=False):

        # x is visual feature with shape (256,)
        B, C = x.shape[0], x.shape[-1]
        f = x       # f = visual_fea
        # Branch of Language-Aware Label Smoothing
        # Maping visual feature (256,) to (num_class,)
        output_fc1 = self.gloss_output_layer(x)  # [B,N]

        # Branch of Inter-Modality Feature

        # # Select top_k label predicted in logits
        # if self.training:
        #     logits_data = logits.clone().detach()
        #     batch_idx = torch.arange(B)
        #     logits_data[batch_idx, labels] = float('-inf')
        #     idx = torch.argsort(logits_data, dim=1, descending=True)[
        #         :, :self.top_k-1]  # [B,K-1]
        #     topk_idx = torch.cat([labels.unsqueeze(1), idx], dim=1)  # [B,K]
        # else:
        #     topk_idx = torch.argsort(logits, dim=1, descending=True)[:, :5]
        # topk_idx = topk_idx.reshape(-1)

        # # Mapping word embedding (num_class,300) to (num_class, visual_feature)
        # e_ = self.word_emb_mapper(self.word_emb_tab)
        # word_embs = e_.index_select(0, topk_idx).reshape(B, -1, C)  # [B,K,C]
        # fused_fea = x.unsqueeze(1) + word_embs
        # word_fused_gloss_logits = self.word_fused_gloss_output_layer(fused_fea)
        # FC2 = {"word_fused_gloss_logits": word_fused_gloss_logits,
        #        "top_idx": topk_idx}

        output_fc2 = None
        topk_idx = None
        if is_training:
            logits_data = output_fc1.clone().detach()
            batch_idx = torch.arange(B)
            logits_data[batch_idx, label] = float('-inf')
            idx = torch.argsort(logits_data, dim=1, descending=True)[
                :, :self.top_k-1]  # [B,K-1]
            topk_idx = torch.cat([label.unsqueeze(1), idx], dim=1)  # [B,K]
            topk_idx = topk_idx.reshape(-1)
            # Mapping word embedding (num_class,300) to (num_class, visual_feature)
            # shape e_ = (num_class ,visual_feature)
            e_ = self.word_emb_mapper(self.word_emb_tab)
            e_ = e_.index_select(0, topk_idx).reshape(
                B, -1, C)  # [batch_size, top_k, visual_fea]
            # shape f = (batch_size, 1, visual_fea)
            f = f.unsqueeze(1)
            # shape F = (batch_size, top_k, visual_fea)
            F = f + e_
            # shape output_fc2 = (batch_size, num_class, num_class)
            output_fc2 = self.word_fused_gloss_output_layer(F)
            topk_idx = topk_idx.reshape(B, self.top_k)
        return output_fc1, output_fc2, topk_idx
