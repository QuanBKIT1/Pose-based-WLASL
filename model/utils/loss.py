import torch as torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothCE(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, epsilon=0.2, reduction='mean', word_emb_tab=None, t=0.5):
        super(LabelSmoothCE, self).__init__()

        self.epsilon = epsilon  # epsilon=0.2
        self.reduction = reduction  # mean
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.word_emb_tab = None
        if word_emb_tab is not None:
            self.word_emb_sim = torch.matmul(F.normalize(
                word_emb_tab, dim=-1), F.normalize(word_emb_tab, dim=-1).T)     # (2000,300) * (300,2000) = (2000,2000)
            self.t = t        # (0.5)

    def forward(self, logits, labels):
        # Get logits from FC1 with shape (N, 300,)
        # Label is integer (N,), convert it to shape (N, 300,)
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothCE()
            >>> logits = torch.randn(64, 300)
            >>> lbs = torch.randint(64) 
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float()  # use fp32 to avoid nan

        assert self.word_emb_sim is not None
        # Get word embedding similar
        soft_labels = self.word_emb_sim[labels]
        idx = torch.arange(labels.shape[0])

        soft_labels[idx, labels] = float('-inf')
        soft_labels /= self.t
        soft_labels = F.softmax(soft_labels, dim=-1)
        soft_labels *= self.epsilon

        # y[label] = 1 - epsilon
        soft_labels[idx, labels] = 1.0-self.epsilon
        soft_labels = soft_labels.detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * soft_labels, dim=-1)

        if self.reduction == 'mean':
            loss = loss.sum() / labels.shape[0]
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss
