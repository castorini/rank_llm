from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def lambdarank(
    y_pred,
    y_true=None,
    eps=1e-10,
    padded_value_indicator=-100,
    weighing_scheme="ndcgLoss2_scheme",
    k=None,
    sigma=1.0,
    mu=10.0,
    reduction="mean",
    reduction_log="binary",
):
    """
    LambdaLoss framework for LTR losses implementations, introduced in "The LambdaLoss Framework for Ranking Metric Optimization".
    Contains implementations of different weighing schemes corresponding to e.g. LambdaRank or RankNet.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param weighing_scheme: a string corresponding to a name of one of the weighing schemes
    :param k: rank at which the loss is truncated
    :param sigma: score difference weight used in the sigmoid function
    :param mu: optional weight used in NDCGLoss2++ weighing scheme
    :param reduction: losses reduction method, could be either a sum or a mean
    :param reduction_log: logarithm variant used prior to masking and loss reduction, either binary or natural
    :return: loss value, a torch.Tensor
    """
    if y_true is None:
        y_true = torch.zeros_like(y_pred).to(y_pred.device)
        y_true[:, 0] = 1

    device = y_pred.device

    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)

    if weighing_scheme != "ndcgLoss1_scheme":
        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

    ndcg_at_k_mask = torch.zeros(
        (y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device
    )
    ndcg_at_k_mask[:k, :k] = 1

    # clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.0)
    y_true_sorted.clamp_(min=0.0)

    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1.0 + pos_idxs.float())[None, :]
    maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(
        min=eps
    )
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    if weighing_scheme is None:
        weights = 1.0
    else:
        weights = globals()[weighing_scheme](G, D, mu, true_sorted_by_preds)  # type: ignore

    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(
        min=-1e8, max=1e8
    )
    scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.0)
    weighted_probas = (
        torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights
    ).clamp(min=eps)
    if reduction_log == "natural":
        losses = torch.log(weighted_probas)
    elif reduction_log == "binary":
        losses = torch.log2(weighted_probas)
    else:
        raise ValueError("Reduction logarithm base can be either natural or binary")

    if reduction == "sum":
        loss = -torch.sum(losses[padded_pairs_mask & ndcg_at_k_mask])
    elif reduction == "mean":
        loss = -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])
    else:
        raise ValueError("Reduction method can be either sum or mean")

    return loss


def ndcgLoss1_scheme(G, D, *args):
    return (G / D)[:, :, None]


def ndcgLoss2_scheme(G, D, *args):
    pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
    delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
    deltas = torch.abs(
        torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.0)
        - torch.pow(torch.abs(D[0, delta_idxs]), -1.0)
    )
    deltas.diagonal().zero_()

    return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])


def rank_net(
    y_pred,
    y_true,
    weighted=False,
    use_rank=False,
    weight_by_diff=False,
    weight_by_diff_powed=False,
):
    """
    RankNet loss introduced in "Learning to Rank using Gradient Descent".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param weight_by_diff: flag indicating whether to weight the score differences by ground truth differences.
    :param weight_by_diff_powed: flag indicating whether to weight the score differences by the squared ground truth differences.
    :return: loss value, a torch.Tensor
    """
    if use_rank is None:
        y_true = torch.tensor(
            [[1 / (np.argsort(y_true)[::-1][i] + 1) for i in range(y_pred.size(1))]]
            * y_pred.size(0)
        ).cuda()

    document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, document_pairs_candidates]
    selected_pred = y_pred[:, document_pairs_candidates]

    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    pred_diffs = pred_diffs[the_mask]

    weight = None
    if weighted:
        values, indices = torch.sort(y_true, descending=True)
        ranks = torch.zeros_like(indices)
        ranks.scatter_(
            1,
            indices,
            torch.arange(1, y_true.numel() + 1).to(y_true.device).view_as(indices),
        )
        pairs_ranks = ranks[:, document_pairs_candidates]
        rank_sum = pairs_ranks.sum(-1)
        weight = 1 / rank_sum[the_mask]  # Relevance Feedback
    else:
        if weight_by_diff:
            abs_diff = torch.abs(true_diffs)
            weight = abs_diff[the_mask]
        elif weight_by_diff_powed:
            true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(
                pairs_true[:, :, 1], 2
            )
            abs_diff = torch.abs(true_pow_diffs)
            weight = abs_diff[the_mask]

    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return nn.BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)


class ADRMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-10):
        super().__init__()
        self.eps = eps

    def forward(
        self, scores: torch.Tensor, reranked_target_loss: float = None
    ) -> torch.Tensor:
        """
        Compute the Approx Discounted Rank MSE (ADR-MSE) loss.
        :param scores: Tensor of shape [batch_size, slate_length] containing scores for each passage.
        :param reranked_target_loss: An additional parameter that is ignored in the computation.
        :return: Scalar tensor representing the ADR-MSE loss.
        """
        batch_size, slate_length = scores.size()

        softmax_scores = torch.softmax(scores, dim=1)
        approx_ranks = torch.cumsum(softmax_scores, dim=1)

        sorted_indices = torch.argsort(scores, dim=1, descending=True)
        ranks = torch.argsort(sorted_indices, dim=1) + 1

        log_discounts = torch.log2(ranks.float() + 1)

        rank_diffs = (ranks.float() - approx_ranks) ** 2

        discounted_diffs = rank_diffs / log_discounts

        loss = discounted_diffs.mean()
        return loss


def listNet(y_pred, y_true, eps=1e-10, padded_value_indicator=-1):
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float("-inf")
    y_true[mask] = float("-inf")

    preds_smax = F.softmax(y_pred, dim=1)
    true_smax = F.softmax(y_true, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)

    return torch.mean(-torch.sum(true_smax * preds_log, dim=1))


loss_dict = {
    "lambdarank": lambdarank,
    "ranknet": rank_net,
    "listnet_loss": listNet,
    "adr_mse_loss": ADRMSELoss(),
}
