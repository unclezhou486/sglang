"""Candidate scoring adapters for DSpark tree construction.

The vllm-ascend MarkovScorer calls ``model.markov_embed`` / ``markov_bias`` /
``map_draft_to_target``. sglang's DSpark head exposes the equivalent stateless
operation as ``step_bias(prev_tokens, hidden_states, state)``; this adapter
bridges the two without changing either model.
"""

from dataclasses import dataclass

import torch


def select_candidates(
    logits: torch.Tensor, width: int, *, keep_greedy: bool = False
):
    """Top-``width`` full-vocabulary edges with normalized log probabilities."""
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    scores, ids = log_probs.topk(width, dim=-1)
    greedy_child = None
    if keep_greedy:
        greedy = logits.argmax(-1)
        present = (ids == greedy.unsqueeze(-1)).any(-1)
        ids[..., -1] = torch.where(present, ids[..., -1], greedy)
        scores[..., -1] = log_probs.gather(-1, ids[..., -1:])[..., 0]
        greedy_child = (ids == greedy.unsqueeze(-1)).long().argmax(-1)
    return scores, ids, greedy_child


@dataclass
class CandidateLayer:
    scores: torch.Tensor
    token_ids: torch.Tensor
    greedy_child: torch.Tensor | None = None


class IndependentScorer:
    """Use per-depth base logits as-is (no Markov correction)."""

    name = "independent"
    depth_penalty = 0.0

    def __init__(self, topk: int):
        self.topk = int(topk)

    def score(
        self,
        base_logits: torch.Tensor,
        frontier_tokens: torch.Tensor,
        width: int,
        *,
        keep_greedy: bool = False,
    ) -> CandidateLayer:
        logits = base_logits.unsqueeze(1)
        scores, ids, greedy = select_candidates(logits, width, keep_greedy=keep_greedy)
        parents = frontier_tokens.shape[1]
        scores = scores.expand(-1, parents, -1)
        ids = ids.expand(-1, parents, -1)
        if greedy is not None:
            greedy = greedy.expand(-1, parents)
        return CandidateLayer(scores, ids, greedy)


class DSparkMarkovScorer:
    """Apply the DSpark Markov bias per frontier token, then select top-k."""

    name = "markov"
    depth_penalty = 0.0

    def __init__(self, head, topk: int):
        if getattr(head, "vocab_shard", None) is not None:
            raise NotImplementedError(
                "DSpark tree mode does not support TP-sharded markov_w2 "
                "(SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD=1); unset it or use topk=1."
            )
        self.head = head
        self.topk = int(topk)

    def score(
        self,
        base_logits: torch.Tensor,
        frontier_tokens: torch.Tensor,
        width: int,
        *,
        keep_greedy: bool = False,
    ) -> CandidateLayer:
        bias, state = self.head.step_bias(
            frontier_tokens, hidden_states=None, state=None
        )
        if state is not None:
            raise NotImplementedError(
                "DSpark tree mode supports stateless markov heads only; "
                "the head returned state (Gated/RNN markov)."
            )
        if bias.shape[-1] != base_logits.shape[-1]:
            raise ValueError(
                "markov bias width != base logits width; "
                "this usually means a TP-sharded vocab was left enabled."
            )
        logits = base_logits.unsqueeze(1) + bias
        scores, ids, greedy = select_candidates(logits, width, keep_greedy=keep_greedy)
        return CandidateLayer(scores, ids, greedy)
