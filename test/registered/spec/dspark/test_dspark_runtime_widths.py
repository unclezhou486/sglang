"""CPU tests for the DSpark runtime gamma / verify-width split."""

from unittest.mock import patch

import pytest

from sglang.srt.speculative.dspark_components.dspark_config import (
    DSparkDraftConfig,
    resolve_runtime_config,
)


def _draft_config(*, gamma=5):
    return DSparkDraftConfig(
        num_hidden_layers=1,
        num_target_layers=1,
        gamma=gamma,
        target_layer_ids=[1],
        mask_token="<mask>",
        mask_token_id=10,
        markov_rank=256,
        markov_head_type="vanilla",
    )


def _resolve(**kwargs):
    with patch(
        "sglang.srt.speculative.dspark_components.dspark_config."
        "parse_dspark_draft_config",
        return_value=_draft_config(),
    ):
        return resolve_runtime_config(
            draft_hf_config=object(),
            target_vocab_size=100,
            **kwargs,
        )


def test_tree_keeps_gamma_and_budget_width():
    cfg = _resolve(
        speculative_num_draft_tokens=9,
        gamma=5,
        verify_num_draft_tokens=9,
    )
    assert cfg.gamma == 5
    assert cfg.verify_num_draft_tokens == 9


def test_non_tree_keeps_gamma_plus_one():
    cfg = _resolve(
        speculative_num_draft_tokens=6,
        gamma=5,
        verify_num_draft_tokens=6,
    )
    assert cfg.gamma == 5
    assert cfg.verify_num_draft_tokens == 6


def test_tree_can_infer_gamma_from_checkpoint():
    cfg = _resolve(
        speculative_num_draft_tokens=9,
        verify_num_draft_tokens=9,
    )
    assert cfg.gamma == 5
    assert cfg.verify_num_draft_tokens == 9


def test_verify_width_below_gamma_plus_one_is_rejected():
    with pytest.raises(ValueError, match="verify width"):
        _resolve(
            speculative_num_draft_tokens=4,
            gamma=5,
            verify_num_draft_tokens=4,
        )
