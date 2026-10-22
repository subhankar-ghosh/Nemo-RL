# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LOCAL PATCH: the S3.24/S3.26 overlong clamp and truncation-abort mask on the SC path."""

import asyncio
import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import nemo_rl.algorithms.async_utils.replay_buffer as replay_buffer_module
from nemo_rl.algorithms.async_utils.replay_buffer import (
    DataPlaneCheckpointBarrier,
    TQReplayBuffer,
)
from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GRPOAdvantageEstimator,
)
from nemo_rl.algorithms.grpo import (
    GRPOConfig,
    _apply_overlong_abort_penalty,
    _mtp_scaled_metrics,
    _overlong_clamp_metrics,
    async_grpo_train,
)
from nemo_rl.algorithms.grpo import _grpo_train_impl
from nemo_rl.algorithms.single_controller_utils.utils import aggregate_step_metrics
from nemo_rl.algorithms.truncation_abort import (
    ROW_CLAMP_RECORD_KEY,
    clamp_prompt_group,
    step_clamp_metrics,
    apply_truncation_abort_clamp,
    truncation_abort_columns,
)
from nemo_rl.experience.interfaces import Completion, PromptGroupRecord

PROMPT_LEN = 10
BIN, ROUND_TO = 64, 8  # clamp target 56
ABORT = {"reason": "repetition", "call_index": 5, "elapsed_s": 12.0}


def _row(gen_len: int) -> list[dict[str, Any]]:
    return [
        {"role": "user", "token_ids": torch.arange(PROMPT_LEN)},
        {
            "role": "assistant",
            "token_ids": torch.arange(gen_len),
            "generation_logprobs": torch.full((gen_len,), -0.5),
        },
    ]


def _record(spec: list[tuple[int, float, dict | None]]) -> PromptGroupRecord:
    return PromptGroupRecord(
        prompt_idx=0,
        prompt=[{"role": "user", "token_ids": torch.arange(PROMPT_LEN)}],
        extra_env_info={},
        metadata={},
        completions=[
            Completion(
                message_log=_row(gen_len),
                env_extras={"_ng_truncation": abort} if abort else {},
                truncated=False,
                reward=reward,
            )
            for gen_len, reward, abort in spec
        ],
        rollout_metrics={},
    )


# Group 0: ok / overlong / aborted / ok.  Group 1: overlong / ok / ok / aborted.
GROUPS = [
    [(20, 1.0, None), (100, 0.0, None), (20, 1.0, ABORT), (20, 0.0, None)],
    [(90, 0.0, None), (20, 1.0, None), (20, 1.0, None), (20, 0.0, ABORT)],
]


def test_clamp_prompt_group_truncates_and_zero_weights():
    record = _record(GROUPS[0])
    weight, row_record = clamp_prompt_group(record, BIN, ROUND_TO)

    assert weight.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert row_record["clamped"] == [False, True, False, False]
    assert row_record["truncation_abort_kind"] == [0, 0, 1, 0]
    clamped = record.completions[1].message_log[1]
    # Clamped to bin - round_to, with the per-token side tensor kept aligned.
    assert PROMPT_LEN + clamped["token_ids"].shape[-1] == BIN - ROUND_TO
    assert clamped["generation_logprobs"].shape == clamped["token_ids"].shape
    assert record.completions[0].message_log[1]["token_ids"].shape[-1] == 20


def _config(**grpo: Any) -> SimpleNamespace:
    return SimpleNamespace(
        policy={
            "logprob_batch_size": 1,
            "train_micro_batch_size": 1,
            "max_total_sequence_length": BIN,
            "make_sequence_length_divisible_by": ROUND_TO,
        },
        grpo=SimpleNamespace(num_generations_per_prompt=4, **grpo),
    )


def _classic_batch() -> dict[str, Any]:
    rows = [spec for group in GROUPS for spec in group]
    return {
        "message_log": [_row(gen_len) for gen_len, _, _ in rows],
        "loss_multiplier": torch.ones(len(rows)),
        "total_reward": torch.tensor([reward for _, reward, _ in rows]),
        **truncation_abort_columns([abort for _, _, abort in rows]),
    }


def _classic_metrics() -> dict[str, Any]:
    """The classic loop's metrics on the same rows, one step = both groups."""
    return _overlong_clamp_metrics(_classic_batch(), _config())


def test_step_metrics_match_classic_loop():
    row_records = [clamp_prompt_group(_record(g), BIN, ROUND_TO)[1] for g in GROUPS]
    sc = step_clamp_metrics(row_records, group_size=4)
    classic = _classic_metrics()

    assert sc.keys() == classic.keys()
    for key, value in classic.items():
        assert (math.isnan(value) and math.isnan(sc[key])) or sc[key] == pytest.approx(
            value
        ), key
    assert sc["num_overlong_clamped"] == 2
    assert sc["overlong_clamped_groups"] == 2
    assert sc["num_truncation_aborted"] == 2


class _FakeDataPlane:
    def put_samples(self, sample_ids, partition_id, fields=None, tags=None):
        return None

    def clear_samples(self, sample_ids, partition_id):
        return None


def test_commit_clamps_before_tensorizing(monkeypatch):
    captured: dict[str, Any] = {}
    real = replay_buffer_module.record_to_train_batch

    def spy(*args, **kwargs):
        captured["batch"] = real(*args, **kwargs)
        return captured["batch"]

    monkeypatch.setattr(replay_buffer_module, "record_to_train_batch", spy)
    buffer = TQReplayBuffer(
        _FakeDataPlane(),
        partition_id="rollout_data",
        pad_value_dict={"token_ids": 0},
        include_message_violation_fields=False,
        overlong_clamp_bin=(BIN, ROUND_TO),
    )
    buffer.set_data_plane_checkpoint_barrier(DataPlaneCheckpointBarrier())
    record = _record(GROUPS[0])
    group_id = buffer.reserve(weight_version=0)
    meta = asyncio.run(buffer.commit(group_id, record, 0, 0))

    assert max(meta.sequence_lengths) <= BIN - ROUND_TO
    assert captured["batch"]["sample_mask"].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert ROW_CLAMP_RECORD_KEY in record.rollout_metrics


def test_mtp_scaled_metrics_on_sc_policy_result():
    # Shape of TQPolicy.finish_train_step's result; aggregate_step_metrics logs the raw keys.
    policy_result = {
        "loss": torch.tensor([0.5]),
        "grad_norm": torch.tensor([1.0]),
        "mtp_metrics": {"mtp_1_loss": 2.0, "loss_scaling_factor": 0.1},
        "all_mb_metrics": {},
    }
    metrics = aggregate_step_metrics(policy_result)
    metrics.update(_mtp_scaled_metrics(policy_result, None))

    assert metrics["mtp/mtp_1_loss"] == pytest.approx(2.0)
    assert metrics["mtp/mtp_1_loss_scaled"] == pytest.approx(0.2)
    assert metrics["mtp/mtp_loss_scaling_factor"] == pytest.approx(0.1)
    assert metrics["mtp/policy_over_scaled_mtp"] == pytest.approx(2.5)


# --- overlong_abort_treatment: penalize (default zero_weight is covered above) ---

PENALTY = -1.0
PENALIZE = dict(overlong_abort_treatment="penalize", overlong_abort_penalty_reward=PENALTY)


def test_config_defaults_to_zero_weight():
    cfg = GRPOConfig()
    assert cfg.overlong_abort_treatment == "zero_weight"
    assert cfg.overlong_abort_penalty_reward == 0.0
    with pytest.raises(ValueError):
        GRPOConfig(overlong_abort_treatment="drop")


def test_token_budget_kind_is_recognised():
    batch = {
        "loss_multiplier": torch.ones(3),
        **truncation_abort_columns(
            [None, {"reason": "token_budget", "call_index": 9, "elapsed_s": 3.0}, None]
        ),
    }
    assert batch["truncation_abort_kind"].tolist() == [0, 3, 0]
    idx, metrics = apply_truncation_abort_clamp(batch, group_size=3)
    assert idx == [1]
    assert batch["loss_multiplier"].tolist() == [1.0, 0.0, 1.0]
    assert metrics["num_token_budget_aborted"] == 1
    assert metrics["token_budget_abort_call_index_max"] == 9


def test_default_mode_leaves_rewards_untouched():
    batch = _classic_batch()
    before = batch["total_reward"]
    _apply_overlong_abort_penalty(batch, _config())
    assert batch["total_reward"] is before
    _, row_record = clamp_prompt_group(_record(GROUPS[0]), BIN, ROUND_TO)
    assert "penalized" not in row_record
    assert row_record["total_reward"] == [1.0, 0.0, 1.0, 0.0]


def test_clamp_prompt_group_penalize_keeps_weight_and_sets_reward():
    record = _record(GROUPS[0])
    weight, row_record = clamp_prompt_group(record, BIN, ROUND_TO, penalty_reward=PENALTY)

    assert weight.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert [c.reward for c in record.completions] == [1.0, PENALTY, PENALTY, 0.0]
    assert row_record["penalized"] == [False, True, True, False]
    assert row_record["total_reward"] == [1.0, PENALTY, PENALTY, 0.0]
    # The clamp still truncates so the row fits the packer.
    assert PROMPT_LEN + record.completions[1].message_log[1]["token_ids"].shape[-1] == BIN - ROUND_TO


def test_classic_penalize_rewrites_reward_and_keeps_weight():
    batch = _classic_batch()
    config = _config(**PENALIZE)
    _apply_overlong_abort_penalty(batch, config)
    # Group 0: ok/overlong/aborted/ok; group 1: overlong/ok/ok/aborted.
    assert batch["total_reward"].tolist() == [1.0, PENALTY, PENALTY, 0.0, PENALTY, 1.0, 1.0, PENALTY]
    metrics = _overlong_clamp_metrics(batch, config)
    assert batch["loss_multiplier"].tolist() == [1.0] * 8
    assert metrics["num_overlong_clamped"] == 2
    assert metrics["num_truncation_aborted"] == 2
    assert metrics["num_overlong_abort_penalized"] == 4
    assert metrics["effective_pct_mixed"] == 100.0
    assert max(sum(m["token_ids"].shape[-1] for m in row) for row in batch["message_log"]) <= BIN - ROUND_TO


def test_penalize_step_metrics_match_classic_loop():
    row_records = [
        clamp_prompt_group(_record(g), BIN, ROUND_TO, penalty_reward=PENALTY)[1]
        for g in GROUPS
    ]
    sc = step_clamp_metrics(row_records, group_size=4)
    batch = _classic_batch()
    config = _config(**PENALIZE)
    _apply_overlong_abort_penalty(batch, config)
    classic = _overlong_clamp_metrics(batch, config)

    assert sc.keys() == classic.keys()
    for key, value in classic.items():
        assert (math.isnan(value) and math.isnan(sc[key])) or sc[key] == pytest.approx(
            value
        ), key


def test_penalty_drives_the_advantage():
    batch = _classic_batch()
    _apply_overlong_abort_penalty(batch, _config(**PENALIZE))
    estimator = GRPOAdvantageEstimator(
        AdvEstimatorConfig(use_leave_one_out_baseline=True, normalize_rewards=False), None
    )
    adv = estimator.compute_advantage(
        prompt_ids=torch.tensor([[0]] * 4 + [[1]] * 4),
        rewards=batch["total_reward"],
        mask=torch.ones(8, 1),
    )[:, 0]
    # Penalized rows sit below every terminated row of their group.
    assert adv[1] < adv[3] and adv[2] < adv[3] and adv[4] < adv[5] and adv[7] < adv[5]
    assert adv[1].item() == pytest.approx(PENALTY - (1.0 + PENALTY + 0.0) / 3)


def test_commit_penalize_writes_penalized_reward_with_full_weight(monkeypatch):
    captured: dict[str, Any] = {}
    real = replay_buffer_module.record_to_train_batch

    def spy(*args, **kwargs):
        captured["batch"] = real(*args, **kwargs)
        return captured["batch"]

    monkeypatch.setattr(replay_buffer_module, "record_to_train_batch", spy)
    buffer = TQReplayBuffer(
        _FakeDataPlane(),
        partition_id="rollout_data",
        pad_value_dict={"token_ids": 0},
        include_message_violation_fields=False,
        overlong_clamp_bin=(BIN, ROUND_TO),
        overlong_abort_penalty_reward=PENALTY,
    )
    buffer.set_data_plane_checkpoint_barrier(DataPlaneCheckpointBarrier())
    group_id = buffer.reserve(weight_version=0)
    meta = asyncio.run(buffer.commit(group_id, _record(GROUPS[0]), 0, 0))

    assert max(meta.sequence_lengths) <= BIN - ROUND_TO
    assert captured["batch"]["sample_mask"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert captured["batch"]["total_reward"].tolist() == [1.0, PENALTY, PENALTY, 0.0]


@pytest.mark.parametrize("train_fn", [_grpo_train_impl, async_grpo_train])
def test_penalty_runs_before_rewards_are_read(train_fn):
    import inspect

    src = inspect.getsource(train_fn)
    penalty = src.index("_apply_overlong_abort_penalty(repeated_batch, master_config)")
    assert penalty < src.index('rewards = repeated_batch["total_reward"]')
    assert penalty < src.index("_overlong_clamp_metrics(repeated_batch, master_config)")
