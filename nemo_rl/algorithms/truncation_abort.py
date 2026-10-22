# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LOCAL PATCH: clamp rollouts the Gym agent aborted early. See status S3.26.

The agent (simple_agent) aborts a rollout on repetition, a wall-clock backstop or a
token budget and reports it in ``_ng_truncation``; rollout collection turns that into the
``truncation_abort_*`` batch columns. By default we zero ``loss_multiplier`` for those rows
and leave them in place -- dropping would be selective on length and would collapse
the GRPO group variance that makes mixed groups useful. With
``grpo.overlong_abort_treatment: penalize`` they keep full weight and a penalty reward.
"""

from typing import Any

import torch

KIND_NONE = 0
KIND_REPETITION = 1
KIND_BACKSTOP = 2
KIND_TOKEN_BUDGET = 3
KIND_COLUMN = "truncation_abort_kind"
_REASON_TO_KIND = {
    "repetition": KIND_REPETITION,
    "backstop": KIND_BACKSTOP,
    "token_budget": KIND_TOKEN_BUDGET,
}
_KIND_TO_NAME = {kind: name for name, kind in _REASON_TO_KIND.items()}


def truncation_abort_columns(records: list[dict | None]) -> dict[str, torch.Tensor]:
    """Build the three batch columns from one ``_ng_truncation`` record per rollout."""
    kinds, indices, elapsed = [], [], []
    for record in records:
        record = record or {}
        kinds.append(_REASON_TO_KIND.get(record.get("reason"), KIND_NONE))
        indices.append(int(record.get("call_index", -1)) if record else -1)
        elapsed.append(float(record.get("elapsed_s", 0.0)) if record else 0.0)
    return {
        KIND_COLUMN: torch.tensor(kinds, dtype=torch.long),
        "truncation_abort_call_index": torch.tensor(indices, dtype=torch.long),
        "truncation_abort_elapsed_s": torch.tensor(elapsed, dtype=torch.float32),
    }


def _distribution(prefix: str, values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {f"{prefix}_{k}": float("nan") for k in ("mean", "p50", "p90", "max")}
    values = values.float()
    return {
        f"{prefix}_mean": values.mean().item(),
        f"{prefix}_p50": values.quantile(0.5).item(),
        f"{prefix}_p90": values.quantile(0.9).item(),
        f"{prefix}_max": values.max().item(),
    }


def aborted_row_indices(repeated_batch: Any) -> list[int]:
    """Rows the agent aborted; empty when the column is absent (non-Gym rollout paths)."""
    if KIND_COLUMN not in repeated_batch:
        return []
    kind = torch.as_tensor(repeated_batch[KIND_COLUMN]).detach().cpu().reshape(-1)
    return torch.nonzero(kind != KIND_NONE, as_tuple=False).reshape(-1).tolist()


def apply_truncation_abort_clamp(
    repeated_batch: Any, group_size: int, zero_weight: bool = True
) -> tuple[list[int], dict[str, float]]:
    """Zero ``loss_multiplier`` for aborted rows (unless ``zero_weight`` is False, i.e.
    penalize mode); return their indices and metrics.

    Returns an empty metric dict when the column is absent (non-Gym rollout paths);
    otherwise always emits every counter, zeros included, so a run with no aborts is
    distinguishable in TensorBoard from a run where the mechanism is not wired.
    """
    if KIND_COLUMN not in repeated_batch:
        return [], {}

    kind = repeated_batch[KIND_COLUMN]
    if not torch.is_tensor(kind):
        kind = torch.as_tensor(kind)
    kind = kind.detach().cpu().reshape(-1).long()
    n_rows = int(kind.numel())

    aborted = kind != KIND_NONE
    aborted_idx = torch.nonzero(aborted, as_tuple=False).reshape(-1).tolist()
    if aborted_idx and zero_weight:
        loss_multiplier = repeated_batch["loss_multiplier"].clone()
        loss_multiplier[torch.as_tensor(aborted_idx, dtype=torch.long)] = 0
        repeated_batch["loss_multiplier"] = loss_multiplier

    call_index = repeated_batch.get("truncation_abort_call_index")
    elapsed = repeated_batch.get("truncation_abort_elapsed_s")
    call_index = (
        torch.as_tensor(call_index).detach().cpu().reshape(-1)
        if call_index is not None
        else torch.full((n_rows,), -1)
    )
    elapsed = (
        torch.as_tensor(elapsed).detach().cpu().reshape(-1)
        if elapsed is not None
        else torch.zeros(n_rows)
    )

    metrics: dict[str, float] = {
        "num_truncation_aborted": len(aborted_idx),
        "truncation_aborted_groups": len({i // max(group_size, 1) for i in aborted_idx}),
    }
    for kind_value, name in _KIND_TO_NAME.items():
        selected = kind == kind_value
        count = int(selected.sum().item())
        metrics[f"num_{name}_aborted"] = count
        metrics[f"{name}_abort_rate"] = (count / n_rows) if n_rows else 0.0
        metrics.update(_distribution(f"{name}_abort_call_index", call_index[selected]))
        metrics.update(_distribution(f"{name}_abort_elapsed_s", elapsed[selected]))
    return aborted_idx, metrics


# SingleController: rows are tensorized per prompt group at TQReplayBuffer.commit, so the
# clamp and abort masks run there; the train pump reduces the per-row records per step.
ROW_CLAMP_RECORD_KEY = "_overlong_clamp_rows"


def clamp_prompt_group(
    record: Any,
    bin_capacity: int,
    round_to: int,
    penalty_reward: float | None = None,
) -> tuple[torch.Tensor, dict[str, list]]:
    """Clamp overlong rows and flag agent aborts in one prompt group, before tensorizing.

    Returns per-row loss weights (0 for clamped or aborted rows) and a JSON-able per-row
    record for :func:`step_clamp_metrics`. Mutates the completions' message logs in place.
    With ``penalty_reward`` set (penalize mode) those rows keep weight 1 and their
    ``Completion.reward`` is overwritten, so the advantage stage sees the penalty.
    """
    from nemo_rl.algorithms.grpo import _clamp_overlong_sequences  # grpo imports us

    completions = record.completions
    batch: dict[str, Any] = {
        "message_log": [c.message_log for c in completions],
        "loss_multiplier": torch.ones(len(completions)),
    }
    zero_weight = penalty_reward is None
    _, clamped_idx = _clamp_overlong_sequences(
        batch, bin_capacity, round_to, zero_weight=zero_weight
    )
    columns = truncation_abort_columns(
        [(c.env_extras or {}).get("_ng_truncation") for c in completions]
    )
    weight = batch["loss_multiplier"]
    aborted = columns[KIND_COLUMN] != KIND_NONE
    penalized = [i in clamped_idx or bool(aborted[i]) for i in range(len(completions))]
    if zero_weight:
        weight[aborted] = 0
    else:
        for completion, hit in zip(completions, penalized):
            if hit:
                completion.reward = penalty_reward
    row_record: dict[str, list] = {
        "clamped": [i in clamped_idx for i in range(len(completions))],
        "total_reward": [float(c.reward) for c in completions],
        **{name: column.tolist() for name, column in columns.items()},
    }
    if not zero_weight:  # key presence marks penalize mode for step_clamp_metrics
        row_record["penalized"] = penalized
    return weight, row_record


def step_clamp_metrics(
    row_records: list[dict[str, list]], group_size: int
) -> dict[str, Any]:
    """Reduce one step's per-group row records to the classic per-step clamp metrics."""
    from nemo_rl.algorithms.grpo import _overlong_clamp_summary

    rows = {
        key: [value for record in row_records for value in record[key]]
        for key in row_records[0]
    }
    batch = {
        "total_reward": torch.tensor(rows["total_reward"], dtype=torch.float32),
        "loss_multiplier": torch.ones(len(rows["clamped"])),
        KIND_COLUMN: torch.tensor(rows[KIND_COLUMN], dtype=torch.long),
        "truncation_abort_call_index": torch.tensor(
            rows["truncation_abort_call_index"], dtype=torch.long
        ),
        "truncation_abort_elapsed_s": torch.tensor(
            rows["truncation_abort_elapsed_s"], dtype=torch.float32
        ),
    }
    clamped_idx = [i for i, clamped in enumerate(rows["clamped"]) if clamped]
    return _overlong_clamp_summary(
        batch,
        clamped_idx,
        n_rows=len(rows["clamped"]),
        group_size=group_size,
        zero_weight="penalized" not in rows,
    )
