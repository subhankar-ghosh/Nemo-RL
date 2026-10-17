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

The agent (simple_agent) aborts a rollout on repetition or a wall-clock backstop and
reports it in ``_ng_truncation``; rollout collection turns that into the
``truncation_abort_*`` batch columns. Here we zero ``loss_multiplier`` for those rows
and leave them in place -- dropping would be selective on length and would collapse
the GRPO group variance that makes mixed groups useful.
"""

from typing import Any

import torch

KIND_NONE = 0
KIND_REPETITION = 1
KIND_BACKSTOP = 2
KIND_COLUMN = "truncation_abort_kind"
_REASON_TO_KIND = {"repetition": KIND_REPETITION, "backstop": KIND_BACKSTOP}
_KIND_TO_NAME = {KIND_REPETITION: "repetition", KIND_BACKSTOP: "backstop"}


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


def apply_truncation_abort_clamp(
    repeated_batch: Any, group_size: int
) -> tuple[list[int], dict[str, float]]:
    """Zero ``loss_multiplier`` for aborted rows; return their indices and metrics.

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
    if aborted_idx:
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
