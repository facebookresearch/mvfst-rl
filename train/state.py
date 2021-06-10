# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions to handle the input state.
"""

from enum import Enum, auto
from typing import Any, List, Union

import torch


# The `Field` enum must be a copy of the one found in `NetworkState.h`.
class Field(Enum):

    # Overridden to guarantee that indices remain consistent with C++ even if
    # the default Python implementation changes (unlikely).
    def _generate_next_value_(name, start, count, last_values):
        return last_values[-1] + 1

    RTT_MIN = 0
    RTT_STANDING = auto()

    LRTT = auto()
    SRTT = auto()
    RTT_VAR = auto()
    DELAY = auto()

    CWND = auto()
    IN_FLIGHT = auto()
    WRITABLE = auto()
    SENT = auto()
    RECEIVED = auto()
    RETRANSMITTED = auto()

    PTO_COUNT = auto()
    TOTAL_PTO_DELTA = auto()
    RTX_COUNT = auto()
    TIMEOUT_BASED_RTX_COUNT = auto()

    ACKED = auto()
    THROUGHPUT = auto()

    LOST = auto()
    PERSISTENT_CONGESTION = auto()

    NUM_FIELD = auto()


# These offsets should match the order of aggregate statistics found
# in `CongestionControlEnv::stateSummary()`.
N = Field.NUM_FIELD.value
OFFSET_SUM = 0
OFFSET_MEAN = N
OFFSET_STD = N * 2
OFFSET_MIN = N * 3
OFFSET_MAX = N * 4
OFFSET_END = N * 5  # end of summary statistics


def get_indices(fields: List[Field]) -> torch.Tensor:
    """
    Return all indices associated to the given fields.

    This includes indices associate to all summary statistics (sum, mean, etc.)
    """
    indices = [
        offset + field.value
        for field in fields
        for offset in [OFFSET_SUM, OFFSET_MEAN, OFFSET_STD, OFFSET_MIN, OFFSET_MAX]
    ]
    return torch.tensor(indices, dtype=torch.int64)


def get_indices_except(fields: Union[Field, List[Field]]) -> torch.Tensor:
    """Same as `get_indices()` but to *exclude* a list of fields"""
    if isinstance(fields, Field):
        fields = [fields]
    skip = set(f.value for f in fields)
    return get_indices(fields=[f for f in Field if f.value not in skip])


def get_from_state(state, field, offset, dim=0):
    """
    Fetch the `state` entry found at index `offset` + `field`.

    :param state: Input state (a PyTorch tensor).
    :param field: The field to fetch (a `Field` enum).
    :param offset: Offset to apply to the index.
    :param dim: The dimension along which we should index.
    """
    idx = offset + field.value
    if dim == 0:
        return state[idx]  # straightforward indexing on first dimension
    else:
        idx_tensor = torch.tensor(idx).to(state.device)
        return state.index_select(dim, idx_tensor).squeeze(dim)


def get_sum(state, field, dim=0):
    """Fetch the sum of `field` in `state`"""
    return get_from_state(state, field, offset=OFFSET_SUM, dim=dim)


def get_mean(state, field, dim=0):
    """Fetch the mean of `field` in `state`"""
    return get_from_state(state, field, offset=OFFSET_MEAN, dim=dim)


def get_std(state, field, dim=0):
    """Fetch the standard deviation of `field` in `state`"""
    return get_from_state(state, field, offset=OFFSET_STD, dim=dim)


def get_min(state, field, dim=0):
    """Fetch the minimum of `field` in `state`"""
    return get_from_state(state, field, offset=OFFSET_MIN, dim=dim)


def get_max(state, field, dim=0):
    """Fetch the maximum of `field` in `state`"""
    return get_from_state(state, field, offset=OFFSET_MAX, dim=dim)


def set_indices(
    state: torch.Tensor,
    indices: Union[List[int], torch.Tensor],
    value: Any,
    dim: int = 0,
):
    """
    Set specific indices in state to the desired value.

    :param state: The state to modify in-place, as a PyTorch tensor.
    :param indices: Indices of the fields that should be set.
    :param value: Value to set.
    :param dim: The dimension of the tensor corresponding to the fields to be set.
    """
    slices = [slice(None)] * dim  # select everything up to `dim`
    slices.append(indices)  # select only the desired indices
    state[slices] = value
