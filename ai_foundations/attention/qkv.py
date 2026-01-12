# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions to extract Q, K, and V matrices for a layer."""

from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp


def get_qkv_matrices(
    qkv: Dict[str, Dict[str, jax.Array]],
    layer: int = 19,
    head: Optional[int] = 0,
) -> Union[
    Tuple[jax.Array, jax.Array, jax.Array],
    Tuple[List[jax.Array], List[jax.Array], List[jax.Array]],
]:
  """Extracts Q, K, and V matrices from the qkv dictionary for a given layer.

  This function can operate in two modes depending on the `head` parameter:
  1.  **Multi-head mode**: Returns lists of Q, K, and V matrices
      for all attention heads.
  2.  **Single-head mode:** Returns single Q, K, and V matrices
      for the specified attention head.

  Args:
      qkv: Dictionary with layer-wise Q, K, and V matrices.
      layer: The layer number from which to extract the matrices.
      head: The specific attention head to extract. If None, all heads are
        processed and returned as lists.

  Returns:
      If `head` is an integer, a tuple of three JAX arrays (Q, K, V) for that
      head.
      If `head` is None, a tuple of three lists of JAX arrays, representing
      the Q, K, and V matrices for all heads.

  Raises:
      AssertionError: If the specified layer or head is not valid.
  """
  layer_key = f"layer_{layer}"
  assert layer_key in qkv, f"Invalid layer '{layer}'."

  layer_qkv = qkv[layer_key]
  query_proj, key_proj, value_proj = (
      layer_qkv["query"],
      layer_qkv["key"],
      layer_qkv["value"],
  )

  # Get shape information from the query projection.
  # Assumes shape is (batch, seq_len, num_heads, embed_dim).
  _, seq_len, num_heads, embed_dim = query_proj.shape

  if head is not None:  # Single head mode.
    assert (
        head < num_heads
    ), f"Invalid head index '{head}'. Model has {num_heads} heads."

    # Extract the specified head for Q. K and V are the same across heads.
    q = query_proj[0, :, head, :].reshape(seq_len, embed_dim)
    k = key_proj[0, :, :, :].reshape(seq_len, embed_dim)
    v = value_proj[0, :, :, :].reshape(seq_len, embed_dim)
    return q, k, v

  else:  # Multi-head mode.
    # Reshape Q to remove the batch dimension
    query_all_heads = query_proj[0, :, :, :].reshape(
        seq_len, num_heads, embed_dim
    )

    # Broadcast the single K and V projections to match the number of heads.
    key_all_heads = jnp.broadcast_to(
        key_proj[0, :, :, :].reshape(seq_len, 1, embed_dim),
        (seq_len, num_heads, embed_dim),
    )
    value_all_heads = jnp.broadcast_to(
        value_proj[0, :, :, :].reshape(seq_len, 1, embed_dim),
        (seq_len, num_heads, embed_dim),
    )

    # Deconstruct the head dimension into lists of 2D arrays.
    query_list = [query_all_heads[:, i, :] for i in range(num_heads)]
    key_list = [key_all_heads[:, i, :] for i in range(num_heads)]
    value_list = [value_all_heads[:, i, :] for i in range(num_heads)]

    return query_list, key_list, value_list
