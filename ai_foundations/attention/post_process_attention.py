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

"""A post-processing utility for attention weights."""

from gemma.modules import K_MASK
import jax
import jax.numpy as jnp


def post_process_attention(
    logits: jax.Array, attention_mask: jax.Array
) -> jax.Array:
  """Applies the attention mask and re-computes the attention_weights.

  Args:
      logits: An ndarray of attention logits.
      attention_mask: An ndarray of booleans defining the attention maks.

  Returns:
      probs: Attention weights after applying attention mask.
  """
  logits = jnp.where(attention_mask, logits, K_MASK)
  _, l, c = logits.shape
  probs = jax.nn.softmax(logits).reshape(l, c)

  return probs
