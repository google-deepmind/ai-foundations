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

"""Replaces the attention mechanism of Gemma.

Replaces the attention mechanism of Gemma such that it exposes the attention
weights and QKV matrices.
"""

from gemma import modules as _modules
from gemma import positional_embeddings
import jax
import jax.numpy as jnp

LayerCache = _modules.LayerCache
AttentionType = _modules.AttentionType
K_MASK = _modules.K_MASK


class AttentionWeightAttention(_modules.Attention):
  """Attention module.

  This replaces `gemma.modules.Attention` such that the `__call__` method also
  returns the attention weights and the Q, K, V matrices such that learners can
  visualize and manipulate these matrices.
  """

  def __call__(  # type: ignore
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      cache: LayerCache | None,
      attn_mask: jax.Array,
  ) -> tuple[LayerCache | None, jax.Array, jax.Array, dict[str, jax.Array]]:
    """Applies multi-head attention to the inputs.

    Args:
      x: Input sequence of shape [batch_size, seq_len, embed_dim].
      segment_pos: Input absolute positions of shape [batch_size, seq_len].
      cache: KV cache or None.
      attn_mask: Attention mask of shape [batch_size, seq_len, cache_size].

    Returns:
      cache: Updated attention KV cache.
      outputs: Output sequence of shape [batch_size, seq_len, embed_dim].
      probs: Attention weights of shape [batch_size, seq_len, seq_len].
      qkv: Dictionary with Q, K, and V matrices.
    """
    if self.use_qkv_einsum:
      # [batch_size, seq_len, num_heads, head_dim].
      query_proj, key_proj, value_proj = self.qkv_einsum('BTD,SNDH->SBTNH', x)
    else:
      query_proj = self.q_einsum('BTD,NDH->BTNH', x)
      key_proj, value_proj = self.kv_einsum('BSD,CKDH->CBSKH', x)

    if self.use_qk_norm:
      query_proj = self._query_norm(query_proj)
      key_proj = self._key_norm(key_proj)

    query_proj = positional_embeddings.apply_rope(
        query_proj,
        segment_pos,
        base_frequency=self.rope_base_frequency,
    )
    query_scaled = query_proj * self.query_pre_attn_scalar
    key_proj = positional_embeddings.apply_rope(
        key_proj,
        segment_pos,
        base_frequency=self.rope_base_frequency,
    )

    # Cache is left aligned.
    # Save the KV values to the cache.
    if cache is not None:
      end_index = cache['end_index'][0]
      slice_indices = (0, end_index % cache['v'].shape[1], 0, 0)
      # [batch_size, cache_size, num_heads, head_dim].
      value_proj = jax.lax.dynamic_update_slice(
          cache['v'],
          value_proj,
          slice_indices,
      )

      # [batch_size, cache_size, num_heads, head_dim].
      key_proj = jax.lax.dynamic_update_slice(
          cache['k'], key_proj, slice_indices
      )

    # Return Q, K, V matrices as qkv_dict so that learners can manipulate them.
    qkv_dict = {'query': query_proj, 'key': key_proj, 'value': value_proj}

    if self.use_gqa:
      # Reshape matrices to enable einsums over groups.
      b, t, kg, h = query_scaled.shape
      query_scaled = query_scaled.reshape(
          (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
      )
      logits = jnp.einsum('BTKGH,BSKH->BTKGS', query_scaled, key_proj)
      b, t, k, g, s = logits.shape
      logits = logits.reshape((b, t, k * g, s))
    else:
      # [batch_size, seq_len, num_heads, cache_size].
      # If cache is None, then cache_size = seq_len.
      logits = jnp.einsum('BTNH,BSNH->BTNS', query_scaled, key_proj)

    if self.attn_logits_soft_cap is not None:
      logits = jnp.tanh(logits / self.attn_logits_soft_cap)
      logits = logits * self.attn_logits_soft_cap

    if self.attn_type == AttentionType.LOCAL_SLIDING:
      if self.sliding_window_size is None:
        raise ValueError(
            'Sliding_window_size must be set if Local Sliding attention type'
        )
      sliding_mask = _modules._create_sliding_mask(
          segment_pos,
          end_index=cache['end_index'][0] if cache is not None else 0,
          # Derive cache length from attn_mask shape in case cache is None
          cache_len=attn_mask.shape[-1],
          sliding_window_size=self.sliding_window_size,
      )
      # [batch_size, seq_len, cache_size].
      attn_mask *= sliding_mask

    # [batch_size, seq_len, num_heads, cache_size].
    padded_logits = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)

    # Multi-head attention matrices.
    # [batch_size, seq_len, num_heads, cache_size].
    probs = jax.nn.softmax(padded_logits, axis=-1).astype(key_proj.dtype)
    probs_orig = probs
    if self.use_gqa:
      # Reshape matrices to enable einsums over groups.
      b, t, kg, h = probs.shape
      probs = probs.reshape(
          (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
      )
      encoded = jnp.einsum('BTKGS,BSKH->BTKGH', probs, value_proj)
      b, t, k, g, h = encoded.shape
      encoded = encoded.reshape((b, t, k * g, h))
    else:
      # [batch_size, seq_len, num_heads, head_dim].
      encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)

    # [batch_size, seq_len, features].
    attn_output = self.attn_vec_einsum('BTNH,NHD->BTD', encoded)

    if cache is not None:
      seq_len = x.shape[1]
      new_cache = {
          # [batch_size, cache_size, num_heads, head_dim].
          'v': value_proj,
          # [batch_size, cache_size, num_heads, head_dim].
          'k': key_proj,
          # [batch_size].
          'end_index': cache['end_index'] + seq_len,
      }
    else:
      new_cache = None

    return new_cache, attn_output, probs_orig, qkv_dict
