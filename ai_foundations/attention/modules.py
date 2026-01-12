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

"""Overwrites the Gemma transformer block and the Gemma-1B model.

Overwrites the Gemma transformer block and the Gemma-1B model such that it
returns the attention weights and the KQV matrices for all layers.
"""

import functools

from ai_foundations import attention
import flax
from flax import linen as nn
from gemma import gm
from gemma import layers as _layers
from gemma import modules as _modules
from gemma import transformer
from gemma.gm.utils import _dtype_params
from gemma.gm.utils import _jax_utils
from gemma.gm.vision import _token_utils
import jax
import jax.numpy as jnp
from kauldron.typing import Bool, Float, Int, UInt8, typechecked  # pylint: disable=g-multiple-import,g-importing-member

LayerCache = _modules.LayerCache
AttentionType = _modules.AttentionType

AttentionWeightAttention = attention.AttentionWeightAttention


class AttentionWeightBlock(_modules.Block):
  """Transformer block.

  This overwrites `gemma.modules.Block` such that the `__call__` method also
  returns the attention weights and the Q, K, V matrices such that learners can
  visualize and manipulate these matrices.
  """

  def setup(self):
    self.pre_attention_norm = _layers.RMSNorm()
    self.attn = AttentionWeightAttention(
        num_heads=self.num_heads,
        features=self.embed_dim,
        head_dim=self.head_dim,
        num_kv_heads=self.num_kv_heads,
        attn_type=self.attn_type,
        query_pre_attn_scalar=self.query_pre_attn_scalar,
        rope_base_frequency=self.rope_base_frequency,
        attn_logits_soft_cap=self.attn_logits_soft_cap,
        sliding_window_size=self.sliding_window_size,
        use_qk_norm=self.use_qk_norm,
    )
    self.post_attention_norm = None
    if self.use_post_attn_norm:
      self.post_attention_norm = _layers.RMSNorm()

    self.pre_ffw_norm = _layers.RMSNorm()
    self.mlp = _modules.FeedForward(
        features=self.embed_dim,
        hidden_dim=self.hidden_dim,
        transpose_gating_einsum=self.transpose_gating_einsum,
    )
    self.post_ffw_norm = None
    if self.use_post_ffw_norm:
      self.post_ffw_norm = _layers.RMSNorm()

  def __call__(  # type: ignore
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      cache: LayerCache | None,
      attn_mask: jax.Array,
  ) -> tuple[LayerCache | None, jax.Array, jax.Array, dict[str, jax.Array]]:
    """Applies the block to the inputs.

    Args:
      x: Input sequence of shape [batch_size, seq_len, embed_dim].
      segment_pos: Input absolute positions of shape [batch_size, seq_len].
      cache: KV cache or None.
      attn_mask: Attention mask of shape [batch_size, seq_len, cache_size].

    Returns:
      cache: Updated attention KV cache.
      outputs: Output sequence of shape [batch_size, seq_len, embed_dim].
      attn_probs: Attention weights of shape [batch_size, seq_len, seq_len].
      qkv: Dictionary with Q, K, and V matrices.
    """
    inputs_normalized = self.pre_attention_norm(x)
    cache, attn_output, attn_probs, qkv_dict = self.attn(
        inputs_normalized, segment_pos, cache, attn_mask
    )

    if self.post_attention_norm is not None:
      attn_output = self.post_attention_norm(attn_output)

    attn_output += x

    outputs = self.pre_ffw_norm(attn_output)

    outputs = self.mlp(outputs)

    if self.post_ffw_norm is not None:
      outputs = self.post_ffw_norm(outputs)

    outputs += attn_output

    return cache, outputs, attn_probs, qkv_dict


@flax.struct.dataclass
class Output:
  """Output of the Gemma model.

  Attributes:
    logits: Predicted logits of the model.
    cache: Updated cache if the input cache is not None, None elsewhere.
    attention_weights: Dictionary with layer-wise attention weights.
    attention_mask: Attention mask for input.
    qkv: Dictionary with layer-wise Q, K, and V matrices.
  """

  # When `return_last_only`, `logits` is `*B V`
  logits: Float['*B L V'] | Float['*B V']
  cache: transformer.Cache | None
  attention_weights: dict[str, jax.Array] | None
  attention_mask: Bool['*B L cache_length'] | None
  qkv: dict[str, dict[str, jax.Array]] | None


class AttentionWeightGemma3_1B(gm.nn.Gemma3_1B):  # pylint: disable=invalid-name
  """Gemma3 transformer architecture.

  This overwrites `gm.nn.Gemma3_1B` such that the `__call__` method also
  returns the attention weights and the Q, K, V matrices for each layer such
  that learners can visualize and manipulate these matrices.
  """

  def setup(self):
    """Setup the model."""
    self.embedder = _modules.Embedder(
        vocab_size=self.config.num_embed,
        embed_dim=self.config.embed_dim,
        vision_proj_dim=self.config.vision_encoder.siglip_encoder.width
        if self.config.vision_encoder
        else None,
    )

    self.blocks = [
        AttentionWeightBlock(
            name=f'layer_{i}',
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
            embed_dim=self.config.embed_dim,
            head_dim=self.config.head_dim,
            hidden_dim=self.config.hidden_dim,
            sliding_window_size=self.config.sliding_window_size,
            use_post_attn_norm=self.config.use_post_attn_norm,
            use_post_ffw_norm=self.config.use_post_ffw_norm,
            attn_logits_soft_cap=self.config.attn_logits_soft_cap,
            attn_type=attn_type,
            query_pre_attn_scalar=self.config.query_pre_attn_scalar(),
            transpose_gating_einsum=self.config.transpose_gating_einsum,
            use_qk_norm=self.config.use_qk_norm,
            rope_base_frequency=self.config.local_base_frequency
            if attn_type == _modules.AttentionType.LOCAL_SLIDING
            else self.config.global_base_frequency,
            rope_scale_factor=self.config.local_scale_factor
            if attn_type == _modules.AttentionType.LOCAL_SLIDING
            else self.config.global_scale_factor,
        )
        for i, attn_type in zip(
            range(self.config.num_layers), self.config.attention_types
        )
    ]
    self.final_norm = _layers.RMSNorm()

    self.vision_encoder = self.config.vision_encoder

  # Calling `model.apply` on Colab makes the Kernel crash unless it is jitted.
  @functools.partial(
      nn.jit,
      static_argnames=('self', 'return_last_only'),
  )
  # The function accepts/returns aribtrary batch shape, but inside the
  # function, the batch dimension is flattened to a single dimension.
  @_jax_utils.flatten_unflatten_batch_dim()
  @typechecked
  def __call__(  # pytype: disable=signature-mismatch
      self,
      tokens: Int['*B L'],
      *,
      images: UInt8['*B N H W C'] | UInt8['*B H W C'] | None = None,
      # TODO(epot): Cleanup and simplify the API.
      positions: Int['*B L'] | None = None,
      positions_offset: Int['*B'] | None = None,
      cache: transformer.Cache | None = None,
      # During training and pre-filling, the attention mask is `*B L L`
      # When sampling (after prefilling), tokens are decoded one by one,
      # so the attention mask is `*B 1 cache_length`
      attention_mask: Bool['*B L cache_length'] | None = None,
      return_last_only: bool | None = None,
  ) -> Output:  # Output['*B']
    """Transformer forward pass.

    You can run this forward pass two ways: with or without an attention kv
    cache.

    Args:
      tokens: input sequence of tokens.
      images: Images to feed to the vision encoder.
      positions: input absolute positions.
      positions_offset: Offset to add to the positions. Used for multi-turn when
        the cache is provided and `positions` is None.
      cache: Attention KV cache or None.
      attention_mask: transformer input mask.
      return_last_only: If `True`, only compute and return the logits of the
        last input token in sequence. Useful for decoding where we don't need to
        compute logits for the whole sequence, but only for the last token.
        Otherwise, return all logits. Default to `False`.

    Returns:
      predicted_logits: output logits predicted by the model
      new_cache: updated cache if the input cache is not None, None elsewhere.
      attention_weights: Dictionary with layer-wise attention weights.
      attention_mask: Attention mask for input.
      qkv: Dictionary with layer-wise Q, K, and V matrices.
    """
    return_last_only = self._get_return_last_only(return_last_only)

    with _dtype_params.initialize_param_with_dtype(self.dtype):

      # Encode the text tokens, eventually including the vision embeddings.
      inputs = self._encode_and_get_inputs(
          tokens=tokens,
          images=images,
          positions=positions,
          positions_offset=positions_offset,
          attention_mask=attention_mask,
      )
      del positions, attention_mask

      x = inputs.embeddings

      old_cache = cache or {}
      new_cache = {}
      attn_weights = {}
      qkv = {}
      for i, block in enumerate(self.blocks):
        layer_name = f'layer_{i}'
        layer_cache, x, layer_attn_weights, layer_qkv = block(
            x,
            inputs.positions,
            old_cache.get(layer_name),
            inputs.attention_mask,
        )
        new_cache[layer_name] = (
            layer_cache  # pytype: disable=container-type-mismatch
        )
        attn_weights[layer_name] = layer_attn_weights
        qkv[layer_name] = layer_qkv

      x = self.final_norm(x)

    if return_last_only:
      last_input_token_idx = jnp.sum(inputs.inputs_mask, axis=-1) - 1
      # TODO(epot): Use `jnp.take_along_axis`
      x = x[jnp.arange(len(x)), last_input_token_idx, ...]
    elif images is not None:
      # Remove the MM extra tokens inserted.
      # During fine-tuning, the prompt is always masked, and the model cannot
      # generate images tokens, so the logits are meaningless anyway.
      x = _token_utils.remove_mm_logits(
          logits=x,
          tokens=tokens,
          num_tokens_per_image=self.config.vision_encoder.num_mm_tokens_per_image,  # pytype: disable=attribute-error
      )

    logits = self.embedder.decode(x)

    if self.config.final_logit_softcap is not None:
      logits /= self.config.final_logit_softcap
      logits = jnp.tanh(logits) * self.config.final_logit_softcap

    return Output(
        logits=logits,
        cache=None if cache is None else new_cache,
        attention_weights=attn_weights,
        attention_mask=inputs.attention_mask,
        qkv=qkv,
    )
