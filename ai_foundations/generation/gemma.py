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

"""Inference function for generating text with pre-trained Gemma models.

This module provides a high-level function to prompt a Gemma model, generate
a text continuation, and retrieve the model's logits for the next token.
"""

from typing import Any, Dict, Literal, Mapping, Optional, Tuple

from ai_foundations import attention
from ai_foundations import generation
from gemma import gm
import jax
import jax.numpy as jnp


def _sample_from_model(
    input_text: str,
    tokenizer: gm.text.Gemma3Tokenizer,
    model: gm.nn.Transformer,
    params: Mapping[str, Any],
    max_new_tokens: int = 1,
    sampling_mode: Literal["random", "greedy"] = "random",
) -> str:
  """Helper function for sampling from a Gemma model.

  Args:
    input_text: The prompt to the model.
    tokenizer: The tokenizer for encoding and decoding text.
    model: The Gemma model to sample from.
    params: The model parameters.
    max_new_tokens: The maximum number of new tokens to generate.
    sampling_mode: The sampling strategy. 'greedy' picks the most likely next
      token, while 'random' samples from the probability distribution.

  Returns:
    The generated text.

  Raises:
    ValueError: If an unsupported `sampling_mode` is provided.
  """

  if sampling_mode not in ["random", "greedy"]:
    raise ValueError(
        f"Sampling mode {sampling_mode} is not supported. Supported options are"
        " 'random' and 'greedy'."
    )

  sampler = gm.text.Sampler(
      model=model,
      params=params,
      tokenizer=tokenizer,
  )

  if sampling_mode == "greedy":
    sampler_output_text = sampler.sample(
        input_text, max_new_tokens=max_new_tokens, sampling=gm.text.Greedy()
    )
  else:
    sampler_output_text = sampler.sample(
        input_text,
        max_new_tokens=max_new_tokens,
        sampling=gm.text.RandomSampling(),
    )
  return sampler_output_text


def prompt_transformer_model(
    input_text: str,
    max_new_tokens: int = 10,
    model_name: Literal["Gemma-1B", "Gemma-4B"] = "Gemma-1B",
    sampling_mode: Literal["random", "greedy"] = "random",
    loaded_model: Optional[
        Tuple[gm.text.Gemma3Tokenizer, gm.nn.Transformer, Any]
    ] = None,
) -> Tuple[str, jax.Array, gm.text.Gemma3Tokenizer]:
  """Generate text from a transformer model (Gemma) based on the input text.

  Args:
    input_text: The input prompt for the model.
    max_new_tokens: The maximum number of new tokens to generate.
    model_name: The name of the model to load. Supported options are 'Gemma-1B'
      and 'Gemma-4B'.
    sampling_mode: Whether to use random or greedy sampling. Supported options
      are 'random' and 'greedy'.
    loaded_model: A tuple containing the tokenizer, the model, and the
      parameters to prevent re-loading of model on every prompt.

  Returns:
    output_text: The generated text, including the input text and the
      model's output.
    next_token_logits: Logits for the next token (probability distribution).
    tokenizer: The tokenizer used for encoding/decoding the text.

  Raises:
    ValueError: If the model_name is not recognized or supported.
  """

  # Process for Gemma-based models.
  if model_name not in ["Gemma-1B", "Gemma-4B"]:
    raise ValueError(
        f"model_name=`{model_name}` is not supported."
        " Supported options are 'Gemma-1B' and 'Gemma-4B'"
    )

  if loaded_model is None:
    tokenizer, model, params = generation.load_gemma(model_name)
  else:
    tokenizer, model, params = loaded_model

  sampler_output_text = _sample_from_model(
      input_text,
      tokenizer,
      model,
      params,
      max_new_tokens=max_new_tokens,
      sampling_mode=sampling_mode,
  )

  # Convert the input text to tokens and apply the model to generate
  # predictions.
  prompt = tokenizer.encode(input_text, add_bos=True)
  prompt = jnp.asarray(prompt)
  out = model.apply(
      {"params": params},
      tokens=prompt,
      return_last_only=True,  # Only return the last token.
  )
  next_token_logits = out.logits
  output_text = input_text + sampler_output_text

  return output_text, next_token_logits, tokenizer


def prompt_attention_transformer_model(
    input_text: str,
    loaded_model: Tuple[
        gm.text.Gemma3Tokenizer, attention.AttentionWeightGemma3_1B, Any
    ],
    sampling_mode: Literal["random", "greedy"] = "random",
) -> Tuple[
    str,  # Output text.
    jax.Array,  # Next-token probabilities.
    gm.text.Gemma3Tokenizer,  # Tokenizer.
    Dict[str, jax.Array],  # Layer-wise attention weights.
    jax.Array,  # Attention mask.
    Dict[str, Dict[str, jax.Array]],  # Layer-wise QKV matrices.
]:
  """Samples one token and extracts attention weights and matrices from Gemma.

  Used for visualizing and working with attention weights.

  Args:
    input_text: The input prompt for the model.
    loaded_model: A tuple containing the tokenizer, the model, and the
      parameters.
    sampling_mode: Whether to use random or greedy sampling. Supported options
      are 'random' and 'greedy'.

  Returns:
    output_text: The generated text, including the input text and the
      model's output.
    next_token_logits: Logits for the next token (probability distribution).
    tokenizer: The tokenizer used for encoding/decoding the text.
    attention_weights: Dictionary with layer-wise attention weights.
    attention_mask: Attention mask for input.
    qkv: Dictionary with layer-wise Q, K, and V matrices.
  """

  tokenizer, model, params = loaded_model

  sampler_output_text = _sample_from_model(
      input_text, tokenizer, model, params, sampling_mode=sampling_mode
  )

  # Convert the input text to tokens and apply the model to generate
  # predictions.
  prompt = tokenizer.encode(input_text, add_bos=True)
  prompt = jnp.asarray([prompt])
  out = model.apply(
      {"params": params},
      tokens=prompt,
      # Return all states for attention weight re-computation.
      return_last_only=False,
  )
  next_token_logits = out.logits
  attention_weights = out.attention_weights
  qkv = out.qkv
  attention_mask = out.attention_mask
  output_text = input_text + sampler_output_text

  return (
      output_text,
      next_token_logits,
      tokenizer,
      attention_weights,
      attention_mask,
      qkv,
  )
