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

"""Core architectural layers for a custom transformer model.

This module contains the custom Keras layers that form the building blocks of a
decoder-only transformer model, including embedding, self-attention, and
feed-forward network layers.
"""

from typing import Any, Callable

import jax
from jax import numpy as jnp
import keras
from keras import layers
from keras import ops

# This value is a standard choice, as set in the
# "Attention Is All You Need" paper.
ANGLE_RATE_MULTIPLIER = 10000


# Decorator so that the custom class can be saved and loaded correctly.
@keras.saving.register_keras_serializable()
class TokenAndPositionEmbedding(layers.Layer):
  """Combines token embeddings with positional embeddings.

  This layer creates combined token and positional embeddings for input
  sequences. The `mask_zero=True` setting in the token embeddings allows for
  automatic masking of padded tokens.

  Attributes:
    max_length: The maximum expected sequence length. This determines the range
      of positional embeddings.
    vocabulary_size: The size of the vocabulary. This determines the size of the
      token embedding matrix.
    embedding_dim: The dimensionality of the token and positional embeddings.
    positional_embedding_type: The type of positional embedding to use. It can
      be 'simple' or 'sinusoidal'.

  Call Arguments:
    x: Input tensor of shape (batch_size, sequence_length).

  Returns:
    jax.Array: Output tensor of shape (batch_size, sequence_length, d_model)
        with token and positional embeddings combined.
  """

  def __init__(
      self,
      max_length: int,
      vocabulary_size: int,
      embedding_dim: int,
      positional_embedding_type: str = "sinusoidal",
  ):
    super().__init__()

    self.embedding_dim = embedding_dim
    self.max_length = max_length
    self.positional_embedding_type = positional_embedding_type

    # Set mask_zero=True so that Keras generates a mask for padded tokens.
    self.token_emb = layers.Embedding(
        input_dim=vocabulary_size, output_dim=embedding_dim, mask_zero=True
    )

    if self.positional_embedding_type == "simple":
      self.pos_emb = layers.Embedding(
          input_dim=max_length, output_dim=embedding_dim
      )
    elif self.positional_embedding_type == "sinusoidal":
      self.pos_emb = self.positional_encoding(
          length=max_length, depth=embedding_dim
      )
    else:
      raise NotImplementedError(
          "Positional embedding type"
          f" {self.positional_embedding_type}"
          " not implemented."
      )

  def positional_encoding(
      self, length: int, depth: int
  ) -> Callable[[Any], jax.Array]:
    """Creates a positional encoding for a sequence of tokens.

    This approach uses sine and cosine functions at varying frequencies to
    create a unique positional representation for each token in the sequence.

    Args:
      length: The length of the sequence (number of tokens).
      depth: The dimensionality of the encoding (must be even).

    Returns:
      A function that returns an array of shape (length, depth) representing
      the positional encoding. This is a function to make it compatible with the
      simple embedding layer.
    """

    depth = depth // 2  # Use integer division to ensure an integer depth.

    positions = jnp.arange(length)[:, jnp.newaxis]  # (seq, 1)
    depths = jnp.arange(depth)[jnp.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (ANGLE_RATE_MULTIPLIER**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = jnp.concatenate(
        [jnp.sin(angle_rads), jnp.cos(angle_rads)], axis=-1
    )

    pos_encoding_matrix = ops.cast(pos_encoding, dtype="float32")

    def apply(*args) -> jax.Array:  # pylint: disable=unused-argument
      return pos_encoding_matrix[jnp.newaxis, :, :]

    return apply

  def call(self, x: jax.Array) -> jax.Array:
    """Applied and combines token embeddings with positional embeddings.

    Args:
      x: Input tensor of shape (batch_size, sequence_length).

    Returns:
      Output tensor of shape (batch_size, sequence_length, d_model) with token
          and positional embeddings combined.
    """
    token_embeddings = self.token_emb(x)

    if self.positional_embedding_type == "sinusoidal":
      # This factor sets the relative scale of the embedding
      # and positonal_encoding.
      token_embeddings *= ops.sqrt(
          ops.cast(self.embedding_dim, dtype="float32")
      )
      position_embeddings = self.pos_emb(None)
    else:
      # Defaults to simple `positional_embedding_type`.
      positions = ops.arange(0, self.max_length, 1)
      position_embeddings = self.pos_emb(positions)

    return token_embeddings + position_embeddings


# Decorator so that the custom class can be saved and loaded correctly.
@keras.saving.register_keras_serializable()
class TransformerBlock(layers.Layer):
  """A single transformer block.

  The transformer block is a fundamental component of the transformer
  architecture, which is commonly used for sequence-based tasks. It consists
  of a MultiHeadAttention layer followed by a feed-forward network,
  with layer normalization and dropout applied at each step.

  Example:
    transformer_block = TransformerBlock(embedding_dim=256, num_heads=8,
                                         mlp_dim=1024)
    output = transformer_block(inputs)

  Attributes:
    embedding_dim: The dimensionality of the input embedding (also the output
      size of the attention layer).
    num_heads: The number of attention heads in the multi-head attention
      mechanism.
    mlp_dim: The number of units in the feed-forward network.
    dropout_rate: Dropout rate, between 0 and 1.
    activation_function: The activation function to use in the feed-forward
      network.
    seed: Random seed for dropout and attention layers to ensure
      reproducibility.

  Call Arguments:
    inputs: Input tensor of shape (batch_size, sequence_length, d_model).

  Returns:
    The output of the Transformer block after applying the multi-head attention,
        feed-forward network, layer normalization, and residual connections.
  """

  def __init__(
      self,
      embedding_dim: int,
      num_heads: int,
      mlp_dim: int,
      dropout_rate: float = 0.0,
      activation_function: str = "relu",
  ):
    super().__init__()

    self.self_attention = MultiHeadSelfAttention(
        embedding_dim, num_heads, dropout_rate
    )
    self.feed_forward = FeedForwardNetwork(
        embedding_dim, mlp_dim, dropout_rate, activation_function
    )

  def call(self, inputs: jax.Array) -> jax.Array:
    """Applies a single transformer block to the input tensor.

    Notes:
      - The transformer block follows the architecture with residual connections
        and layer normalization.

    Args:
      inputs: The input tensor of shape (batch_size, seq_len, embed_dim).

    Returns:
      The output tensor of shape (batch_size, seq_len, embed_dim) after applying
          the transformer block.
    """

    # First block: masked self-attention.
    attn_output = self.self_attention(inputs)

    # Second block: feedforward network applied on attention output.
    ffn_output = self.feed_forward(attn_output)

    return ffn_output


# Decorator so that the custom class can be saved and loaded correctly.
@keras.saving.register_keras_serializable()
class FeedForwardNetwork(layers.Layer):
  """Feed forward network layer.

  This layer implements a two-layer feedforward network with a residual
  connection and layer normalization. It's a common component in transformer
  architectures, used to introduce non-linearity and improve the model's ability
  to capture complex relationships.

  Attributes:
    embedding_dim: The dimensionality of the embedding space.
    mlp_dim: The dimensionality of the hidden layer in the feedforward network
      (often larger than embedding_dim).
    dropout_rate: The dropout rate applied to the output of the feedforward
      network.
    activation_function: The activation function used in the first dense layer.

  Call Arguments:
    x: Input tensor of shape (batch_size, sequence_length, embedding_dim).

  Returns:
    Output tensor of shape (batch_size, sequence_length, embedding_dim) with
        the feed-forward network applied.
  """

  def __init__(
      self,
      embedding_dim: int,
      mlp_dim: int,
      dropout_rate: float = 0.0,
      activation: str = "relu",
  ):
    super().__init__()
    # Define a two-layer feedforward network.
    self.ffn = keras.Sequential([
        # Expand dimension.
        layers.Dense(mlp_dim, activation=activation),
        # Project back to embedding_dim.
        layers.Dense(embedding_dim),
    ])
    self.dropout = layers.Dropout(dropout_rate)
    self.layernorm = layers.LayerNormalization()

  def call(self, x: jax.Array) -> jax.Array:
    """Applies the feedforward network to the input tensor.

    Args:
      x: Input tensor of shape (batch_size, sequence_length, embedding_dim).

    Returns:
      Output tensor of shape (batch_size, sequence_length, embedding_dim).
    """

    ffn_output = self.ffn(x)
    ffn_output = self.dropout(ffn_output)
    # Add residual connection followed by layer normalization.
    output = self.layernorm(x + ffn_output)
    return output  # type: ignore


# Decorator so that the custom class can be saved and loaded correctly.
@keras.saving.register_keras_serializable()
class MultiHeadSelfAttention(layers.Layer):
  """Multi-head self-attention Layer.

  This layer implements multi-head self-attention, a key component in
  transformer architectures. It computes attention weights for each head and
  applies them to the input to generate a contextually enriched representation.

  Attributes:
    embedding_dim: The dimensionality of the embedding space.
    num_heads: The number of attention heads.
    dropout_rate: The dropout rate applied to the attention output.

  Call Arguments:
    x: Input tensor of shape (batch_size, sequence_length, d_model).

  Returns:
    Output tensor of shape (batch_size, sequence_length, embedding_dim)
        with self-attention applied.
  """

  def __init__(
      self, embedding_dim: int, num_heads: int, dropout_rate: float = 0.0
  ):
    super().__init__()

    # Multi-head self-attention layer.
    self.mha = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embedding_dim
    )
    self.dropout = layers.Dropout(dropout_rate)
    self.layernorm = layers.LayerNormalization()

  def call(self, x: jax.Array) -> jax.Array:
    """Applies multi-head self-attention to the input tensor.

    Args:
      x: Input tensor of shape (batch_size, sequence_length, embedding_dim).

    Returns:
      Output tensor of shape (batch_size, sequence_length, embedding_dim).
    """

    # Apply self-attention. The mask is typically a look-ahead mask.
    attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
    attn_output = self.dropout(attn_output)

    # Add residual connection followed by layer normalization.
    output = self.layernorm(x + attn_output)

    return output
