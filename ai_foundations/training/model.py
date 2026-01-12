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

"""A function to build and compile a Transformer model.

This module provides a primary function to assemble the model layers,
configure the optimizer and loss, and return a compiled Keras model ready for
training.
"""

from typing import Literal

from ai_foundations.training.losses import CustomMaskPadLoss
from ai_foundations.transformers import TokenAndPositionEmbedding
from ai_foundations.transformers import TransformerBlock
import keras
from keras import layers


def create_model(
    vocabulary_size: int,
    max_length: int,
    embedding_dim: int = 256,
    mlp_dim: int = 256,
    num_heads: int = 2,
    num_blocks: int = 1,
    optimizer: Literal["adamw", "sgd"] = "adamw",
    learning_rate: float = 5e-4,
    dropout_rate: float = 0.0,
    activation_function: str = "relu",
    pad_token_id: int = 0,
) -> keras.Model:
  """Creates a transformer-based model for sequence processing tasks.

  Example:
    model = create_model(vocabulary_size=5000, max_length=100,
                         embedding_dim=256, mlp_dim=512,
                         num_heads=8, num_blocks=2)
    print(model.summary())

  Notes:
    - The model uses causal (masked) attention to ensure that each token only
      attends to previous tokens and not future tokens.
    - The final dense layer produces a logit over the vocabulary for each token
      in the sequence.
    - The loss function is `CustomMaskPadLoss`, which ignores padding tokens in
      the loss computation.

  Args:
    vocabulary_size: The size of the vocabulary, i.e., the number of unique
      tokens.
    max_length: The maximum length of the input sequences.
    embedding_dim: The dimensionality of the embedding space.
    mlp_dim: The number of units in the feed-forward network of each transformer
      block.
    num_heads: The number of attention heads in the multi-head attention
      mechanism.
    num_blocks: The number of transformer blocks to stack in the model.
    optimizer: The optimizer to use for training, either 'adamw' (Adam with
      weight decay) or 'sgd'.
    learning_rate: The learning rate for the optimizer.
    dropout_rate: The dropout rate to prevent overfitting.
    activation_function: The activation function to use in the feed-forward
      network of each transformer block.
    pad_token_id: The ID used to represent padding tokens in the sequence. This
      is used to mask padded tokens in the loss calculation.

  Returns:
    The compiled Keras model which outputs the probability of the next token
        prediction.

  Raises:
      NotImplementedError: If an unsupported optimizer is specified.
  """
  # Create input layer.
  inputs = layers.Input(shape=(max_length,), dtype="int32")

  # Embedding layer that combines token and positional embeddings.
  embedding_layer = TokenAndPositionEmbedding(
      max_length, vocabulary_size, embedding_dim
  )
  x = embedding_layer(inputs)

  # Apply a stack of transformer blocks.
  for _ in range(num_blocks):
    transformer_block = TransformerBlock(
        embedding_dim,
        num_heads,
        mlp_dim,
        dropout_rate=dropout_rate,
        activation_function=activation_function,
    )
    x = transformer_block(x)

  # Apply dense layer, it returns raw logit of next token prediction.
  outputs = layers.Dense(vocabulary_size)(x)

  # Build the model.
  model = keras.Model(inputs=inputs, outputs=outputs)

  # Set up optimizer based on input string.
  optimizer_instance = get_optimizer(optimizer, learning_rate)

  # Define the loss function and compile the model.
  loss_fn = CustomMaskPadLoss(pad_token_id=pad_token_id)
  model.compile(optimizer=optimizer_instance, loss=loss_fn)

  # Final output layer returns the probability of next token prediction.
  return model


def get_optimizer(
    optimizer_name: Literal["adamw", "sgd"], learning_rate: float
) -> keras.optimizers.Optimizer:
  """Helper function to get the appropriate optimizer instance.

  Args:
    optimizer_name: The name of the optimizer.
    learning_rate: The learning rate for the optimizer.

  Returns:
    The corresponding optimizer instance.

  Raises:
    NotImplementedError: If an unsupported optimizer is specified.
  """

  if optimizer_name.lower() == "sgd":
    return keras.optimizers.SGD(learning_rate=learning_rate)
  elif optimizer_name.lower() == "adamw":
    return keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=0.005,
        gradient_accumulation_steps=None,
    )
  else:
    raise NotImplementedError(f"Optimizer {optimizer_name} is not implemented.")
