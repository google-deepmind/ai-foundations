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

"""A Keras callback to generate sample text during model training.

This module defines the TextGenerator callback, which can be used with
`model.fit()` to monitor a language model's progress by generating and
printing sample text at the end of specified epochs.
"""

import random
from typing import Any, Dict, List, Optional

from ai_foundations import generation
import jax
import jax.numpy as jnp
import keras
from keras import ops


class TextGenerator(keras.callbacks.Callback):
  """A callback to generate text from a trained model.

    1. Feed an initial prompt to the model.
    2. Predict probabilities for the next token.
    3. Sample the next token and add it to the input for the next prediction.

  Attributes:
    max_tokens: Number of tokens to generate.
    start_tokens: Token indices for the initial prompt.
    tokenizer: The tokenizer used to decode generated token indices.
    pad_token_id: The padding token ID.
    print_every: Print the generated text every `print_every` epochs.
    **callback_kwargs: Any additional keyword arguments.
  """

  def __init__(
      self,
      max_tokens: int,
      start_tokens: List[int],
      tokenizer: Any,
      pad_token_id: int = 0,
      print_every: int = 1,
      **callback_kwargs: Dict[str, Any],
  ):
    super().__init__(**callback_kwargs)

    self.max_tokens = max_tokens
    self.start_tokens = start_tokens
    self.tokenizer = tokenizer
    self.print_every = print_every
    self.pad_token_id = pad_token_id  # ID for padding token.

  def on_epoch_end(
      self, epoch: int, logs: Dict[str, Any] | None = None
  ) -> None:
    """Generate and print text after each epoch based on starting tokens.

    Args:
      epoch: The current epoch number.
      logs: Logs from the training process.
    """

    if self.model is None:
      return

    max_length = self.model.layers[0].output.shape[1]
    # Make a copy of the start tokens.
    start_tokens = list(self.start_tokens)
    if (epoch + 1) % self.print_every != 0:
      return

    num_tokens_generated = 0
    tokens_generated = []

    # Introduce randomness by re-intializing JAX RNG with a different seed on
    # each call. While this harms reproducability, it avoids having to pass a
    # JAX key on every call, which would likely be confusing to learners.
    main_key = jax.random.PRNGKey(random.randint(0, 1000000))

    while num_tokens_generated < self.max_tokens:
      pad_len = max_length - len(start_tokens)
      sample_index = len(start_tokens) - 1

      # Handle padding to ensure the sequence is of the correct length.
      if pad_len < 0:
        x = start_tokens[:max_length]
        sample_index = max_length - 1
      elif pad_len > 0:
        x = start_tokens + [self.pad_token_id] * pad_len
      else:
        x = start_tokens

      x = jnp.array([x])
      y = self.model.predict(x, verbose=0)

      # Convert logits to probabilities using softmax.
      probabilities = ops.softmax(y, axis=-1)

      key, main_key = jax.random.split(main_key)
      sample_token = generation.random_decoding(
          probabilities[0][sample_index], key
      )

      tokens_generated.append(sample_token)
      start_tokens.append(sample_token)
      num_tokens_generated = len(tokens_generated)

    # Combine the starting tokens with the generated tokens.
    output_tokens = self.start_tokens + tokens_generated
    output_tokens = list(map(int, output_tokens))

    # Decode and print the generated text.
    txt = self.tokenizer.decode(output_tokens)
    print("Generated text:\n", txt, "\n")


class CustomAccuracyPrinter(keras.callbacks.Callback):
  """Custom Keras callback function to print training progress in Lab 3.12.

  Attributes:
    print_every: Print the training progress every `print_every` epochs.
  """

  def __init__(self, print_every: int = 1):
    self.print_every = print_every

  def on_epoch_end(
      self, epoch: int, logs: Optional[Dict[str, Any]] = None
  ) -> None:
    """Prints training and validation metrics at the end of each epoch.

    This function is executed at the end of each epoch. It prints the
    training loss and the validation loss and training and validation
    accuracies, if available. If self.print_every is greater than 1,
    updates are only printed every self.print_every epoch.

    Note that at this stage, learners have not learned the difference between
    validation and test sets and therefore validation loss and validation
    accuracy is renamed to test loss and test accuracy.

    Args:
      epoch: The current epoch number.
      logs: A dictionary containing the current loss and any other metrics that
        were specified when compiling the model.
    """

    if (epoch + 1) % self.print_every != 0:
      return

    if logs is not None:
      log_parts = []
      log_parts.append(f"Epoch {epoch}: Training loss: {logs['loss']:.5f}")
      if "accuracy" in logs:
        log_parts.append(f"training accuracy: {logs['accuracy']*100:.2f}%")
      if "val_loss" in logs:
        log_parts.append(f"test loss: {logs['val_loss']:.5f}")
      if "val_accuracy" in logs:
        log_parts.append(f"test accuracy: {logs['val_accuracy']*100:.2f}%")

      print(", ".join(log_parts))
