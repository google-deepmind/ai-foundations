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

"""A custom Keras loss function for handling padded sequence data.

This module defines the CustomMaskPadLoss class, which wraps the standard
SparseCategoricalCrossentropy loss to ignore a specified padding token ID
during training.
"""

from typing import Any

import jax
import keras


# Decorator so that the custom class can be saved and loaded correctly.
@keras.saving.register_keras_serializable()
class CustomMaskPadLoss(keras.losses.Loss):
  """Custom loss function for masked padding in sequence-based tasks.

  This loss function computes the SparseCategoricalCrossentropy
  loss while ignoring the padding tokens (specified by `pad_token_id`).
  The padding tokens are not included in the loss calculation,
  allowing the model to focus on meaningful tokens during training.

  Attributes:
    name: The name of the loss function, used by Keras.
    pad_token_id: The ID of the padding token. If provided, padding tokens will
      be ignored during loss calculation. If None, no padding is masked.
    **kwargs: Additional keyword arguments.
  """

  def __init__(
      self,
      pad_token_id: int | None = None,
      **kwargs: Any,
  ):
    super().__init__(name="custom_mask_pad_loss", **kwargs)
    self.pad_token_id = pad_token_id

  def call(self, y_true: jax.Array, y_pred: jax.Array) -> jax.Array:
    """Computes the custom loss.

    The call function optionally masks the padding tokens and normalizes
    the loss by the number of non-masked tokens. The loss is computed using
    the SparseCategoricalCrossentropy loss function.

    Args:
      y_true: The true labels.
      y_pred: The model's predictions.

    Returns:
      The computed loss.
    """
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        # The model's output is a probability distribution. If
        # it is raw logit, this should be True.
        from_logits=True,
        ignore_class=self.pad_token_id,
        # Average the loss across the batch size.
        reduction="sum_over_batch_size",
    )

    loss = loss_fn(y_true, y_pred)
    return loss
