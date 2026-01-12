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

"""Functions for defining and training an MLP implemented in Keras."""


from typing import List, Optional, Tuple

import jax
import keras


def build_mlp(
    hidden_dims: List[int], n_classes: int, activation: str = "relu"
) -> keras.Model:
  """Initializes an MLP with a SoftMax output layer implemented in Keras.

  Args:
    hidden_dims: A list of dimensions for all hidden layers. The number of
      elements in this list defines the number of hidden layers. Specify an
      empty list for building a model with only a SoftMax layer.
    n_classes: Number of classes for the output layer.
    activation: Activation function for hidden layers. Use "linear" for the
      identity function.

  Returns:
    A keras.Model instance that implements the MLP.
  """

  operations = []

  # Hidden layers.
  for dim in hidden_dims:
    operations.append(keras.layers.Dense(dim, activation=activation))

  # Output layer.
  operations.append(keras.layers.Dense(n_classes, activation="softmax"))

  model = keras.Sequential(operations)
  return model


def train_mlp(
    mlp_model: keras.Model,
    x: jax.Array,
    y: jax.Array,
    epochs: int,
    validation_data: Optional[Tuple[jax.Array, jax.Array]] = None,
    learning_rate: float = 0.01,
    callbacks: Optional[List[keras.callbacks.Callback]] = None,
) -> keras.callbacks.History:
  """This function trains an MLP implemented in Keras with Adam and CE loss.

  Args:
    mlp_model: A keras.Model instance implementing the MLP.
    x: Training features.
    y: Training label IDs.
    epochs: Number of training epochs.
    validation_data: Optional tuple (x_val, y_val) with validation features and
      labels for validating performance after each epoch.
    learning_rate: Learning rate for Adam.
    callbacks: Optional list of callbacks, for example for custom logging after
      each epoch.

  Returns:
    A keras.callbacks.History instance containing the losses and accuracies of
      the training run.
  """

  # Initialize optimizer.
  optimizer = keras.optimizers.AdamW(
      learning_rate=learning_rate, weight_decay=0.005
  )
  # Intialize CE loss.
  loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

  mlp_model.compile(
      optimizer=optimizer, loss=loss_function, metrics=["accuracy"]
  )

  # Train model.
  history = mlp_model.fit(
      x,
      y,
      validation_data=validation_data,
      epochs=epochs,
      verbose=0,
      callbacks=callbacks,
  )

  return history
