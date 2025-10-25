# Copyright 2025 Google LLC
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

"""Implements functions for generating plots and visualizations."""

import collections
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import keras
import matplotlib.colors as pltcolors
import matplotlib.lines as pltlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px


Counter = collections.Counter
namedtuple = collections.namedtuple

COLOR_PALETTE = ["#FFC20A", "#0C7BDC", "#71A816"]

HyperParameterConfig = namedtuple(
    "HyperParameterConfig",
    [
        "hidden_dims",
        "dropout_rate",
        "weight_decay_strength",
        "use_early_stopping",
    ],
)


def plot_next_token(
    probs_or_logits: np.ndarray | Dict[str, float],
    prompt: str,
    keep_top: int = 30,
    tokenizer: Optional[Any] = None,
):
  """Plot the probability distribution of the next tokens.

  This function generates a bar plot showing the top `keep_top` tokens by
  probability.

  Args:
    probs_or_logits: The raw logits output by the model or the probability
      distribution for the next token prediction. Can also be a dictionary as
      returned by an n-gram model.
    prompt: The input prompt used to generate the next token predictions.
    keep_top: The number of top tokens to display in the plot.
    tokenizer: The tokenizer used to decode token IDs to human-readable text.

  Returns:
    Displays a plot showing the probability distribution of the top tokens.
  """

  if isinstance(probs_or_logits, dict):
    # Extract probabilities from n-gram dictionary.
    probs = jnp.array(list(probs_or_logits.values()))
  elif np.isclose(probs_or_logits.sum(), 1):
    probs = probs_or_logits
  else:
    # Apply softmax to logits to get probabilities.
    probs = jax.nn.softmax(probs_or_logits)

  # Select the top `keep_top` tokens by probability.
  indices = jnp.argsort(probs)

  # Reverse to get highest probabilities first.
  indices = indices[-keep_top:][::-1]

  # Get the probabilities and corresponding tokens.
  probs = probs[indices].astype(np.float32)

  if tokenizer is not None:
    # Decode indices using tokenizer.
    tokens = [repr(tokenizer.decode(index.item())) for index in indices]
  elif isinstance(probs_or_logits, dict):
    # Extract tokens from n-gram dictionary.
    tokens = list(probs_or_logits.keys())
  else:
    # Return the raw indices if no decoding information is supplied.
    tokens = indices

  # Create the bar plot using Plotly.
  fig = px.bar(x=tokens, y=probs)

  # Customize the plot layout.
  fig.update_layout(
      title=(
          f'Probability distribution of next tokens given the prompt="{prompt}"'
      ),
      xaxis_title="Tokens",
      yaxis_title="Probability",
  )

  # Display the plot.
  fig.show()


def plot_word_frequencies(token_counts: Counter[str]):
  """Plot the token frequencies of a token_counts Counter.

  Args:
    token_counts: A collections.Counter object of tokens and their frequencies.
  """

  # Create an array of ranks from 1 to the total number of unique words. The
  # most frequent word has rank 1, the second most frequent has rank 2, etc.
  ranks = jnp.arange(1, len(token_counts) + 1)

  # Extract frequency values from the token_counts dictionary.
  frequencies = jnp.array([freq for _, freq in token_counts.most_common()])

  # Extract unique tokens from the token_counts dictionary.
  words = [word for word, _ in token_counts.most_common()]

  plt.figure(figsize=(10, 8))
  fontsize = 14

  # Create the log-log plot.
  plt.loglog(ranks, frequencies, marker="o", linestyle="none", label="word")

  # Annotate the first five most frequent words.
  for i in range(min(5, len(words))):
    rank, frequency = float(ranks[i]), float(frequencies[i])
    plt.annotate(words[i], (rank, frequency), fontsize=fontsize, ha="right")

  # Annotate the five least frequent words, with offset to avoid overlap.
  for i in range(max(0, len(words) - 5), len(words)):
    rank, frequency = float(ranks[i]), float(frequencies[i])
    plt.annotate(
        words[i],
        (rank, frequency),
        fontsize=fontsize,
        ha="right",
        va="bottom",  # Align text vertically to the bottom.
        xytext=(0, (i - (len(words) - 5)) * 20),  # Offset vertically.
        textcoords="offset points",
    )  # Offset relative to the point.

  # Label the axes and add a title.
  plt.xlabel("Rank $r$ of word (log scale)", fontsize=fontsize)
  plt.ylabel("Frequency $f$ of word (log scale)", fontsize=fontsize)
  plt.title(
      "Frequency vs. Rank of Tokens in the Africa Galore Dataset",
      fontsize=fontsize,
  )
  _ = plt.legend(fontsize=fontsize)  # Display the legend.

  plt.show()


def plot_data_and_decision_boundary(
    input_features: jax.Array,
    labels: List[str],
    weight_vector: Optional[jax.Array] = None,
    bias_term: Optional[jax.Array] = None,
    title: str = "2D Prompt Embeddings and Decision Boundary",
    provide_feedback: bool = False,
):
  """Plots the 2D data points and the decision boundary if weights and bias are provided.

  Args:
    input_features: Input features (data points).
    labels: List of labels for each datapoint.
    weight_vector: Weight vector defining the decision boundary. This should be
      either a 2-dimensional JAX Array or, if no separate bias term is defined,
      a 3-dimensional JAX Array where the first component is the bias term.
    bias_term: Bias term for the decision boundary.
    title: Title for the plot.
    provide_feedback: If true, the function provides feedback to learners on the
      accuracy of their decision boundary.

  Raises:
    ValueError if a dataset with more than two classes is specified.
  """

  classes = sorted(list(set(labels)))
  if len(classes) != 2:
    raise ValueError("The function only works for binary classification tasks.")

  plt.figure(figsize=(6, 6))

  # Convert labels to numeric values for plotting.
  numeric_labels = jnp.where(labels == classes[1], 1, 0)

  # If first column is a 1 for bias term, remove feature.
  if input_features.shape[1] == 3:
    input_features = input_features[:, 1:]

  colors = COLOR_PALETTE[0:2]
  # Plot data points.
  for i in range(2):
    plt.scatter(
        input_features[numeric_labels == i][:, 0],
        input_features[numeric_labels == i][:, 1],
        color=colors[i],
        label=f"Next token: '{classes[i]}'",
    )

  classification_errors = []
  # Support specification of weight_vector with bias term.
  if (
      weight_vector is not None
      and bias_term is None
      and weight_vector.shape[0] == 3
  ):
    bias_term = weight_vector[0]  # First component is the bias term.
    # Second and third components are the weight vector.
    weight_vector = weight_vector[1:]

  # Highlight points near the decision boundary (orthogonal points).
  if weight_vector is not None and bias_term is not None:
    # Calculate inner products for all points.
    inner_products = jnp.dot(input_features, weight_vector) + bias_term

    # Identify points near the decision boundary (inner product close to zero).
    # 0.1 is the threshold for "near zero".
    orthogonal_points = input_features[jnp.abs(inner_products) < 0.1]

    if orthogonal_points.shape[0] > 0:
      plt.scatter(
          orthogonal_points[:, 0],
          orthogonal_points[:, 1],
          color=COLOR_PALETTE[2],
          edgecolor="black",
          s=100,
          label="Orthogonal Points",
      )

    # Draw the decision boundary.
    x_vals = jnp.linspace(-2, 2, 100)
    y_vals = -(weight_vector[0] * x_vals + bias_term) / (
        weight_vector[1] + 1e-8
    )
    plt.plot(x_vals, y_vals, "k--", label="Decision Boundary")

    # Draw the weight vector.
    plt.quiver(
        0,
        -bias_term / weight_vector[1],
        weight_vector[0],
        weight_vector[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=COLOR_PALETTE[2],
        label="Weight vector",
    )

    # Compute number of classification errors.
    for i in range(2):
      inner_products = (
          jnp.dot(input_features[numeric_labels == i], weight_vector)
          + bias_term
      )
      sign_factor = 2 * i - 1
      classification_errors.append(
          sum(jnp.where(inner_products * sign_factor < 0, 0, 1))
      )

  plt.axhline(0, color="gray", linewidth=0.5)
  plt.axvline(0, color="gray", linewidth=0.5)
  plt.xlim(-2, 2)
  plt.ylim(-2, 2)
  plt.legend()
  plt.title(title)
  plt.grid(True)
  plt.show()

  if provide_feedback and weight_vector is not None and bias_term is not None:
    print("Classification errors:")
    print(f"{classes[0]}: {classification_errors[0]}")
    print(f"{classes[1]}: {classification_errors[1]}")

    if sum(classification_errors) == 0:
      print(
          "\n\n✅ Well done! Your decision boundary correctly separates"
          " all data points."
      )
    else:
      print(
          f"\n\n❌ There are still {sum(classification_errors)}"
          " datapoints that are not classified correctly.\nTry adjusting"
          " the weights and the bias again to move the decision boundary."
      )


def plot_loss_curve(history: Dict[str, Any]):
  """Plot the loss curve(s).

  Args:
    history: A dictionary as output by a Keras training run. The training loss
      should be stored under the key "loss", the optional test loss should be
      stored under the key "val_loss".

  Raises:
    ValueError if history does not contain an entry for key "loss".
  """

  if "loss" not in history:
    raise ValueError(
        "Argument history must include training loss stored under key 'loss'."
    )

  max_loss = max(history["loss"])
  plt.plot(history["loss"], label="Train Loss", color=COLOR_PALETTE[1])
  if "val_loss" in history:
    plt.plot(history["val_loss"], label="Test Loss", color=COLOR_PALETTE[0])
    max_val_loss = max(history["val_loss"])
    max_loss = max(max_loss, max_val_loss)

  plt.ylim(0, max_loss * 1.1)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Loss Curve")
  plt.legend()
  plt.grid(True)
  plt.show()


def plot_accuracy_curve(history: Dict[str, Any]):
  """Plot the accuracy curve(s).

  Args:
    history: A dictionary as output by a Keras training run. The training
      accuracy should be stored under the key "accuracy", the optional test loss
      should be stored under the key "val_accuracy".

  Raises:
    ValueError if history does not contain an entry for key "loss".
  """

  if "accuracy" not in history:
    raise ValueError(
        "Argument history must include training accuracy stored under key"
        " 'accuracy'."
    )

  plt.plot(history["accuracy"], label="Train accuracy", color=COLOR_PALETTE[1])
  if "val_loss" in history:
    plt.plot(
        history["val_accuracy"], label="Test accuracy", color=COLOR_PALETTE[0]
    )
  plt.ylim(0.0, 1.05)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Accuracy Curve")
  plt.legend()
  plt.grid(True)
  plt.show()


def plot_data_and_mlp(
    features: jax.Array,
    label_ids: jax.Array,
    labels: List[str],
    features_test: Optional[jax.Array] = None,
    label_ids_test: Optional[jax.Array] = None,
    mlp_model: Optional[keras.Model] = None,
    title: Optional[str] = None,
):
  """Plots 2D feature data, optionally with test data and a decision boundary.

  This function generates a scatter plot for 2D classification data. It
    visualizes the training points, and optionally, test points (marked with
    asterisks). If a trained Keras MLP model is provided, it also plots the
    model's decision boundaries as filled contours.

  Args:
    features: A 2D JAX array of training features, where each row is a data
      point.
    label_ids: A 1D JAX array of integer labels for the training features.
    labels: A list of string names corresponding to the integer label_ids.
    features_test: An optional 2D JAX array of test features.
    label_ids_test: An optional 1D JAX array of integer labels for the test
      features.
    mlp_model: An optional trained Keras model used to plot decision boundaries.
    title: An optional title for the plot.
  """

  if features_test is not None and label_ids_test is not None:
    X = jnp.vstack([features, features_test])  # pylint: disable=invalid-name
  else:
    X = features  # pylint: disable=invalid-name

  h = 0.02
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

  colors = COLOR_PALETTE
  cmap = pltcolors.ListedColormap(colors)

  if mlp_model is not None:
    grid = jnp.c_[xx.ravel(), yy.ravel()]
    Z = mlp_model(grid)  # pylint: disable=invalid-name
    Z = jnp.argmax(Z, axis=1)  # pylint: disable=invalid-name
    Z = np.array(Z).reshape(xx.shape)  # pylint: disable=invalid-name
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=cmap)

  for i in range(3):
    plt.scatter(
        features[label_ids == i][:, 0],
        features[label_ids == i][:, 1],
        label=labels[i],
        color=colors[i],
    )
    if features_test is not None and label_ids_test is not None:
      plt.scatter(
          features_test[label_ids_test == i][:, 0],
          features_test[label_ids_test == i][:, 1],
          color=colors[i],
          marker="*",
      )

  dataset_legend = None
  if features_test is not None and label_ids_test is not None:
    # Add second legend.
    dataset_legend_handle = [
        pltlines.Line2D(
            [], [], marker=".", color="black", markersize=10, linestyle="None"
        ),
        pltlines.Line2D(
            [], [], marker="*", color="black", markersize=10, linestyle="None"
        ),
    ]

    dataset_legend = plt.legend(
        handles=dataset_legend_handle,
        labels=["train", "test"],
        loc="center right",
        bbox_to_anchor=(1.0, 0.6),
        title="Dataset",
    )

  if title is not None:
    plt.title(title)
  plt.xlabel("Embedding dim 1")
  plt.ylabel("Embedding dim 2")
  plt.legend()
  if dataset_legend is not None:
    plt.gca().add_artist(dataset_legend)
  plt.grid(True)
  plt.show()


def visualize_mlp_architecture(hidden_dims: List[int], n_classes: int):
  """Visualizes the architecture of an MLP.

  Args:
    hidden_dims: A list where each integer is the number of neurons in a hidden
      layer.
    n_classes: The number of neurons in the output layer.
  """
  # Combine all layer dimensions: Input (2) -> Hidden -> Output.
  layer_dims = [2] + hidden_dims + [n_classes]
  num_layers = len(layer_dims)

  # Determine the maximum width needed for any layer for figure scaling.
  max_display_width = max([min(d, 8) for d in layer_dims] + [1])

  fig, ax = plt.subplots(figsize=(max_display_width, num_layers * 1.0))

  # Calculate neuron positions.
  neuron_positions = []
  # Layers are positioned from bottom (y=0) to top.
  layer_y_coords = np.linspace(0, num_layers - 1, num_layers) * 3

  for i, dim in enumerate(layer_dims):
    layer_positions = []
    y = layer_y_coords[i]

    # For horizontal positioning, we consider the display width (max 8).
    display_width = min(dim, 8)
    x_start = (max_display_width - display_width) / 2.0

    if dim > 10:
      # First 5 neurons.
      for j in range(5):
        layer_positions.append((x_start + j * 0.66, y))
      # Last 5 neurons, leaving a gap for dots.
      for j in range(5):
        layer_positions.append((x_start + j * 0.66 + 4.67, y))
    else:
      # All neurons for smaller layers.
      for j in range(dim):
        layer_positions.append((x_start + j, y))
    neuron_positions.append(layer_positions)

  # Draw connections (lines) between layers.
  for i in range(num_layers - 1):
    # Connect all neurons of one layer to all neurons of the next.
    for start_pos in neuron_positions[i]:
      for end_pos in neuron_positions[i + 1]:
        ax.plot(
            [start_pos[0], end_pos[0]],
            [start_pos[1], end_pos[1]],
            "k-",
            alpha=0.3,
        )

  # Draw neurons (circles), labels, and dots.
  for i, layer_positions_list in enumerate(neuron_positions):
    dim = layer_dims[i]
    y = layer_y_coords[i]

    # Determine layer type and color.
    if i == 0:
      label = f"Input layer\n(dim={dim})"
      color = COLOR_PALETTE[0]
    elif i == num_layers - 1:
      label = f"Output layer\n(dim={dim})"
      color = COLOR_PALETTE[2]
    else:
      # Label hidden layers sequentially from bottom up.
      label = f"Hidden layer {i}\n(dim={dim})"
      color = COLOR_PALETTE[1]

    # Draw the neurons.
    for pos in layer_positions_list:
      circle = plt.Circle(
          pos, radius=0.25, facecolor=color, edgecolor="black", zorder=4
      )
      ax.add_patch(circle)

    # Add layer label to the left of the layer.
    ax.text(
        -2.5, y, label, ha="center", va="center", fontsize=12, fontweight="bold"
    )

    # If the layer is large, draw ellipsis.
    if dim > 10:
      display_width = 8
      x_start = (max_display_width - display_width) / 2.0
      dots_x = x_start + 3.65
      ax.text(dots_x, y, "...", ha="center", va="center", fontsize=20, zorder=5)

    # Add activation function label between layers.
    if i < num_layers - 1:
      activation_y = (y + layer_y_coords[i + 1]) / 2
      activation_label = "SoftMax" if i == num_layers - 2 else "ReLU"
      # Center the activation label horizontally.
      center_x = max_display_width / 2.0
      ax.text(
          center_x,
          activation_y,
          activation_label,
          ha="center",
          va="center",
          fontsize=10,
          style="italic",
          bbox=dict(boxstyle="round,pad=0.3", fc="wheat", ec="none", alpha=0.8),
          zorder=5,
      )

  # Final touches.
  ax.set_aspect("equal")
  ax.axis("off")
  fig.tight_layout()
  plt.show()


def plot_spiral_data(dataset_url: str):
  """Dowloads and plots a four-class spiral dataset.

  Args:
    dataset_url: URL to CSV file with spiral dataset.
  """
  # Load the CSV file into a Pandas DataFrame.
  df = pd.read_csv(dataset_url)

  # Extract features and labels.
  X = df[["Feature_1", "Feature_2"]].values  # pylint: disable=invalid-name
  y = df["Label"].values

  # Plot the data.
  with plt.style.context(("tableau-colorblind10",)):
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors="k")
    plt.title("Spiral Dataset with 4 Classes")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.grid(True)
    plt.show()


def visualize_hyperparameter_loss(
    experiment_log: dict[HyperParameterConfig, float],
):
  """A function to visualize the hyperparamter-loss curves.

  The plots are constructed from the experiment log that learners constructed
  in the lab "Mitigate Overfitting". It checks whether learners filled in all
  necessary entries in their experiment log and for the existing entries, it
  creates the plots.

  Args:
    experiment_log: A dictionary with hyperparameter settings as keys and the
      corresponding test loss as values.
  """

  hyperparameter_types = [
      "dropout_rate",
      "weight_decay_strength",
      "use_early_stopping",
  ]

  baseline_config = HyperParameterConfig(
      hidden_dims=(10, 1000),
      dropout_rate=0.0,
      weight_decay_strength=0.0,
      use_early_stopping=False,
  )

  if baseline_config not in experiment_log:
    print(
        "❌ The results of the baseline experiment are not in the experiment"
        " log.\n Make sure you ran the cell that added the loss of the baseline"
        " experiment to the log."
    )
    return

  baseline_loss = experiment_log[baseline_config]

  for hyperparam_type in hyperparameter_types:
    hyperparam_values = []
    losses = []
    if hyperparam_type == "use_early_stopping":
      hyperparam_values.append(False)
    else:
      hyperparam_values.append(0.0)

    losses.append(baseline_loss)

    for config in experiment_log:
      if config == baseline_config:
        continue

      if (
          hyperparam_type != "use_early_stopping"
          and getattr(config, hyperparam_type) != 0.0
      ) or (
          hyperparam_type == "use_early_stopping"
          and getattr(config, hyperparam_type)
      ):
        hyperparam_values.append(getattr(config, hyperparam_type))
        losses.append(experiment_log[config])

    if len(hyperparam_values) < 2:
      print(
          "❌ There are still losses missing for the hyperparameter"
          f" '{hyperparam_type}'.\n Make sure that you ran all the experiments"
          " above and that you have added all the test losses to your"
          " experiment log.'"
      )
      continue

    if hyperparam_type == "use_early_stopping":
      # Convert boolean values to strings for categorical plotting.
      x_labels = [str(val) for val in hyperparam_values]
      plt.bar(x_labels, losses, color=COLOR_PALETTE[1], width=0.5)
    else:
      # Use a line plot for the other (numerical) hyperparameters.
      plt.plot(
          hyperparam_values,
          losses,
          marker="o",
          linestyle="-",
          color=COLOR_PALETTE[1],
          label="Test loss",
      )
      plt.legend()
      plt.grid(True)

    plt.xlabel("Hyperparameter value")
    plt.ylabel("Loss")
    plt.title(f"Loss for hyperparameter '{hyperparam_type}'")
    plt.show()
