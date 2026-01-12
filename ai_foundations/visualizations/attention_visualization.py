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

"""A utility function to visualize attention weights."""

from typing import List, Optional
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def visualize_attention(
    tokens: List[str],
    attention_weights: jax.Array,
    layer: int,
    head: Optional[int] = None,
    line_color: str = "blue",
    min_line_thickness: float = 0.5,
    max_line_thickness: float = 10.0,
    show_all_weights: bool = False,
) -> None:
  """Visualizes attention weights with line thickness proportional to weights.

  This function generates a plot showing the attention from each source token
  (top row) to each target token (bottom row). The thickness of the
  connecting lines is proportional to the attention weight.

  Args:
      tokens: A list of string tokens.
      attention_weights: A JAX array of attention weights. The shape can be 4D
        (batch, heads, seq_len, seq_len), 3D (heads, seq_len, seq_len), or 2D
        (seq_len, seq_len).
      layer: The model layer number to display in the plot title.
      head: The attention head to visualize. If None, defaults to the first head
        (index 0).
      line_color: The color of the connecting lines.
      min_line_thickness: The line thickness for the minimum attention weight.
      max_line_thickness: The line thickness for the maximum attention weight.
      show_all_weights: If True, visualizes attention from every token to all
        others. If False, only shows attention from the last token.

  Raises:
      ValueError: If the attention_weights tensor has a batch size greater
        than 1, an invalid head index is provided, or the final shape of the
        weights matrix does not match the number of tokens.
  """
  n = len(tokens)
  figsize = (int(n * 0.8), 4)
  head_to_use = head if head is not None else 0

  # Squeeze out a singleton batch dimension if it exists.
  if attention_weights.ndim == 4:
    if attention_weights.shape[0] == 1:
      attention_weights = attention_weights.squeeze(axis=0)
    else:
      raise ValueError(
          "Input attention_weights has a batch size > 1. Please provide"
          " weights for a single example."
      )

  # If there is a 3D tensor, assume it's (n, num_heads, n) and select a head.
  if attention_weights.ndim == 3:
    if head_to_use >= attention_weights.shape[0]:
      raise ValueError(
          f"Invalid head index '{head_to_use}'. Tensor has only"
          f" {attention_weights.shape[0]} heads."
      )
    attention_weights = attention_weights[:, head_to_use, :].squeeze()

  # Final validation to ensure we have a 2D matrix matching token count.
  if attention_weights.shape != (n, n):
    raise ValueError(
        "The shape of attention_weights after processing must be (n, n), where"
        f" n is the number of tokens. Got {attention_weights.shape} for n={n}."
    )

  # The target tokens are the source tokens shifted by one, with a BOS token.
  tokens_target = ["<BOS>"] + tokens[:-1]

  _, ax = plt.subplots(figsize=figsize)

  # Set up coordinates for token positions.
  x_coords = jnp.linspace(0, n - 1, n)
  y_output = 1.0  # Y-coordinate for output tokens (top).
  y_input = 0.0  # Y-coordinate for input tokens (bottom).

  # Display the input and output tokens.
  for i, token_str in enumerate(tokens):
    ax.text(
        x_coords[i].item(),
        y_output,
        token_str,
        ha="center",
        va="bottom",
        fontsize=12,
    )
  for i, token_str in enumerate(tokens_target):
    ax.text(
        x_coords[i].item(),
        y_input,
        token_str,
        ha="center",
        va="top",
        fontsize=12,
    )

  max_weight = jnp.max(attention_weights).item()
  min_weight = jnp.min(attention_weights).item()

  # Avoid division by zero if all weights are identical using jnp.isclose.
  if jnp.isclose(max_weight, min_weight):
    thickness_scale = 0
    base_thickness = (min_line_thickness + max_line_thickness) / 2
  else:
    thickness_scale = (max_line_thickness - min_line_thickness) / (
        max_weight - min_weight
    )
    base_thickness = min_line_thickness

  # Determine which output tokens to draw lines from.
  start_index = 0 if show_all_weights else n - 1

  for i in range(start_index, n):
    for j in range(n):
      weight = attention_weights[i, j].item()

      if jnp.isclose(max_weight, min_weight):
        line_thickness = base_thickness
      else:
        line_thickness = (
            base_thickness + (weight - min_weight) * thickness_scale
        )

      # Use a different color for attention to the start-of-sequence token.
      color = "lightgray" if j == 0 else line_color

      line = plt.Line2D(
          [x_coords[i], x_coords[j]],
          [y_output - 0.05, y_input + 0.05],  # Small offset for text clarity.
          color=color,
          linewidth=line_thickness,
          alpha=0.7,
      )
      ax.add_line(line)

  # Add labels for the input and output rows.
  label_x_pos = -1.5  # X-coordinate to place labels left of the tokens.
  ax.text(
      label_x_pos,
      y_output,
      "Output:",
      ha="left",
      va="bottom",
      fontweight="bold",
      fontsize=12,
  )
  ax.text(
      label_x_pos,
      y_input,
      "Input:",
      ha="left",
      va="top",
      fontweight="bold",
      fontsize=12,
  )

  # Final plot adjustments.
  ax.set_xlim(-2, n)
  ax.set_ylim(-0.2, 1.2)
  ax.axis("off")
  if head is not None:
    title = f"Attention Weights for Layer {layer}, Head {head_to_use}"
  else:
    title = f"Attention Weights for Layer {layer}"
  plt.title(title)
  plt.tight_layout()
  plt.show()
