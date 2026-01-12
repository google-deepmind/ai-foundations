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

"""Plotting utilities for embedding vectors.

This module provides a helper to visualize selected dimensions of word
embeddings.
"""

from typing import List
import matplotlib.pyplot as plt
import numpy as np


def plot_embeddings_dimensions(
    embeddings: np.ndarray, words: List[str], dim_x: int, dim_y: int
):
  """Scatter plot of two embedding dimensions for the given words.

  Args:
    embeddings: Array of shape [num_words, embedding_dim]. Embedding matrix.
    words: List of length num_words. Labels for each point.
    dim_x: Index of the embedding dimension to use for the x-axis.
    dim_y: Index of the embedding dimension to use for the y-axis.
  """
  x = embeddings[:, dim_x]
  y = embeddings[:, dim_y]
  plt.figure()
  # Plot the 2D embeddings using a scatter plot
  plt.scatter(x, y)
  # Annotate the points with the corresponding word labels
  for i, word in enumerate(words):
    plt.text(x[i], y[i], word)
  # Set plot title and labels
  plt.xlabel(f"Dimension {dim_x}")
  plt.ylabel(f"Dimension {dim_y}")
  plt.title(f"Embedding dimensions {dim_x} versus {dim_y}")
  plt.grid(True)
  plt.show()
