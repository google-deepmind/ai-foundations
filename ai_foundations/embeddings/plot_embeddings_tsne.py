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

"""Utilities to visualize word embedding vectors.

Projects high-dimensional embeddings to 2D using t-SNE and displays a labeled
scatter plot. Intended for quick qualitative inspection of token embeddings.
"""

from typing import List, Optional

import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold


TSNE = manifold.TSNE
COLOR_PALETTE = ["#FFC20A", "#0C7BDC", "#71A816", "#B70F35"]


def plot_embeddings_tsne(
    embeddings: np.ndarray,
    labels: List[str],
    colors: Optional[List[int]] = None,
):
  """Visualizes high-dimensional embeddings in 2D using t-SNE.

  This function takes a set of high-dimensional vectors (embeddings),
  reduces their dimensionality to two using the t-SNE algorithm, and then
  generates a scatter plot. Each point in the plot represents an embedding,
  annotated with its corresponding text label.

  Args:
      embeddings: A numpy array of shape `(n_tokens, embedding_dim)` containing
        the high-dimensional embedding vectors to visualize.
      labels: A list of string labels, where `labels[i]` corresponds to the
        embedding `embeddings[i]`. The length of this list must be equal to
        `n_samples`.
      colors: An optional list of numerical values used to color the points in
        the scatter plot, often representing clusters or categories. If `None`,
        all points will be plotted in a single default color.
  """

  tsne = TSNE(n_components=2, random_state=42, perplexity=5)
  embeddings_2d_tsne = tsne.fit_transform(embeddings)

  cmap = pltcolors.ListedColormap(COLOR_PALETTE)

  plt.figure(figsize=(8, 6))
  plt.scatter(
      embeddings_2d_tsne[:, 0],
      embeddings_2d_tsne[:, 1],
      c=COLOR_PALETTE[1] if colors is None else colors,
      s=100,
      cmap=cmap if colors is not None else None,
  )

  for i, label in enumerate(labels):
    plt.annotate(
        label,
        (embeddings_2d_tsne[i, 0], embeddings_2d_tsne[i, 1]),
        textcoords="offset points",
        xytext=(0, 5),
        ha="center",
        fontsize=12,
    )

  plt.title("Token embeddings visualization (t-SNE)", fontsize=14)
  plt.xlabel("t-SNE dimension 1", fontsize=12)
  plt.ylabel("t-SNE dimension 2", fontsize=12)
  plt.grid(True)
  plt.show()
