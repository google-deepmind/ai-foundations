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

"""Utility functions to test learner's exercises with embeddings.

This module provides functions to validate if a learner can
correctly work with numpy arrays and embeddings.
"""

from typing import Callable

from ai_foundations.feedback.utils import render_feedback
import numpy as np


def test_numpy_arrays(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    P: np.ndarray,  # pylint: disable=invalid-name
    Q: np.ndarray,  # pylint: disable=invalid-name
    R: np.ndarray,  # pylint: disable=invalid-name
):
  """Tests if the learner correctly implements six numpy arrays.

  This function checks whether the learner's arrays (a, b, c, P, Q, R)
  are the same as the ones provided by the reference implementation by
  checking for correct shape and element-wise equality.

  Args:
      a: The learner's implementation of numpy array 'a'.
      b: The learner's implementation of numpy array 'b'.
      c: The learner's implementation of numpy array 'c'.
      P: The learner's implementation of numpy array 'P'.
      Q: The learner's implementation of numpy array 'Q'.
      R: The learner's implementation of numpy array 'R'.
  """

  # Reference solution arrays.
  ref_a = np.array([7, 3, 1, 4])
  ref_b = np.array([1.5, -2.5])
  ref_c = np.array([4, 4, 4])
  ref_P = np.array([[7, 4], [3, 5], [1, 6], [4, 7]])  # pylint: disable=invalid-name
  ref_Q = np.array([[7, 3, 1, 4], [4, 5, 6, 7]])  # pylint: disable=invalid-name
  ref_R = np.array([[4, 4, 4]])  # pylint: disable=invalid-name

  try:
    # Package candidate and reference arrays with their names for iteration.
    arrays_to_check = [
        ("a", a, ref_a),
        ("b", b, ref_b),
        ("c", c, ref_c),
        ("P", P, ref_P),
        ("Q", Q, ref_Q),
        ("R", R, ref_R),
    ]

    # Iterate and check each array.
    for name, candidate_arr, reference_arr in arrays_to_check:
      if not np.array_equal(candidate_arr, reference_arr):
        raise ValueError(
            f"The array '{name}' is not correct.",
            f"Your implementation of array '{name}' did not match the expected "
            "output. Please check its shape and values and run the test again.",
        )

  except (
      AttributeError,
      NameError,
      ReferenceError,
      RuntimeError,
      SyntaxError,
      TypeError,
      ValueError,
  ) as e:
    render_feedback(e)

  else:
    print("✅ Nice! Your answer looks correct.")


def test_embedding_dimension(embedding_dim: int, embeddings: np.ndarray):
  """Tests if the learner correctly extracts the embedding dimension.

  This function checks whether the learner correctly extracted the second
  dimension (number of columns) from the `embeddings` numpy array.

  Args:
    embedding_dim: The learner's extracted embedding dimension.
    embeddings: The numpy array from which the dimension is extracted.
  """
  try:
    # The reference solution is the second element of the shape tuple.
    reference_embedding_dim = embeddings.shape[1]

    if embedding_dim != reference_embedding_dim:
      raise ValueError(
          "The `embedding_dim` is not correct.",
          f"Your implementation returned {embedding_dim}, but the expected "
          f"embedding dimension is {reference_embedding_dim}. "
          "Hint: Use the `.shape` attribute of the numpy array and index the "
          "correct dimension.",
      )

  except (
      AttributeError,
      NameError,
      ReferenceError,
      RuntimeError,
      SyntaxError,
      TypeError,
      ValueError,
  ) as e:
    render_feedback(e)

  else:
    print("✅ Excellent! You've correctly extracted the embedding dimension.")


def test_numpy_slicing(
    third_row: np.ndarray, seventh_column: np.ndarray, embeddings: np.ndarray
):
  """Tests if the learner correctly slices a numpy array.

  This function checks whether the learner correctly extracted the third row
  (index 2) and the seventh column (index 6) from the `embeddings` array.

  Args:
    third_row: The learner's extracted third row.
    seventh_column: The learner's extracted seventh column.
    embeddings: The numpy array from which to slice.
  """
  try:
    # Define the reference solutions for slicing.
    reference_third_row = embeddings[2, :]
    reference_seventh_column = embeddings[:, 6]

    # Check the third row.
    if not np.array_equal(third_row, reference_third_row):
      raise ValueError(
          "The `third_row` is not correct.",
          "Your implementation for `third_row` did not match the expected "
          "output. Remember that indexing starts at 0 and you need to select "
          "all columns for that row.",
      )

    # Check the seventh column.
    if not np.array_equal(seventh_column, reference_seventh_column):
      raise ValueError(
          "The `seventh_column` is not correct.",
          "Your implementation for `seventh_column` did not match the expected "
          "output. Remember that indexing starts at 0 and you need to select "
          "all rows for that column.",
      )

  except (
      AttributeError,
      IndexError,
      NameError,
      ReferenceError,
      RuntimeError,
      SyntaxError,
      TypeError,
      ValueError,
  ) as e:
    render_feedback(e)

  else:
    print("✅ Nice! Your answer looks correct.")


def test_dot_product(dot_product: float, embeddings: np.ndarray):
  """Tests if the learner correctly computes the dot product of two vectors.

  This function checks whether the learner correctly computed the dot product
  between the third row (index 2) and the fourth row (index 3) of the
  `embeddings` array.

  Args:
    dot_product: The learner's computed dot product.
    embeddings: The numpy array from which to extract the vectors.
  """
  try:
    # Define the reference solution.
    reference_third_row = embeddings[2, :]
    reference_fourth_row = embeddings[3, :]
    reference_dot_product = np.dot(reference_third_row, reference_fourth_row)

    # Check if the computed dot product is correct using np.isclose for
    # floating-point comparisons.
    if not np.isclose(dot_product, reference_dot_product):
      raise ValueError(
          "The `dot_product` is not correct.",
          "Your computed dot product did not match the expected value. "
          "Ensure you have extracted the correct rows (index 2 and 3) and "
          "are using the correct numpy function to compute the dot product.",
      )

  except (
      AttributeError,
      IndexError,
      NameError,
      ReferenceError,
      RuntimeError,
      SyntaxError,
      TypeError,
      ValueError,
  ) as e:
    render_feedback(e)

  else:
    print("✅ Nice! Your answer looks correct.")


def test_get_embedding(
    get_embedding: Callable[[str, np.ndarray, list[str]], np.ndarray],
    embeddings: np.ndarray,
    labels: list[str],
):
  """Tests if the learner correctly implements the `get_embedding` function.

  This function checks if the learner's implementation can correctly retrieve a
  token's embedding from an embedding matrix given a list of labels.

  Args:
    get_embedding: The learner's implementation of the `get_embedding` function.
    embeddings: The embedding matrix.
    labels: The list of tokens corresponding to the rows in `embeddings`.
  """

  def _reference_get_embedding(
      token: str, embeddings: np.ndarray, labels: list[str]
  ) -> np.ndarray:
    """Reference implementation for retrieving a token embedding."""
    if token not in labels:
      raise ValueError(f"No embeddings for {token} exist.")
    token_idx = labels.index(token)
    embedding = embeddings[token_idx, :]
    return embedding

  try:
    test_token = labels[len(labels) // 2]  # Pick a token from the middle.
    candidate_embedding = get_embedding(test_token, embeddings, labels)
    reference_embedding = _reference_get_embedding(
        test_token, embeddings, labels
    )
    if not np.array_equal(candidate_embedding, reference_embedding):
      raise ValueError(
          f"Embedding for token '{test_token}' is incorrect.",
          "Your function returned an incorrect embedding for a valid token. "
          "Check your logic for finding the token index and slicing the "
          "embeddings matrix.",
      )

  except (
      AttributeError,
      IndexError,
      NameError,
      ReferenceError,
      RuntimeError,
      SyntaxError,
      TypeError,
      ValueError,
  ) as e:
    render_feedback(e)

  else:
    print("✅ Nice! Your answer looks correct.")


def test_print_similarity(
    print_similarity: Callable[[str, str, np.ndarray, list[str]], float],
    get_embedding: Callable[[str, np.ndarray, list[str]], np.ndarray],
    cos_sim: Callable[[np.ndarray, np.ndarray], float],
    embeddings: np.ndarray,
    labels: list[str],
):
  """Tests if the learner correctly implements the `print_similarity` function.

  This function checks if the learner's implementation can correctly compute
  and return the cosine similarity between two tokens.

  Args:
      print_similarity: The learner's implementation of the function.
      get_embedding: The learner's implementation of `get_embedding`.
      cos_sim: A function for computing the cosine similarity of two vectors.
      embeddings: The embedding matrix.
      labels: The list of tokens corresponding to the rows in `embeddings`.
  """
  try:
    # Extract two valid tokens.
    token1 = labels[1]
    token2 = labels[len(labels) - 2]

    # Get the learner's computed similarity.
    candidate_similarity = print_similarity(token1, token2, embeddings, labels)

    # Compute the reference similarity.
    embedding1 = get_embedding(token1, embeddings, labels)
    embedding2 = get_embedding(token2, embeddings, labels)
    reference_similarity = cos_sim(embedding1, embedding2)

    if not np.isclose(candidate_similarity, reference_similarity):
      raise ValueError(
          "The returned similarity value is incorrect.",
          "Your function returned an incorrect cosine similarity value. "
          "Ensure you are calling `get_embedding` for both tokens and then "
          "passing those embeddings to the `cos_sim` function.",
      )

  except (
      AttributeError,
      IndexError,
      NameError,
      ReferenceError,
      RuntimeError,
      SyntaxError,
      TypeError,
      ValueError,
  ) as e:
    render_feedback(e)

  else:
    print("✅ Nice! Your answer looks correct.")
