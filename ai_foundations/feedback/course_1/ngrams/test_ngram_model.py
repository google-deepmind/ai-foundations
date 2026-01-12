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

"""A utility function to test a learner's n-gram model estimation.

This module provides a function to validate if a learner can correctly generate
estimate n-gram probabilities from a text corpus.
"""

import collections
from typing import Callable, Dict, List

from ai_foundations.feedback.utils import render_feedback

Counter = collections.Counter


def test_build_ngram_model(
    build_ngram_model: Callable[[List[str], int], Dict[str, Dict[str, float]]],
    get_ngram_counts: Callable[[List[str], int], Dict[str, Counter[str]]],
):
  """Tests if the learner correctly implements the `build_ngram_model` method.

  This function checks whether the n-gram probabilities dictionary is the
  same as provided by the reference implementation.

  Args:
    build_ngram_model: The learner's implementation of the `build_ngram_model`
      function.
    get_ngram_counts: The `get_ngram_counts` function that was defined in the
      previous coding activity.
  """

  def _reference_implementation(
      dataset: List[str], n: int
  ) -> Dict[str, Dict[str, float]]:
    """Builds an n-gram language model.

    This function takes a list of text strings (paragraphs or sentences) as
    input, generates n-grams from each text using the function get_ngram_counts
    and converts them into probabilities. The resulting model is a dictionary,
    where keys are (n-1)-token contexts and values are dictionaries mapping
    possible next tokens to their conditional probabilities given the context.

    Args:
      dataset: A list of text strings representing the dataset.
      n: The size of the n-grams (e.g., 2 for a bigram model).

    Returns:
      A dictionary representing the n-gram language model, where keys are
          (n-1)-tokens contexts and values are dictionaries mapping possible
          next tokens to their conditional probabilities.
    """

    ngram_model = {}

    ngram_counts = get_ngram_counts(dataset, n)

    # Loop through the possible contexts. `context` is a string and
    # `next_tokens` is a dictionary mapping possible next tokens to their counts
    # of following `context`.
    for context, next_tokens in ngram_counts.items():

      # Compute Count(A) and P(B | A) here.
      context_total_count = sum(next_tokens.values())
      ngram_model[context] = {}
      for token, count in next_tokens.items():
        ngram_model[context][token] = count / context_total_count

    return ngram_model

  test_dataset1 = [
      "Table Mountain is tall.",
      "Table Mountain is beautiful.",
      "I like to climb Table Mountain in the winter.",
  ]
  test_dataset2 = ["a b c", "a b c d", "a c d e", "a b c d e", "b c d e"]
  try:
    candidate_trigram_model_1 = build_ngram_model(test_dataset1, 3)
    target_trigram_model_1 = _reference_implementation(test_dataset1, 3)

    candidate_trigram_model_2 = build_ngram_model(test_dataset2, 3)
    target_trigram_model_2 = _reference_implementation(test_dataset2, 3)

    candidate_bigram_model_1 = build_ngram_model(test_dataset1, 2)
    target_bigram_model_1 = _reference_implementation(test_dataset1, 2)

    candidate_bigram_model_2 = build_ngram_model(test_dataset2, 2)
    target_bigram_model_2 = build_ngram_model(test_dataset2, 2)

    if (
        candidate_trigram_model_1 != target_trigram_model_1
        or candidate_trigram_model_2 != target_trigram_model_2
        or candidate_bigram_model_1 != target_bigram_model_1
        or candidate_bigram_model_2 != target_bigram_model_2
    ):
      raise RuntimeError(
          "Your implementation is not correct.",
          "Your implementation of <code>build_ngram_model</code> did not"
          " return the correct dictionary of dictionaries that map contexts to"
          " next words to their condititonal probabilities P(next word |"
          " context).<br>Please check your solution and run this test again.",
      )

  except (
      NameError,
      KeyError,
      ReferenceError,
      RuntimeError,
      SyntaxError,
      ValueError,
  ) as e:
    render_feedback(e)

  else:
    print("✅ Nice! Your implementation looks correct.")
