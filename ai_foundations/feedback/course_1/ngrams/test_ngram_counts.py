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

"""A utility function to test a learner's n-gram generation function.

This module provides a function to validate if a learner can
correclty generate n-grams from a string of text.
"""

import collections
from typing import Callable, Dict, List, Tuple
from ai_foundations.feedback.utils import render_feedback

DefaultDict = collections.defaultdict
Counter = collections.Counter


def test_ngram_counts(
    get_ngram_counts: Callable[[List[str], int], Dict[str, Counter[str]]],
    generate_ngrams: Callable[[str, int], List[Tuple[str, ...]]],
):
  """Tests if the learner correctly implements the `get_ngram_counts` method.

  This function checks whether the n-gram counts dictionary is the
  same as provided by the reference implementation.

  Args:
      get_ngram_counts: The learner's implementation of the `get_ngram_counts`
        function.
      generate_ngrams: The `generate_ngrams` function that was defined in the
        previous coding activity.
  """

  def _reference_implementation(
      dataset: List[str], n: int
  ) -> Dict[str, Counter[str]]:
    """Computes the n-gram counts from a dataset.

    This function takes a list of text strings (paragraphs or sentences) as
    input, constructs n-grams from each text, and creates a dictionary where:

    * Keys represent n-1 token long contexts `context`.
    * Values are a Counter object `counts` such that `counts[next_token]` is the
    * count of `next_token` following `context`.

    Args:
        dataset: The list of text strings in the dataset.
        n: The size of the n-grams to generate (e.g., 2 for bigrams, 3 for
          trigrams).

    Returns:
        A dictionary where keys are (n-1)-token contexts and values are Counter
        objects storing the counts of each next token for that context.
    """

    ngram_counts = DefaultDict(Counter)

    # Loop through all paragraphs.
    for paragraph in dataset:

      # Loop through all n-grams for the paragraph.
      for ngram in generate_ngrams(paragraph, n):

        # Extract the context. This will be all but the last token.
        context = " ".join(ngram[:-1])

        # Extract the next token. This will be the last token of the n-gram.
        next_token = ngram[-1]

        # Increment the counter for the context - next_token pair by 1.
        ngram_counts[context][next_token] += 1

    return dict(ngram_counts)

  try:
    sample_data = [
        "This is a sample sentence.",
        "Another sample sentence.",
        "Split a sentence.",
        "Table Mountain is tall.",
        "Table Mountain is beautiful.",
    ]

    if _reference_implementation(sample_data, 2) != get_ngram_counts(
        sample_data, 2
    ) or _reference_implementation(sample_data, 3) != get_ngram_counts(
        sample_data, 3
    ):
      raise ValueError(
          "Returned dictionary of n-gram counts is not correct.",
          "Your implementation returned a dictionary of n-gram counts"
          " that is not correct. Check your function and run this test"
          " again.",
      )

  except (
      KeyError,
      NameError,
      ReferenceError,
      RuntimeError,
      SyntaxError,
      ValueError,
  ) as e:
    render_feedback(e)

  else:
    print("✅ Nice! Your implementation looks correct.")
