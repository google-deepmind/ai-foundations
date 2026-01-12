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

from typing import Callable, List, Tuple
from ai_foundations.feedback.utils import render_feedback


def test_generate_ngrams(
    generate_ngrams: Callable[[str, int], List[Tuple[str, ...]]],
    split_text: Callable[[str], List[str]],
):
  """Tests if the learner correctly implements the `generate_ngrams` method.

  This function provides three test cases of things that may go wrong:
    - The length of the generated n_grams is not correct.
    - The function returns a list of lists instead of a list of tuples.
    - The identity of the generated n_grams is not correct.

  Args:
    generate_ngrams: The function implemented by the learner.
    split_text: The whitespace tokenizer.
  """

  def _reference_implementation(text: str, n: int) -> List[Tuple[str, ...]]:
    """Reference implementation that generates n-grams from a given text.

    Args:
      text: The input text string.
      n: The size of the n-grams (e.g., 2 for bigrams, 3 for trigrams).

    Returns:
      A list of n-grams, each represented as a list of tokens.
    """

    # Tokenize text.
    tokens = split_text(text)

    # Construct the list of n-grams.
    ngrams = []

    num_of_tokens = len(tokens)
    for i in range(0, num_of_tokens - n + 1):
      ngrams.append(tuple(tokens[i : i + n]))

    return ngrams

  try:

    text1 = "This is a text!"
    text2 = "n-gram language models are fun ."

    if len(_reference_implementation(text1, 2)) != len(
        generate_ngrams(text1, 2)
    ) or len(_reference_implementation(text2, 3)) != len(
        generate_ngrams(text2, 3)
    ):
      raise ValueError(
          "Number of n-grams not correct.",
          "Your implementation returned a number of n-grams that is not"
          " correct. Check your function and run this test again.",
      )

    if not isinstance(generate_ngrams(text1, 2)[0], tuple):
      raise ValueError(
          "n-grams are not returned as tuples.",
          "Your implementation did not return a list of tuples. Make sure that"
          " your function returns a list of tuples and run this test"
          " again.",
      )

    if (
        _reference_implementation(text1, 2) != generate_ngrams(text1, 2)
        or _reference_implementation(text2, 2) != generate_ngrams(text2, 2)
        or _reference_implementation(text1, 3) != generate_ngrams(text1, 3)
        or _reference_implementation(text2, 3) != generate_ngrams(text2, 3)
    ):
      raise ValueError(
          "Returned list of n-grams not correct.",
          "Your implementation returned a list of n-grams that is not"
          " correct. Check your function and run this test again.",
      )

  except (
      RuntimeError,
      KeyError,
      SyntaxError,
      ReferenceError,
      NameError,
      ValueError,
  ) as e:
    render_feedback(e)

  else:
    print("✅ Nice! Your implementation looks correct.")
