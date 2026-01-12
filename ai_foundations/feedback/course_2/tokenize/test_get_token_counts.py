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

"""A utility function to test a learner's get_token_counts function.

This module provides a function to validate if a learner can
correclty count tokens from a list of tokens.
"""

import collections
from typing import Callable, List
from ai_foundations.feedback.utils import render_feedback

Counter = collections.Counter


def test_get_token_counts(
    get_token_counts: Callable[[List[str]], Counter[str]],
    tokens_list: List[str],
):
  """Tests if the learner correctly implements the `get_token_counts` method.

  This function checks whether the token_counts Counter is the
  same as provided by the reference implementation.

  Args:
      get_token_counts: The learner's implementation of the `get_token_counts`
        function.
      tokens_list: A list of tokens in the corpus.
  """

  def _reference_implementation(tokens_list: List[str]) -> Counter[str]:
    """Calculates the frequency of each token in a list of tokens.

    Args:
      tokens_list: A list of string tokens.

    Returns:
      A Counter where keys are the unique tokens and values are their
        corresponding frequencies.
    """

    token_counts = Counter(tokens_list)
    return token_counts

  try:

    reference_token_counts = _reference_implementation(tokens_list)
    candidate_token_counts = get_token_counts(tokens_list)

    if reference_token_counts != candidate_token_counts:
      raise ValueError(
          "Returned Counter of token counts is not correct.",
          "Your implementation returned token counts that are not correct."
          " Check your function and run this test again.",
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
