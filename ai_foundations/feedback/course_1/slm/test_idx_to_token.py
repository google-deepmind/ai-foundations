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

"""A utility function to test the implementation of the index-to-token mapping.

This module provides a function to test a learner's dictionary
that maps integer indices to vocabulary tokens, ensuring it is correct.
"""

from typing import Dict, List
from ai_foundations.feedback.utils import render_feedback


def test_index_to_token(index_to_token: Dict[int, str], vocabulary: List[str]):
  """Tests if the index_to_token dictionary is implemented correctly.

  This function compares the learner's dictionary against a correctly
  implemented solution to ensure it accurately maps a zero-based index to each
  token from the provided vocabulary.

  Args:
    index_to_token: The learner's dictionary mapping an integer index to a
      token.
    vocabulary: The list of vocabulary tokens to use for the mapping.
  """

  correct_solution = {idx: token for idx, token in enumerate(vocabulary)}

  hint = """
    1. Create a dictionary where the key is <code>index</code> and the value
       is <code>token</code>.<br>
    2. You can use the Python <code>enumerate</code> function to loop through
       the vocabulary and get both the index and the token at the same time.
    """

  try:
    if not isinstance(index_to_token, dict):
      raise ValueError(
          "Sorry, your answer is not correct.",
          "Make sure that you return a dictionary.",
      )

    if index_to_token != correct_solution:
      raise ValueError(
          "Sorry, your answer is not correct.",
          "Your index-to-token mapping does not match the expected solution.",
      )

  except ValueError as e:
    render_feedback(e, hint)

  else:
    print("✅ Nice! Your answer looks correct.")
