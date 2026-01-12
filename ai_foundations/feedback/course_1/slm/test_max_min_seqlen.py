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

"""A utility function to test the max and min sequence length computation.

This module provides a function to test a learner's computation of the maximum
and minimum sequence length in a dataset.
"""

from typing import List

from ai_foundations.feedback.utils import render_feedback


def test_max_min_seqlen(
    shortest_paragraph_length: int,
    longest_paragraph_length: int,
    encoded_tokens: List[List[int]],
):
  """Tests if the paragraph length variables are set correctly.

  Args:
    shortest_paragraph_length: The learner's result for computing the length of
      the shortest paragraph.
    longest_paragraph_length: The learner's result for computing the length of
      the longest paragraph.
    encoded_tokens: A list of list of token IDs where each list corresponds to
      the token IDs of one paragraph.
  """

  target_min_para_len = len(min(encoded_tokens, key=len))
  target_max_para_len = len(max(encoded_tokens, key=len))

  try:
    if shortest_paragraph_length != target_min_para_len:
      raise ValueError(
          "Your value for <code>shortest_paragraph_length</code> is not"
          " correct.",
          "Check your computation and run this cell again.",
      )

    if longest_paragraph_length != target_max_para_len:
      raise ValueError(
          "Your value for <code>longest_paragraph_length</code> is not"
          " correct.",
          "Check your computation and run this cell again.",
      )

  except ValueError as e:
    render_feedback(e)

  else:
    print("✅ Nice! Your answer looks correct.")
