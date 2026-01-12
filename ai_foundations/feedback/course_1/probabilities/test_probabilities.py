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

"""A utility function to test a learner's probability distribution.

This module provides a function to check if a list of numbers forms a valid
probability distribution (i.e., values are between 0 and 1, they sum to 1, there
is one probability per word).
"""

from typing import List

from ai_foundations.feedback.utils import render_feedback
import numpy as np


def test_probabilities(candidate_words: List[str], learner_probs: List[float]):
  """Tests if `learner_probs` is a proper probability distribution.

  Conditions checked (in order):
  1. Lengths match
  2. No negative numbers
  3. All values ≤ 1
  4. Sum = 1

  Args:
    candidate_words: List of words for which probabilities are provided.
    learner_probs: List of probabilities corresponding to the candidate words.
  """

  try:
    # 1. Check for length mismatch.
    if len(candidate_words) != len(learner_probs):
      raise ValueError(
          "Mismatch between words and probabilities.",
          f"You supplied {len(learner_probs)} probabilities for"
          f" {len(candidate_words)} words."
          " Make sure there is exactly one probability per word.",
      )

    # 2. Check for negative probabilities.
    if any(p < 0 for p in learner_probs):
      offending = [p for p in learner_probs if p < 0]
      offending_str = ", ".join(str(p) for p in offending)
      raise ValueError(
          "Probabilities cannot be negative.",
          f"The following value(s) are < 0: {offending_str}. Replace the"
          " negative value(s) with numbers in the range [0, 1].",
      )

    # 3. Check for probabilities greater than 1.
    if not all(p <= 1 for p in learner_probs):
      offending = [p for p in learner_probs if p > 1]
      offending_str = ", ".join(str(p) for p in offending)
      raise ValueError(
          "Probabilities must not exceed 1.",
          f"The following value(s) are > 1: {offending_str}."
          " Replace the value(s) > 1 with numbers in the range [0, 1].",
      )

    # 4. Check if probabilities sum to 1.
    current_sum = np.sum(learner_probs)
    if not np.isclose(current_sum, 1.0):
      raise ValueError(
          "Probabilities must sum to 1.",
          f"The sum of your probabilities is: {current_sum:.4f}."
          " Please adjust the numbers and try again.",
      )

  except ValueError as e:
    render_feedback(e)

  else:
    print("✅ You've set the probabilities successfully. Well done!")
