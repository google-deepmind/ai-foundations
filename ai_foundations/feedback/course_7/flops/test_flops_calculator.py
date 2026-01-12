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

"""A utility function to test a learner's FLOPs calculation.

This module provides a function to validate if a user can correctly implement
the back-of-the-envelope formula for estimating training FLOPs.
"""

from typing import Callable

from ai_foundations.feedback.utils import render_feedback
import numpy as np


def test_compute_num_flops(
    compute_num_flops_func: Callable[[float, float], float],
) -> None:
  """Tests if the learner correctly implements the FLOPs estimation formula.

  This function checks if the provided `compute_num_flops_func` correctly
  calculates the training FLOPs based on the formula 6 * P * N across
  several test cases.

  Args:
    compute_num_flops_func: The learner's implementation of the
      `compute_num_flops` function.
  """

  hint = """
      Remember, the estimation formula is <code>FLOPs ≈ 6 * P * N</code>.
      <br />
      In Python, this translates to <code>6 * param_count * num_tokens</code>.
      Make sure your function returns this value.
      """

  test_cases = {
      "Small Model": {"params": 1e9, "tokens": 20e9},
      "Large Model": {"params": 70e9, "tokens": 1.5e12},
      "Simple Case": {"params": 100, "tokens": 1000},
  }

  try:
    for name, values in test_cases.items():
      params = values["params"]
      tokens = values["tokens"]

      student_result = compute_num_flops_func(params, tokens)

      reference_result = 6 * params * tokens

      if not np.isclose(student_result, reference_result):
        raise ValueError(
            "Your calculation is incorrect.",
            f"For the '{name}' test case, the expected result was "
            f"approximately {reference_result:.2e}, but your function "
            f"returned {student_result:.2e}.",
        )

  except (
      KeyError,
      NameError,
      ReferenceError,
      RuntimeError,
      SyntaxError,
      ValueError,
  ) as e:
    render_feedback(e, hint)

  else:
    print("✅ Nice! Your answer looks correct.")
