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

"""A collection of test functions for the GPU Memory Calculator lab.

This module provides a suite of validation and feedback functions, one for each
activity in the lab, to check a learner's implementation of the memory
calculation formulas.
"""

from typing import Callable

from ai_foundations.feedback.utils import render_feedback
from ai_foundations.utils.formatting import bytes_to_gb
import numpy as np


def test_calculate_param_memory(
    param_memory_func: Callable[[float, int], float],
) -> None:
  """Tests if the learner correctly calculates parameter memory."""

  try:
    param_count = 1e9
    bytes_per_param = 4

    # Call the learner's function
    student_result = param_memory_func(param_count, bytes_per_param)

    # Calculate the correct answer
    reference_result_gb = bytes_to_gb(param_count * bytes_per_param)

    # Check if the calculation is correct
    if not np.isclose(student_result, reference_result_gb):
      raise ValueError(
          "Your calculation is incorrect.",
          "Your function returned an incorrect value for a set of test"
          " parameters. Please check your formula.",
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
    print("✅ Nice! Your answer looks correct.")


def test_calculate_input_data_memory(
    input_data_memory_func: Callable[[int, int, int], float],
) -> None:
  """Tests if the learner correctly calculates input data memory."""

  try:
    # Use the standard constants for the test case
    batch_size = 8
    seq_len = 1024
    bytes_per_id = 4

    # Call the learner's function
    student_result = input_data_memory_func(batch_size, seq_len, bytes_per_id)

    # Calculate the correct answer
    reference_result_gb = bytes_to_gb(batch_size * seq_len * bytes_per_id)

    # Check if the calculation is correct
    if not np.isclose(student_result, reference_result_gb):
      raise ValueError(
          "Your calculation is incorrect.",
          "Your function returned an incorrect value for a set of test"
          " parameters. Please check your formula.",
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
    print("✅ Nice! Your answer looks correct.")


def test_calculate_gradient_memory(
    gradient_memory_func: Callable[[float, int], float],
) -> None:
  """Tests if the learner correctly calculates gradient memory.

  Args:
    gradient_memory_func: The learner's implementation of the function to test.
  """

  try:
    param_count = 1e9
    bytes_per_param = 4

    # Call the learner's function
    student_result = gradient_memory_func(param_count, bytes_per_param)

    # Calculate the correct answer
    reference_result_gb = bytes_to_gb(param_count * bytes_per_param)

    # Check if the calculation is correct
    if not np.isclose(student_result, reference_result_gb):
      raise ValueError(
          "Your calculation is incorrect.",
          "Your function returned an incorrect value for a set of test"
          " parameters. Please check your formula.",
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
    print("✅ Nice! Your answer looks correct.")


def test_calculate_optimizer_memory(
    optimizer_memory_func: Callable[[float, int], float],
) -> None:
  """Tests if the learner correctly calculates Adam optimizer state memory.

  Args:
    optimizer_memory_func: The learner's implementation of the function to test.
  """

  try:
    param_count = 1e9
    bytes_per_param = 4

    # Call the learner's function
    student_result = optimizer_memory_func(param_count, bytes_per_param)

    # Calculate the correct answer (note the '2 *' for Adam)
    reference_result_gb = bytes_to_gb(2 * param_count * bytes_per_param)

    # Check if the calculation is correct
    if not np.isclose(student_result, reference_result_gb):
      raise ValueError(
          "Your calculation is incorrect.",
          "Your function returned an incorrect value for a set of test"
          " parameters. Please check your formula.",
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
    print("✅ Nice! Your answer looks correct.")


def test_calculate_activation_memory(
    activation_memory_func: Callable[[int, int, int, int, int], float],
) -> None:
  """Tests if the learner correctly calculates the estimated activation memory.

  Args:
    activation_memory_func: The learner's implementation of the function to
      test.
  """

  try:
    # Use the standard constants for the test case
    batch_size = 8
    seq_len = 1024
    num_layers = 32
    hidden_size = 4096
    bytes_per_param = 4

    # Call the learner's function
    student_result = activation_memory_func(
        batch_size, seq_len, num_layers, hidden_size, bytes_per_param
    )

    # Calculate the correct answer
    reference_result_gb = bytes_to_gb(
        batch_size * seq_len * num_layers * hidden_size * bytes_per_param
    )

    # Check if the calculation is correct
    if not np.isclose(student_result, reference_result_gb):
      raise ValueError(
          "Your calculation is incorrect.",
          "Your function returned an incorrect value for a set of test"
          " parameters. Please check your formula.",
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
    print("✅ Nice! Your answer looks correct.")
