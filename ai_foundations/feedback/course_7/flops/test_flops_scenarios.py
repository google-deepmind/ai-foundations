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

"""Provides a test function to validate the FLOPs scenarios activity.

This module contains the test_final_results function, which checks if a learner
has correctly calculated and entered the FLOPs values for several predefined
model and dataset scenarios.
"""

from typing import Any, Dict, List

from ai_foundations.feedback.utils import render_feedback
import numpy as np


def test_final_results(learner_data: List[Dict[str, Any]]) -> bool:
  """Tests if the learner correctly calculated and entered the FLOPs.

  This function iterates through a list of dictionaries provided by the
  learner, validates the scenario names, and checks if the entered 'FLOPs'
  value is numerically close to the correct calculation.

  Args:
    learner_data: A list of dictionaries, where each dictionary represents a
      scenario. Each dictionary should contain at least a 'Scenario' key with a
      string name and a 'FLOPs' key with the calculated floating-point value.

  Returns:
    True if all scenarios in the provided data are correct, otherwise it renders
      feedback and returns False.
  """

  hint = """
      Double-check your calculated FLOPs values. Make sure you have entered the
      raw number (not the formatted string) into the 'FLOPs' field for each
      dictionary.
      <br/>
      For example: For T5-Large, the value should be close to <code>4.62e+21</code>.
      """

  reference_flops = {
      "BERT-Base": 6 * 110e6 * 3.3e9,
      "T5-Large": 6 * 770e6 * 1e12,
      "Gemma-1B (Africa Galore)": 6 * 1e9 * 30e3,
      "PaLM": 6 * 540e9 * 780e9,
  }

  try:
    if len(learner_data) != len(reference_flops):
      raise ValueError(
          "The number of scenarios in your list is incorrect.",
          f"Expected {len(reference_flops)} but found {len(learner_data)}.",
      )

    for item in learner_data:
      name = item.get("Scenario")
      student_flops = item.get("FLOPs")

      if name not in reference_flops:
        raise ValueError(
            f"Scenario name '{name}' is not one of the expected names."
        )

      reference_value = reference_flops.get(name)

      if not np.isclose(student_flops, reference_value, rtol=1e-2):
        raise ValueError(
            f"The 'FLOPs' value for '{name}' is incorrect.",
            f"Expected approximately {reference_value:.2e}, but got"
            f" {student_flops:.2e}.",
        )

    print("✅ Nice! Your answer looks correct.")
    return True

  except (
      KeyError,
      NameError,
      ReferenceError,
      RuntimeError,
      SyntaxError,
      ValueError,
  ) as e:
    render_feedback(e, hint)
    return False
