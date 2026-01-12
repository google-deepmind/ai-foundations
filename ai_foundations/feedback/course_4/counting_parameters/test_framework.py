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

"""Test framework for different test functions."""

import time
from typing import Any, Dict, List
from IPython.display import display
from IPython.display import HTML


def test_framework(
    function_name: str,
    reference_implementation_str: str,
    learner_solutions: List[Any],
    correct_solutions: List[int],
    near_misses: List[Dict[Any, str]],
) -> None:
  """Test framework for different test functions.

  Args:
    function_name: The name of the function to test.
    reference_implementation_str: The reference implementation of the function.
    learner_solutions: The solutions provided by the learner.
    correct_solutions: The correct solutions.
    near_misses: The near misses.

  Returns:
    None
  """

  assert len(learner_solutions) == len(correct_solutions)
  assert len(learner_solutions) == len(near_misses)

  correct = True
  for learner_sol, correct_sol, near_miss_d in zip(
      learner_solutions, correct_solutions, near_misses
  ):

    near_miss_d[...] = (
        'You did not fully implement the function. It returns "..."'
    )
    near_miss_d[correct_solutions[0]] = (
        "You seem to have hard-coded the"
        " the number of parameters in your implementation. Make sure to use"
        " the values in `hyperparams` to compute the number of parameters."
    )

    if learner_sol != correct_sol:
      if learner_sol in near_miss_d:
        error_message = near_miss_d[learner_sol]
      elif isinstance(learner_sol, str):
        if "is not defined" in learner_sol:
          learner_sol = (
              learner_sol
              + ". Make sure you have executed the"
              " previous cell before running the tests."
          )
        error_message = learner_sol
      else:
        error_message = (
            "Your solution did not return the correct number of parameters."
        )
      correct = False
      break

  if not correct:
    # Remove function defintion and leading indentation.
    base_indentation = len(reference_implementation_str) - len(
        reference_implementation_str.lstrip(" ")
    )
    reference_implementation_str = "\n".join([
        line[base_indentation:]
        for line in reference_implementation_str.split("\n")[1:]
    ])
    display(
        HTML(
            "<h3>Your implementation is incorrect!</h3>"
            f"<p><strong>Details:</strong><br>{error_message}</p>"
        )
    )
    print()
    time.sleep(0.5)
    give_hints = input("Would you like see the solution? Type Yes or No.\n")
    if give_hints.lower() in ["yes", "y"]:
      solution = f"""
        <h4>Solution</h4>

        <p>Add these lines to the function body of
        <code>{function_name}</code>:</p>
        <br>

        <pre>
{reference_implementation_str}
        </pre>
        """
      display(HTML(solution))
    else:
      display(
          HTML(
              "<p>Great, keep trying! Modify the code according to the"
              " error message above and run this cell again.</p>"
          )
      )
  else:
    print("✅ All tests passed. Your implementation is looking good.")
