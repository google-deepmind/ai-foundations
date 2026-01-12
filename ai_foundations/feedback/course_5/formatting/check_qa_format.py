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

"""A function to check if the flashcard format was implemented correctly.

The utility function tests for the presence of the `start_of_turn` and
`end_of_turn` tokens, the category and the string "Category".
"""

from typing import Callable, Tuple
import pandas as pd


def check_qa_format(
    func: Callable[[pd.Series | dict[str, str]], Tuple[str, str]],
    dataset: pd.DataFrame,
    sot: str = "<start_of_turn>",
    eot: str = "<end_of_turn>",
) -> None:
  """Check the correctness of a `format_qa` implementation.

  The formatted question and answer that are returned by the `format_qa`
  function should contain all the input desired on the flashcards for Course 5,
  Lab 2.

  The function checks for the following:
  - The start of turn token `<start_of_turn>`.
  - The end of turn token `<end_of_turn>`.
  - The category and the string "Category".
  - The category in the answer matches the category in the question.

  Args:
    func: A function to format the question and answer.
    dataset: A pandas dataframe containing the dataset. The first row will be
      used to call the function `func`.
    sot: The start of turn token.
    eot: The end of turn token.
  """

  query = dataset.iloc[0]
  _, answer = func(query)

  if answer.find(sot) == -1:
    print(f"❌ The answer did not contain the start token {sot}.")
  elif answer.find(eot) == -1:
    print(f"❌ The answer did not contain the end token {eot}.")
  elif answer.lower().find("category") == -1:
    print("❌ The answer should start with \"Category: \". Check for typos.")
  elif answer.find(query["category"]) == -1:
    print("❌ The answer should contain the category. For example, an "
          "answer to a question about Jollof rice should contain \"Category: "
          "Food\".")
  else:
    print("✅ Nice! Your answer looks correct.")
