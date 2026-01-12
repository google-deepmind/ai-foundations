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

"""A utility function for question-answer formatting."""

import pandas as pd  # For loading the dataset.


def format_qa(data: pd.Series | dict[str, str],
              sot="<start_of_turn>",
              eot="<end_of_turn>") -> tuple[str, str]:
  """Add special delimiters at start and end of question and answer.

  The function takes a row of a dataframe and returns the question and answer
  in the flashcard format, including start and end of turn delimiters, for
  Course 5 of the AI Research Foundations curriculum.

  Args:
    data: Row of a dataframe with cols "category", "question" and "answer".
    sot: String of the token for start of turn. Default: "<start_of_turn>".
    eot: String of the token for end of turn. Default: "<end_of_turn>".

  Returns:
    formatted_q: Formatted string of the question.
    formatted_a: Formatted string of the answer.
  """

  category = data["category"]
  question = data["question"]
  answer = data["answer"]

  # Add start and end tokens. Start answer like "Category: Food."
  formatted_q = f"{sot}user\n{question}{eot}\n"
  formatted_a = f"{sot}model\nCategory: {category}\n{answer}{eot}"

  return formatted_q, formatted_a
