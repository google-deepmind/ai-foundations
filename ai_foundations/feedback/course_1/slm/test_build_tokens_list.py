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

"""A utility function to test the implementation of building a list of tokens.

This module provides a function to test a learner's implementation
of constructing a list of all tokens in a dataset.
"""

from typing import Callable, List
from ai_foundations.feedback.utils import render_feedback


def test_build_tokens_list(
    tokens: List[str],
    tokenizer: Callable[[str], List[str]],
    dataset: List[str],
):
  """Tests the learner's implementation of a `build_vocabulary` function.

  This function validates that the provided list of tokens contains all
  tokens in the correct order obtained by splitting the each entry in dataset
  using the split text function.

  Can display a hint for learners if they get stuck.

  Args:
      tokens: The learner's list of tokens extracted from dataset.
      tokenizer: A tokenizer that splits a string into individual tokens.
      dataset: A list of texts comprising the entire dataset.
  """

  hint = """
    1. Use a for loop to go through every paragraph in the dataset.<br>
    2. Split each paragraph into individual tokens using the
       <code>split_text</code> function.
       """

  try:
    target_tokens = []
    for paragraph in dataset:
      target_tokens.extend(tokenizer(paragraph))

    if target_tokens != tokens:
      raise ValueError(
          "Sorry, your answer is not correct.",
          "`tokens` does not include all tokens from the dataset in the"
          " correct order.",
      )

  except ValueError as e:
    render_feedback(e, hint)

  else:
    print("✅ Nice! Your answer looks correct.")
