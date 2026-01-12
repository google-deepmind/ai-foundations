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

"""A utility function to test a learner's n-gram model solution.

This module provides a function to validate if a user can
correctly identify candidate words for a given context in a trigram model.
"""

from typing import Dict, Iterable
from ai_foundations.feedback.utils import render_feedback


def test_candidate_tokens(
    trigram_model: Dict[str, Dict[str, float]],
    candidate_tokens: Iterable[str],
    candidate_tokens_probabilities: Iterable[float],
):
  """Tests if the learner correctly identifies candidate tokens from a model.

  This function checks if the provided `candidate_tokens` and
  `candidate_tokens_probabilities` are set correctly. It checks if the learner
  correctly identifies candidate tokens from a model.

  Args:
    trigram_model: A dictionary representing the ngram model, mapping contexts
      to a dictionary of candidate words and their probabilities.
    candidate_tokens: A list of strings representing the learner's identified
      candidate tokens.
    candidate_tokens_probabilities: A list of corresponding token probabilities.
  """

  hint = """
    The ngram model is a dictionary where each context maps to another
    dictionary of candidate tokens.<br>
    To get a list of candidate tokens for a specific context, put the context in
    the model as <code>ngram_model[context]</code> and then use the
    <code>.keys()</code> method.<br>
    For extracting the probabilities, you can use the <code>.values()</code>
    method.
    """

  context = "looking for"

  try:
    if context not in trigram_model.keys():
      raise KeyError(
          "Sorry, your answer is not correct.",
          f"The context '{context}' does not exist in the trigram model.",
      )

    candidate_tokens = [token for token in candidate_tokens]
    target_tokens = [token for token in trigram_model[context].keys()]

    if candidate_tokens != target_tokens:
      raise ValueError(
          "Sorry, your answer is not correct.",
          "Your list of tokens does not match the expected candidates.",
      )

    candidate_tokens_probabilities = list(candidate_tokens_probabilities)
    target_tokens_probabilities = list(trigram_model[context].values())

    if candidate_tokens_probabilities != target_tokens_probabilities:
      raise ValueError(
          "Your answer is not correct.",
          "Your list of probabilities does not match the expected"
          " probabilities.",
      )

  except (KeyError, NameError, RuntimeError, SyntaxError, ValueError) as e:
    render_feedback(e, hint)

  else:
    print("✅ Nice! Your answer looks correct.")
