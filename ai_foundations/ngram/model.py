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

"""A class to build an n-gram model.

This module provides a class `NGramModel` that estimates an unsmoothed n-gram
model from a corpus of strings.
"""

import collections
import random
from typing import Callable

DefaultDict = collections.defaultdict
Counter = collections.Counter


class NGramModel:
  """A class for an n-gram language model.

  This model learns probabilities of token sequences from a given dataset and
  can generate new text based on those learned probabilities.

  Attributes:
    n: The order of the n-gram model (e.g., 2 for bigram, 3 for trigram).
    probabilities: A nested dictionary where the outer key is the context (an
      n-1 gram tuple) and the value is another dictionary. This inner
      dictionary's keys are possible next tokens and its values are their
      probabilities.
    tokenize_function: A function that takes a string and returns a list of
      tokens.
  """

  def __init__(
      self,
      dataset: list[str],
      n: int,
      tokenize_function: Callable[[str], list[str]] | None = None,
  ):
    """Initializes the NGramModel.

    Args:
      dataset: A list of strings representing the training documents.
      n: The order of the n-gram model (e.g., 2 for bigram, 3 for trigram).
      tokenize_function: A function that takes a string and returns a list of
        tokens.

    Raises:
      ValueError if n < 2.
    """
    if n < 2:
      raise ValueError("n must be 2 or greater for an n-gram model.")

    def _space_tokenize(text: str) -> list[str]:
      """Implements a whitespace tokenizer."""
      return text.split(" ")

    self.n = n
    if tokenize_function is None:
      tokenize_function = _space_tokenize

    self.tokenize_function = tokenize_function
    self.probabilities = {}
    self.estimate_probabilities(dataset)

  def estimate_probabilities(self, dataset: list[str]) -> None:
    """Estimates the conditional probabilities of n-grams from the dataset.

    This method populates the `self.probabilities` dictionary. The structure
    is a nested dictionary where the outer key is the context (an n-1 gram
    tuple) and the value is another dictionary. This inner dictionary's
    keys are possible next tokens and its values are their probabilities.

    For example, for a trigram model (n=3), a key might be ('the', 'quick')
    and its value could be {'brown': 0.8, 'red': 0.2}.

    Args:
      dataset: A list of strings to train the model on.
    """
    ngram_counts = DefaultDict(Counter)

    for text in dataset:
      tokens = self.tokenize_function(text)
      if len(tokens) < self.n:
        continue  # Skip texts that are too short to form an n-gram.

      # Iterate through the tokens to create n-grams.
      for i in range(len(tokens) - self.n + 1):
        ngram = tuple(tokens[i : i + self.n])
        context = ngram[:-1]
        next_token = ngram[-1]

        ngram_counts[context][next_token] += 1

    if not ngram_counts:
      print(
          "Warning: No n-grams were found in the dataset. The model is empty."
      )
      return

    # Calculate probabilities.
    self.probabilities = DefaultDict(dict)

    for context, next_tokens in ngram_counts.items():
      context_total = sum(next_tokens.values())
      for next_token, next_token_count in next_tokens.items():
        self.probabilities[context][next_token] = (
            next_token_count / context_total
        )

  def generate(
      self,
      num_tokens_to_generate: int,
      prompt: str,
      sampling_mode: str = "random",
  ) -> str:
    """Generates `num_tokens_to_generate` tokens following a given prompt.

    The generation process uses the last n-1 tokens of the prompt as the
    initial context. It then samples from the probability distribution for
    that context to pick the next token. This new token is then used to
    update the context, and the process repeats.

    If at any point the model encounters a context that it has not seen
    during training, it will print a warning and stop generating text.

    Args:
      num_tokens_to_generate: The number of tokens to generate.
      prompt: The initial string to start the generation from.
      sampling_mode: Whether to use random or greedy sampling. Supported options
        are 'random' and 'greedy'.

    Returns:
      A string containing the prompt followed by the generated tokens.
    """
    if sampling_mode not in ["random", "greedy"]:
      raise ValueError(
          f"Sampling mode {sampling_mode} is not supported. Supported options"
          " are 'random' and 'greedy'."
      )

    tokens = self.tokenize_function(prompt)

    if len(tokens) < self.n - 1:
      print(
          f"Warning: Prompt must contain at least {self.n - 1} tokens."
          " Generation failed."
      )
      return prompt

    # The context is the last n-1 tokens of the prompt.
    context = tuple(tokens[-(self.n - 1) :])
    generated_tokens = []

    for _ in range(num_tokens_to_generate):
      if context not in self.probabilities:
        print(
            "⚠️ No valid continuation found. Change the prompt or"
            " try sampling another continuation.\n"
        )
        break

      # Get the distribution of the next possible tokens.
      next_token_distribution = self.probabilities[context]
      possible_next_tokens = list(next_token_distribution.keys())
      token_probabilities = list(next_token_distribution.values())

      # Choose the next token based on the learned probabilities.
      if sampling_mode == "random":
        next_token = random.choices(
            possible_next_tokens, weights=token_probabilities, k=1
        )[0]
      else:
        next_token = max(next_token_distribution.items(), key=lambda x: x[1])[0]
      generated_tokens.append(next_token)

      # Update the context by sliding the window.
      context = context[1:] + (next_token,)

    return prompt + " " + " ".join(generated_tokens)
