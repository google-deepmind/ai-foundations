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

"""Utility functions to test an implementation of Challenge Lab 2 tasks."""

import collections
from collections.abc import Callable, Sequence

from ai_foundations.tokenization.bpe_arabic_tokenizer import ArabicBPETokenizer
import matplotlib.pyplot as plt
import numpy as np

Counter = collections.Counter


def test_get_token_counts(
    corpus: list[str] | str,
    get_token_counts: Callable[
        [list[str] | str, ArabicBPETokenizer], Counter[str]
    ],
    num_increments: int = 4,
    merges_per_increment: int = 300,
    show_n_counts: int = 10,
) -> None:
  """Tests a `get_token_counts` function that was implemented by a student.

  This function initializes an `ArabicBPETokenizer`, then iteratively
  continues BPE training for a specified number of increments. At each
  increment, it calls the provided `get_token_counts` function to
  get the current token counts and visualizes the evolution of token
  frequency distributions on a log-log plot.

  Args:
    corpus: The corpus of text used for tokenization.
    get_token_counts: The function to be tested, which should calculate token
      counts from a corpus and tokenizer.
    num_increments: The number of times to continue BPE training.
    merges_per_increment: The number of merges to perform in each increment.
    show_n_counts: The number of top counts to print at each increment.
  """
  print("Testing get_token_counts")
  print("-------------------------\n")
  print("Initialising with zero merges:\n")

  tokenizer = ArabicBPETokenizer(corpus, num_merges=0)

  counts_evolution = []
  label_list = []

  for i in range(num_increments):
    print(f"Phase {i+1}/{num_increments}:")
    tokenizer.continue_bpe(corpus, num_merges=merges_per_increment)

    num_merges = len(tokenizer.merges)
    label_list.append(f"merges={num_merges}")

    token_counts = get_token_counts(corpus, tokenizer)

    counts_evolution.append(token_counts)
    top_n_counts = token_counts.most_common(show_n_counts)
    print(f"First {show_n_counts} counts: {top_n_counts}")

  # Visualize the results on a log-log plot.
  plot_multiple_token_counts(counts_evolution, label_list)


def plot_multiple_token_counts(
    token_counts_list: Sequence[Counter[str]],
    label_list: Sequence[str],
    title: str | None = None,
) -> None:
  """Plots token frequency distributions for multiple tokenization states.

  Creates a log-log plot showing token frequencies ranked by occurrence
  for each set of token counts. Each line represents a different
  tokenization state, allowing comparison of how token distributions
  evolve (e.g., during BPE training). Tokens are sorted by frequency
  (descending) and plotted on a log-log scale with rank on the x-axis and
  count on the y-axis.

  Args:
    token_counts_list: Sequence of Counter objects, each mapping tokens to
      their frequencies.
    label_list: Sequence of labels for each line in the plot.
    title: Optional plot title. If None, no title is set on the plot.
  """
  for token_counts, line_label in zip(
      token_counts_list, label_list, strict=True
  ):
    counts = np.sort(list(token_counts.values()))[::-1]
    plt.plot(np.arange(1, len(counts) + 1), counts, label=line_label)
  plt.legend()

  if title is not None:
    plt.title(title)
  plt.yscale("log")
  plt.xscale("log")
  plt.xlabel("Token Frequency Rank")
  plt.ylabel("Token Frequency")


def test_count_split_diacritics_in_sequence(
    count_split_diacritics: Callable[[list[str]], int],
) -> None:
  """Tests a student-implemented function that counts tokenized split diacritics.

  A split diacritic is defined as a boundary between two tokens where the
  preceding token ends with a base letter and the following token begins with a
  non-spacing mark (a diacritic/harakat). The function tests a
  student-implemented function that counts these splits in a tokenization of an
  Arabic text string.

  Args:
    count_split_diacritics: The student's implementation to be tested.
  """

  original_text = "مَرْحَبًا"

  print("Illustrative test")
  print("-----------------")
  print(f"Original text: {original_text}")

  # Case 1: 3 tokens, 2 splits
  case_1 = ["مَر", "ْح", "َبًا"]
  print(f"\nTokenization of {original_text} to test (case 1):")
  for i, token in enumerate(case_1):
    print(f"Token {i}: {token}")
  test_output_1 = count_split_diacritics(case_1)
  print(f"Total number of split diacritics: {test_output_1}")

  # Case 2: 2 tokens, 0 splits
  case_2 = ["مَرْ", "حَبًا"]
  print(f"\nTokenization of {original_text} to test (case 2):")
  for i, token in enumerate(case_2):
    print(f"Token {i}: {token}")
  test_output_2 = count_split_diacritics(case_2)
  print(f"Total number of split diacritics: {test_output_2}")

  # --- Hidden Tests with Varying Token Counts ---
  print("\nHidden tests for assessment")
  print("---------------------------")

  # Case 3: 2 tokens (Minimal sequence)
  # Testing a single split where the second token is ONLY a diacritic.
  # Text: "بِ" -> ["ب", "ِ"]
  case_3 = ["ب", "ِ"]
  print("Tokenization of [hidden string] to test (case 3):")
  test_output_3 = count_split_diacritics(case_3)
  print(f"Total number of split diacritics: {test_output_3}")

  # Case 4: 4 tokens (Multi-split sequence)
  # Testing multiple splits across a medium-length sequence.
  # Text: "كُتِبَ" -> ["ك", "ُت", "ِب", "َ"]
  # Splits: (ك + ُ), (ت + ِ), (ب + َ)
  case_4 = ["ك", "ُت", "ِب", "َ"]
  print("\nTokenization of [hidden string] to test (case 4):")
  test_output_4 = count_split_diacritics(case_4)
  print(f"Total number of split diacritics: {test_output_4}")

  # Case 5: 5 tokens (Mixed boundary types)
  # Testing a longer sequence where some boundaries are clean and others are
  # split.
  # Text: "السَّلَامُ" -> ["ال", "س", "ّ", "َلَام", "ُ"]
  # Splits: (س + ّ), (م + ُ). Note: (ال + س) is NOT a split.
  case_5 = ["ال", "س", "ّ", "َلَام", "ُ"]
  print("\nTokenization of [hidden string] to test (case 5):")
  test_output_5 = count_split_diacritics(case_5)
  print(f"Total number of split diacritics: {test_output_5}")


def test_count_split_diacritics_in_corpus(
    corpus: Sequence[str],
    tokenizer: ArabicBPETokenizer,
    count_split_diacritics_in_corpus: Callable[
        [Sequence[str], ArabicBPETokenizer], int
    ],
) -> None:
  """Tests a function that counts split diacritics across a corpus.

  This function evaluates the provided `count_split_diacritics_in_corpus`
  by splitting the input `corpus` into folds and calling the function
  on each fold. It prints the counts obtained for each fold.

  Args:
    corpus: The corpus, expected to be a list of paragraphs, where each
      paragraph is a list of tokens.
    tokenizer: An instance of `ArabicBPETokenizer`.
    count_split_diacritics_in_corpus: The function to be tested, which should
      take a sub-corpus and the tokenizer and return the count of split
      diacritics.
  """
  print("Testing count_split_diacritics_in_corpus")
  n_paragraphs = len(corpus)
  n_folds = 5
  if n_paragraphs != 200:
    print("❌ Failed: problem with corpus.")
    return

  len_fold = n_paragraphs // n_folds
  counts_found = np.empty(n_folds, dtype=int)

  for fold_idx in range(n_folds):
    print(f"Fold {fold_idx + 1}:")
    fold = corpus[fold_idx * len_fold : (fold_idx + 1) * len_fold]
    counts_found[fold_idx] = count_split_diacritics_in_corpus(fold, tokenizer)
    print(f"\tSplit diacritic count = {counts_found[fold_idx]}")

  num_splits_found = np.sum(counts_found)

  print(f"\nFound {num_splits_found} diacritic splits in total corpus.")
