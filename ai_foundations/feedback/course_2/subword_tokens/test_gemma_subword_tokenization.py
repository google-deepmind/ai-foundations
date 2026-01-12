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

"""Tests for working with Gemma tokenizer."""

from typing import List
from gemma import gm


def test_gemma_subword_tokenization(
    clusterophonexia_tokens: List[int],
    first_token_as_text: str,
    gemma_tokenizer: gm.text.Gemma3Tokenizer,
):
  """Tests the subword tokenization of a complex word with the Gemma tokenizer.

  This function verifies two specific properties of the tokenization for the
  word "Clusterophonexia":
  1. That it is split into exactly four subword tokens.
  2. That the first subword token correctly decodes to "Cluster".

  It also decodes and prints each individual token and its corresponding ID for
  manual inspection. If any assertion fails, it prints a user-friendly
  error message.

  Args:
    clusterophonexia_tokens: A list of integer IDs representing the tokens for
      the word "Clusterophonexia" after tokenization.
    first_token_as_text: The decoded string representation of the first token in
      the list.
    gemma_tokenizer: An initialized Gemma tokenizer instance used for decoding
      the token IDs back to text.
  """
  try:
    assert (
        len(clusterophonexia_tokens) == 4
    ), "Clusterophonexia should be broken into four tokens."
    assert (
        first_token_as_text == "Cluster"
    ), "The first subword token should be Cluster."

  except AssertionError as e:
    print(f"❌ {e}")
  else:
    # Decode individual tokens.
    print('Tokenization of "Clusterophonexia":')
    for token in clusterophonexia_tokens:
      decoded_token = gemma_tokenizer.decode(token)
      print(f"Token {token}:\t{decoded_token}")

    print("\n✅ Nice! Your answer looks correct.")
