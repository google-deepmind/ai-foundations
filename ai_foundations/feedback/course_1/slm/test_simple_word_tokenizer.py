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

"""A utility function for a tokenizer exercise.

This module provides a function to test a learner's custom simple word
tokenizer, checking its internal vocabulary and the consistency of its encode
and decode methods.
"""

from typing import Any, List
from ai_foundations.feedback.utils import render_feedback


def test_simple_word_tokenizer(
    tokenizer: Any, vocabulary: List[str], train_dataset: List[str]
):
  """Tests the simple word tokenizer's vocabulary and its encode/decode methods.

  This function performs two checks:
  1. It validates that the simple word tokenizer's internal vocabulary matches
     the expected vocabulary.
  2. It ensures that encoding a sentence and then decoding it results in the
     original text.

  Args:
    tokenizer: The learner's custom tokenizer object to be tested.
    vocabulary: The expected list of vocabulary tokens.
    train_dataset: A list of sentences for testing the encode/decode
      functionality.
  """

  hint = """
    This test checks two things:<br><br>
    1. <b>Vocabulary Check:</b> It verifies that the vocabulary stored inside
       your tokenizer (<code>tokenizer.vocabulary</code>) is the same as the
       original <code>vocabulary</code> list.<br>
    2. <b>Encode/Decode Check:</b> It tests if encoding a sentence and then
       immediately decoding it returns the original sentence. This ensures
       that your <code>encode</code> and <code>decode</code> methods work
       together correctly.
    """

  try:
    if sorted(tokenizer.vocabulary) != sorted(vocabulary):
      raise ValueError(
          "The tokenizer's vocabulary does not match the expected vocabulary."
      )

    if (
        tokenizer.decode(tokenizer.encode(train_dataset[0])).split()
        != train_dataset[0].split()
    ):
      raise ValueError(
          "Encoding and then decoding a sentence does not produce the original"
          " text."
      )

  except ValueError as e:
    render_feedback(e, hint)

  else:
    print("✅ Nice! The tokenizer seems to be working correctly.")
