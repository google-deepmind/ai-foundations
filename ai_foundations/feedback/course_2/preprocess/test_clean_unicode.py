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

"""A utility function to test the implementation of the clean_unicode function.

This module provides a function to test a learner's implementation of a
`clean_unicode` function, ensuring it correctly removes special characters and
preserves punctuation.
"""

import collections
import re
from typing import Callable, List, Tuple
from ai_foundations.feedback.utils import render_feedback

Counter = collections.Counter


def _norm_spaces(s: str) -> str:
  """Normalize spaces in a string.

  This function collapses multiple spaces into a single space and trims
  any leading or trailing whitespace.

  Args:
    s: Input string.

  Returns:
    A new string with normalized spacing.
  """
  return re.sub(r"\s+", " ", s.strip())


def _tokens(s: str) -> List[str]:
  """Split a string into word tokens.

  Uses a regex to extract alphanumeric tokens (letters, digits, underscore)
  from the input string.

  Args:
    s: Input string.

  Returns:
    A list of tokens found in the string.
  """
  return re.findall(r"\w+", s, flags=re.UNICODE)


def compare_unicode_result(
    student_text: str, gold_text: str
) -> Tuple[bool, str]:
  """Compare a learner's output against the expected gold output.

  This function applies a sequence of ordered rules to identify common
  mistakes in text cleaning tasks, such as spacing issues, missing words,
  leftover special characters, or punctuation errors.

  Args:
    student_text: The string returned by the learner's function.
    gold_text: The reference string that represents the correct output.

  Returns:
    A tuple of:
      - bool: True if the student output is accepted as correct, False
      otherwise.
      - str: A feedback message describing the result.
  """
  if not isinstance(student_text, str):
    return (False, "❌ Incorrect output. Your function should return a string.")

  # Check for exact match.
  if student_text == gold_text or _norm_spaces(student_text) == _norm_spaces(
      gold_text
  ):
    return (True, "✅ Nice! Your answer looks correct.")

  # Check whether special characters have not been removed (currency, emoji,
  # etc.)
  if re.search(r"[^\w\s\.\!\?]", student_text):
    return (
        False,
        (
            "❌ Incorrect output. There are still special characters that have"
            " not been removed."
        ),
    )

  # Check if words are correct, but punctuation differs.
  stu_tokens = _tokens(student_text)
  gold_tokens = _tokens(gold_text)
  if stu_tokens == gold_tokens:
    return (
        False,
        (
            "❌ Incorrect output. Words are correct, but punctuation differs. "
            "You should add punctuation marks to the cleaned text."
        ),
    )

  # Check for missing tokens.
  missing = []
  c_gold = Counter(gold_tokens)
  c_stu = Counter(stu_tokens)
  for w, cnt in c_gold.items():
    if c_stu[w] < cnt:
      missing.append(w)
  if missing:
    missing_unique = sorted(set(missing), key=missing.index)
    return (
        False,
        (
            "❌ Incorrect output. The following tokens are missing from the"
            f" output: {', '.join(missing_unique)}"
        ),
    )

  # Check for any other mismatch.
  return (
      False,
      "❌ Incorrect output. Your output does not match the expected result.",
  )


def test_clean_unicode(clean_unicode: Callable[[str], str]) -> None:
  """Test a learner's implementation of the `clean_unicode` function.

  Runs one official test case and provides detailed feedback if the learner's
  output does not match the expected result.

  Args:
    clean_unicode: The learner's function to be tested. It should take a string
      with special characters and return a cleaned string with only valid words
      and punctuation preserved.
  """

  hint = (
      "Remember to:<br>"
      "- Return a string, not a list or set.<br>"
      "- Remove all special characters (currency symbols, emojis, etc.).<br>"
      "- Preserve punctuation marks like <code>.</code> and <code>!</code>.<br>"
      "- Ensure all words from the input remain in the output.<br>"
      "- Avoid extra or missing spaces."
  )

  try:
    text = "Bag of rice now cost ₦150000 naira. Ah! 😱 Èdakun o"
    gold = "Bag of rice now cost 150000 naira. Ah!  Èdakun o"
    student = clean_unicode(text)
    passed, msg = compare_unicode_result(student, gold)
    if not passed:
      raise ValueError("Sorry, your answer is not correct.", msg)
  except ValueError as e:
    render_feedback(e, hint)
  else:
    print("✅ Nice! Your answer looks correct.")
