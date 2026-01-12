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

"""A utility function to test the implementation of the clean_html function.

This module provides a function to test a learner's implementation of a
`clean_html` function, ensuring it correctly strips simple HTML tags and
converts common HTML entities.
"""

import collections
import re
from typing import Callable, List, Tuple
from ai_foundations.feedback.utils import render_feedback

Counter = collections.Counter


def _norm_spaces(s: str) -> str:
  """Normalize spaces in a string.

  Collapses consecutive whitespace into a single space and trims leading
  and trailing whitespace.

  Args:
    s: Input string.

  Returns:
    A new string with normalized spacing.
  """
  return re.sub(r"\s+", " ", s.strip())


def _tokens(s: str) -> List[str]:
  """Split a string into word tokens.

  Uses a Unicode-aware regex to extract alphanumeric tokens (letters,
  digits, underscore) from the input.

  Args:
    s: Input string.

  Returns:
    A list of tokens found in the string.
  """
  return re.findall(r"\w+", s, flags=re.UNICODE)


def _html_escape(text: str) -> str:
  """HTML-escape a snippet for safe display in feedback.

  Converts the characters "&", "<", and ">" to their entity forms so they
  render literally in feedback messages.

  Args:
    text: Raw text to escape.

  Returns:
    Escaped text.
  """
  return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def compare_html_result(student_text: str, gold_text: str) -> Tuple[bool, str]:
  """Compare a learner's output against the expected gold output.

  Applies a sequence of ordered checks to diagnose common mistakes in
  simple HTML cleaning tasks, such as unremoved tags, unconverted entity
  references, spacing issues, or missing words.

  Args:
      student_text: The string returned by the learner's function.
      gold_text: The reference string that represents the correct output.

  Returns:
      A tuple of:
          - bool: True if the student output is accepted as correct; False
          otherwise.
          - str: A feedback message describing the result.
  """
  if not isinstance(student_text, str):
    return (
        False,
        "❌ Incorrect output type. Your function must return a string.",
    )

  # Check for exact match (allowing space normalization).
  is_exact = student_text == gold_text
  is_norm_equal = _norm_spaces(student_text) == _norm_spaces(gold_text)
  if is_exact or is_norm_equal:
    return (True, "✅ Nice! Your aswer looks correct.")

  # Check if  HTML tags have not been removed.
  if re.search(r"<[^>]+>", student_text):
    return (
        False,
        (
            "❌ HTML tags are still present. Remove all &lt;...&gt; tags. You"
            ' can use a non-greedy regex like re.sub(r"&lt;.*?&gt;", "", text).'
        ),
    )

  # Check if common HTML entities have not been converted.
  remaining_entities_set = set()
  for ent in ("&nbsp;", "&amp;", "&lt;", "&gt;"):
    if ent in student_text:
      remaining_entities_set.add(ent)
  # Detect any other generic entity references (named or numeric).
  generic_ents = re.findall(
      r"&(?:#\d+|#x[0-9A-Fa-f]+|[A-Za-z]+);", student_text
  )
  for ent in generic_ents:
    remaining_entities_set.add(ent)
  if remaining_entities_set:
    ents_raw = ", ".join(sorted(remaining_entities_set))
    ents = _html_escape(ents_raw)
    return (
        False,
        (
            "❌ Some HTML entities were not converted: "
            f'{ents}. Replace &amp;nbsp;→ "" '
            "(or regular space as appropriate), "
            '&amp;amp;→"&amp;", &amp;lt;→"&lt;" and &amp;gt;→"&gt;".'
        ),
    )

  # Check if words are correct but spacing/punctuation differs (not covered by
  # Rule 1).
  if _tokens(student_text) == _tokens(gold_text):
    return (
        False,
        (
            "❌ The list of words is correct, but spacing/punctuation differs. "
            "Normalize whitespace and ensure entities are properly replaced."
        ),
    )

  # Check for missing tokens.
  missing = []
  c_gold = Counter(_tokens(gold_text))
  c_stu = Counter(_tokens(student_text))
  for w, cnt in c_gold.items():
    if c_stu[w] < cnt:
      missing.append(w)
  if missing:
    missing_unique = sorted(set(missing), key=missing.index)
    return (
        False,
        (
            "❌ The following tokens are missing from the output:"
            f" {', '.join(missing_unique)}"
        ),
    )

  # Check for any other mismatch.
  return (False, "❌ The output does not match the expected result.")


def test_clean_html(clean_html: Callable[[str], str]) -> None:
  """Test a learner's implementation of the `clean_html` function.

  Runs one official test case and provides categorized feedback if the
  learner's output does not match the expected result.

  Args:
      clean_html: The learner's function to be tested. It should take a string
        containing simple HTML and entity references and return a cleaned string
        with tags removed and common entities converted.
  """

  try:
    text = (
        "<p>The Krowor Municipal District was carved out of the"
        " Ledzokuku-Krowor Municipal District in 2018 &amp; it's population is"
        " &gt; 200000.</p>"
    )
    gold = (
        "The Krowor Municipal District was carved out of the Ledzokuku-Krowor"
        " Municipal District in 2018 & it's population is > 200000."
    )
    student = clean_html(text)
    passed, msg = compare_html_result(student, gold)
    if not passed:
      raise ValueError(
          "Sorry, your answer is not correct.",
          msg,
      )
  except ValueError as e:
    render_feedback(e)
  else:
    print("✅ Nice! Your answer looks correct.")
