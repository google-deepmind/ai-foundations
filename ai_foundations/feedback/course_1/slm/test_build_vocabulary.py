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

"""A utility function to test the implementation of the build_vocab function.

This module provides a function to test a learner's implementation of a
`build_vocab` function, ensuring it correctly returns a list of unique tokens.
"""

from typing import Callable, List
from ai_foundations.feedback.utils import render_feedback


def test_build_vocabulary(build_vocabulary: Callable[[List[str]], List[str]]):
  """Tests the learner's implementation of a `build_vocabulary` function.

  This function validates that the provided `build_vocabulary` function
  correctly converts a list of tokens into a list of unique tokens, ensuring the
  final output is a list.

  Args:
    build_vocabulary: The learner's function to be tested. It should accept a
      list of strings and return a list of unique strings.
  """

  hint = """
    1. Create a unique set of tokens e.g, if you have
       <code>['hello', 'world', 'world']</code>, it becomes
       <code>{'hello', 'world'}</code>.
       There is a Python <code>set</code> function you can use.<br>
    2. Convert the set to a list e.g <code>{'hello', 'world'}</code> becomes
       <code>['hello', 'world']</code>. There is a Python <code>list</code>
       function that you can use.
    """

  try:
    if isinstance(build_vocabulary(["hello", "world", "world"]), set):
      raise ValueError(
          "Sorry, your answer is not correct.",
          "Make sure that you return a list, not a set.",
      )
    if build_vocabulary(["hello", "world", "world"]) != [
        "hello",
        "world",
    ] and build_vocabulary(["hello", "world", "world"]) != ["world", "hello"]:
      raise ValueError(
          "Sorry, your answer is not correct.",
          "Your function does not return the expected list of unique words.",
      )

  except ValueError as e:
    render_feedback(e, hint)

  else:
    print("✅ Nice! Your answer looks correct.")
