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

"""A helper module for rendering interactive feedback.

This module provides a utility function for displaying standardized, formatted
error messages and optionally prompting the user to view a hint.
"""

import time
from typing import Optional

import IPython.display


display = IPython.display.display
HTML = IPython.display.HTML


def render_feedback(error: Exception, hint: Optional[str] = None):
  """Displays an error message and optionally asks to show a hint.

  Args:
    error: The Exception containing the feedback.
    hint: The optional hint string to display if the user agrees.
  """

  # Unpack the title and detail from the ValueError.
  if len(error.args) == 2:
    title, detail = error.args
  elif isinstance(error.args[0], tuple) and len(error.args[0]) == 2:
    title, detail = error.args[0]
  else:
    # Handle cases where the error has only one argument.
    title = "Sorry, your answer is not correct."
    detail = str(error)

  # Display the primary error message.
  display(HTML(f"<h3>❌ {title}</h3><p>{detail}</p>"))

  # If a hint is provided, ask the user if they want to see it.
  if hint:
    time.sleep(1)
    give_hints = input("Would you like a hint? Type Yes or No. ")
    if give_hints.lower() in ["yes", "y"]:
      display(HTML(f"<h3>Hint:</h3><p>{hint}</p>"))
