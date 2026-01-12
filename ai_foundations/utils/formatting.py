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

"""Utility functions for formatting large numbers for display.

This module provides helper functions to convert large floating point numbers,
such as parameter counts or FLOPs, into more human-readable string formats.
"""

import IPython.display

display = IPython.display.display
HTML = IPython.display.HTML


def format_flops(flops: float) -> str:
  """Formats a large FLOPs number into a human-readable string.

  Args:
    flops: A floating point number representing the total FLOPs.

  Returns:
    A string formatted using scientific notation.
  """
  if flops is None:
    return "N/A"

  return f"{flops:.2e}"


def format_large_number(num: float) -> str:
  """Converts a large number to a string with a suffix.

  This function takes a number and formats it with the most appropriate
  magnitude suffix (trillion, billion, million) to improve readability.

  Args:
    num: The floating point number to be formatted.

  Returns:
    str: A string formatted with a suffix (e.g., "1.5 trillion") or as a
      comma-separated number if smaller than a million.
  """
  if num is None:
    return "N/A"

  trillion = 1e12
  billion = 1e9
  million = 1e6

  if num >= trillion:
    return f"{num / trillion:,.1f} trillion"
  if num >= billion:
    return f"{num / billion:,.1f} billion"
  if num >= million:
    return f"{num / million:,.1f} million"
  return f"{num:,.0f}"


def bytes_to_gb(num_bytes: float) -> float:
  """Converts a raw byte count into gigabytes (GB).

  Args:
    num_bytes: The number of bytes to be converted.

  Returns:
    The equivalent number of gigabytes.
  """
  return num_bytes / (1024**3)


def display_memory(
    component_name: str, gb_value: float, decimal_places: int = 2
) -> None:
  """Formats and displays a memory value in a styled HTML block.

  Args:
    component_name: The name of the memory component (e.g., "Parameters").
    gb_value: The memory size in gigabytes.
    decimal_places: The number of decimal places to format the output to.
  """
  display(
      HTML(
          f"<blockquote>Memory for {component_name}:"
          f" <b>{gb_value:.{decimal_places}f} GB</b></blockquote>"
      )
  )
