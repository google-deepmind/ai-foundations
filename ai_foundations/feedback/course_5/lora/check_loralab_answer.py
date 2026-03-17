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

"""A utility function to give feedback in the LoRA lab.

A student has to compute the proportion of trainable parameters in a dense layer
with LoRA compared to a regular dense layer.
"""

import jax.numpy as jnp


def check_loralab_answer(
    answer: str, num_units: int = 512, rank: int = 8, tolerance: float = 0.0001
) -> None:
  """Check the answer on the proportion of LoRA parameters.

  A student has to compute the proportion of trainable parameters in a dense
  layer with LoRA compared to a regular dense layer.

  This function marks an answer as correct irrespective of whether or not a
  student considered the bias weights.

  The answer is correct if it is within the tolerance of the correct answer.
  The correct answer is the proportion of LoRA parameters to the total number
  of parameters in the full weight matrix. The correct answer is different
  if the student considers bias weights or not.

  Args:
    answer: The answer that the student gives.
    num_units: The number of input and output units.
    rank: The rank of the LoRA matrices.
    tolerance: Tolerance allowed for the correct answer.
  """

  answer_without_bias = 2 * rank / num_units
  # The answer with bias is simplified from:
  # (2 * rank * num_units + num_units) / (num_units * num_units + num_units).
  answer_with_bias = (2 * rank + 1) / (num_units + 1)

  if answer > 1 or answer < 0:
    print("❌ The answer needs to be between 0 and 1.")
  elif ((jnp.abs(answer - answer_without_bias) > tolerance)
        and (jnp.abs(answer - answer_with_bias) > tolerance)):
    print(
        "❌ Your answer is not correct.\n"
        f"Matrices A and B each have {rank} x {num_units}"
        f" = {rank * num_units} trainable parameters.\n"
        f"The full weight matrix has {num_units} x {num_units}"
        f" = {num_units * num_units} parameters.\n"
        f"There are {num_units} bias weights."
    )
  else:
    print(
        "✅ Nice! Your answer looks correct.\n"
        "If you consider bias weights, the answer is"
        f" {jnp.round(answer_with_bias, decimals=5)}.\n"
        "If you don't consider bias weights, the answer is"
        f" {jnp.round(answer_without_bias, decimals=5)}."
    )
