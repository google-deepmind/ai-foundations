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

"""A loader utility for pre-trained Gemma models.

This module contains a function to instantiate a Gemma model and load its
corresponding tokenizer and pre-trained weights.
"""

from typing import Any, Literal, Mapping, Tuple

from ai_foundations import attention
from gemma import gm


def load_gemma(
    model_name: Literal[
        "Gemma-1B", "Gemma-4B", "Gemma-1B-AttentionWeight"
    ] = "Gemma-1B",
) -> Tuple[
    gm.text.Gemma3Tokenizer,
    gm.nn.Gemma3_1B | gm.nn.Gemma3_4B,
    Mapping[str, Any],
]:
  """Loads a Gemma model and its associated tokenizer and parameters.

  Args:
    model_name: The name of the Gemma model to load. Options are: 'Gemma-1B' and
      'Gemma-4B'.

  Returns:
    tokenizer: Tokenizer for the specified Gemma model.
    model: The Gemma model.
    params: The parameters for the specified Gemma model.

  Raises:
    ValueError: If an unsupported model name is provided.
  """

  # Model loading based on model_name.
  if model_name in ["Gemma-1B", "Gemma-1B-AttentionWeight"]:
    tokenizer = gm.text.Gemma3Tokenizer()
    if model_name == "Gemma-1B":
      model = gm.nn.Gemma3_1B()
    else:
      model = attention.AttentionWeightGemma3_1B()
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_PT)
  elif model_name == "Gemma-4B":
    tokenizer = gm.text.Gemma3Tokenizer()
    model = gm.nn.Gemma3_4B()
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_PT)
  else:
    raise ValueError(
        f"Unsupported model name: {model_name}."
        " Please use 'Gemma-1B', 'Gemma-4B', or 'Gemma-1B-AttentionWeight'."
    )

  return tokenizer, model, params
