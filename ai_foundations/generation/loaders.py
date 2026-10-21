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

import contextlib
import functools
from typing import Any, Iterator, Literal, Mapping, Tuple

from ai_foundations import attention
from gemma import gm

from orbax import checkpoint as ocp

# Maximum amount of checkpoint data, in GB, that is held in host memory while
# the weights are copied to the GPU. With the default (96 GB), the whole
# Gemma-4B checkpoint is held in host memory at once, which can use up the RAM
# of a Colab VM. The value must be larger than the largest single parameter
# array (about 1.3 GB for Gemma-4B).
_RESTORE_CONCURRENT_GB = 2


@contextlib.contextmanager
def _limit_restore_memory(concurrent_gb: int) -> Iterator[None]:
  """Limits the host memory that orbax uses while restoring a checkpoint.

  Args:
    concurrent_gb: The maximum amount of checkpoint data, in GB, that is held
      in host memory at the same time.

  Yields:
    Nothing. Checkpoints restored inside the context use the limit.
  """
  original_init = ocp.StandardCheckpointHandler.__init__

  @functools.wraps(original_init)
  def init(self, *args, **kwargs):
    kwargs.setdefault("restore_concurrent_gb", concurrent_gb)
    original_init(self, *args, **kwargs)

  ocp.StandardCheckpointHandler.__init__ = init
  try:
    yield
  finally:
    ocp.StandardCheckpointHandler.__init__ = original_init


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
    with _limit_restore_memory(_RESTORE_CONCURRENT_GB):
      params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_PT)
  elif model_name == "Gemma-4B":
    tokenizer = gm.text.Gemma3Tokenizer()
    model = gm.nn.Gemma3_4B()
    with _limit_restore_memory(_RESTORE_CONCURRENT_GB):
      params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_PT)
  else:
    raise ValueError(
        f"Unsupported model name: {model_name}."
        " Please use 'Gemma-1B', 'Gemma-4B', or 'Gemma-1B-AttentionWeight'."
    )

  return tokenizer, model, params
