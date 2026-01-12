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

"""A utility function to load Gemma embeddings from a URL."""

import io
from typing import List, Tuple
import urllib.request
import numpy as np


def load_gemma_embeddings(url: str) -> Tuple[np.ndarray, List[str]]:
  """Load Gemma embeddings from a URL.

  Args:
    url: The URL to load the embeddings from.

  Returns:
    A tuple containing the embeddings and labels.
  """
  with urllib.request.urlopen(url) as response:
    buffer = io.BytesIO(response.read())
    bundle = np.load(buffer)
  return bundle['embeddings'], bundle['labels'].tolist()
