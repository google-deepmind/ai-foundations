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

"""Text generation and inference utilities for language models."""

from .gemma import prompt_attention_transformer_model
from .gemma import prompt_transformer_model
from .generate import generate_text
from .generate import greedy_decoding
from .generate import sampling as random_decoding
from .loaders import load_gemma
