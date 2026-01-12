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

"""Test functions for training."""

from .constants import TEST_MODEL_HYPERPARAMETERS
from .reference_implementations import parameter_count_attention
from .reference_implementations import parameter_count_embedding
from .reference_implementations import parameter_count_layer_norm
from .reference_implementations import parameter_count_mlp
from .reference_implementations import parameter_count_output_layer
from .reference_implementations import parameter_count_transformer
from .reference_implementations import parameter_count_transformer_block
from .test_parameter_counts import test_parameter_count_attention
from .test_parameter_counts import test_parameter_count_embedding
from .test_parameter_counts import test_parameter_count_layer_norm
from .test_parameter_counts import test_parameter_count_mlp
from .test_parameter_counts import test_parameter_count_output_layer
from .test_parameter_counts import test_parameter_count_transformer
from .test_parameter_counts import test_parameter_count_transformer_block
