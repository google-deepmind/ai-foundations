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

"""Hyperparameters for the model."""

TEST_MODEL_HYPERPARAMETERS = [
    {
        "max_length": 128,
        "embedding_dim": 256,
        "mlp_dim": 384,
        "num_heads": 4,
        "num_blocks": 2,
        "vocabulary_size": 262144,
    },
    {
        "max_length": 128,
        "embedding_dim": 128,
        "mlp_dim": 512,
        "num_heads": 6,
        "num_blocks": 8,
        "vocabulary_size": 2048,
    },
]
