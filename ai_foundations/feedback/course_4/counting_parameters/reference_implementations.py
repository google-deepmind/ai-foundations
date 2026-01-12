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

"""Reference implementations for parameter counting."""

from typing import Dict


def parameter_count_layer_norm(hyperparams: Dict[str, int]) -> int:
  """Computes the number of trainable parameters for a LayerNorm.

  Assumes a standard LayerNorm over the model dimension with learnable scale
  (gamma) and bias (beta) parameters, each of size `embedding_dim`.

  Args:
    hyperparams: Model hyperparameters. Expects `"embedding_dim"`.

  Returns:
    Total parameters for LayerNorm (scale + bias) equal to `2 * embedding_dim`.
  """
  embedding_dim = hyperparams["embedding_dim"]
  parameter_count = embedding_dim + embedding_dim
  return parameter_count


def parameter_count_attention(hyperparams: Dict[str, int]) -> int:
  """Computes parameters for a multi-head attention sublayer.

  Counts parameters for the query, key, value, and output linear projections,
  each modeled as a dense layer with bias of shape
  `embedding_dim x embedding_dim` plus a bias vector of size `embedding_dim`.

  Args:
     hyperparams: Model hyperparameters. Expects `"embedding_dim"`.

  Returns:
    Total number of trainable parameters for the attention sublayer.
  """
  embedding_dim = hyperparams["embedding_dim"]
  # Parameters for query projection.
  # Note that for the key, query, and value projections,
  # the first dimension is d_head * n_head which happens to be embedding_dim.
  q_parameter_count = (embedding_dim + 1) * embedding_dim
  # Parameters for key projection.
  k_parameter_count = (embedding_dim + 1) * embedding_dim

  # Parameters for value projection.
  v_parameter_count = (embedding_dim + 1) * embedding_dim

  # Parameters for output projection.
  o_parameter_count = (embedding_dim + 1) * embedding_dim

  # Parameters for layer normalization component.
  parameter_count = (
      q_parameter_count
      + k_parameter_count
      + v_parameter_count
      + o_parameter_count
  )
  return parameter_count


def parameter_count_mlp(hyperparams: Dict[str, int]) -> int:
  """Computes parameters for the MLP sublayer.

  The MLP is modeled as two dense layers with biases:
  - First projection: `embedding_dim -> mlp_dim`
  - Second projection: `mlp_dim -> embedding_dim`

  Args:
    hyperparams: Model hyperparameters. Expects `"embedding_dim"` and
      `"mlp_dim"`.

  Returns:
    Total number of trainable parameters for the MLP sublayer.
  """
  embedding_dim = hyperparams["embedding_dim"]
  mlp_dim = hyperparams["mlp_dim"]

  # Parameters for first projection component.
  ffn_parameter_count = (embedding_dim + 1) * mlp_dim
  # Parameters for second projection component.
  output_parameter_count = (mlp_dim + 1) * embedding_dim

  parameter_count = ffn_parameter_count + output_parameter_count
  return parameter_count


def parameter_count_embedding(hyperparams: Dict[str, int]) -> int:
  """Computes parameters for the token embedding matrix.

  The embedding matrix has shape `vocabulary_size x embedding_dim`.

  Args:
    hyperparams: Model hyperparameters. Expects `"vocabulary_size"` and
      `"embedding_dim"`.

  Returns:
      int: Total number of trainable parameters for the embedding layer.
  """
  vocabulary_size = hyperparams["vocabulary_size"]
  embedding_dim = hyperparams["embedding_dim"]
  # The embedding matrix is of size `vocabulary_size` x `embedding_dim`.
  parameter_count = vocabulary_size * embedding_dim
  return parameter_count


def parameter_count_output_layer(hyperparams: Dict[str, int]) -> int:
  """Computes parameters for the output projection layer.

  The output projection maps from `embedding_dim` to `vocabulary_size` and
  includes a bias term for each vocabulary entry.

  Args:
    hyperparams: Model hyperparameters. Expects `"vocabulary_size"` and
      `"embedding_dim"`.

  Returns:
    Total number of trainable parameters for the output layer.
  """
  embedding_dim = hyperparams["embedding_dim"]
  vocabulary_size = hyperparams["vocabulary_size"]

  # Parameters for output projection.
  output_parameter_count = (embedding_dim + 1) * vocabulary_size

  # Only the projection component has parameters,
  # the activation function does not.
  parameter_count = output_parameter_count

  return parameter_count


def parameter_count_transformer_block(hyperparams: Dict[str, int]) -> int:
  """Computes parameters for a transformer block (attention + MLP).

  Sums the parameters from the multi-head attention component (plus its
  LayerNorm) and the MLP component (plus its LayerNorm).

  Args:
    hyperparams : Model hyperparameters needed by the attention and MLP
      parameter count calculators. Typically includes `"embedding_dim"`,
      `"mlp_dim"`, and potentially others.

  Returns:
    Total number of trainable parameters for one transformer block.
  """

  embedding_dim = hyperparams["embedding_dim"]

  # Parameters for multi-head attention mechanism.
  mha_parameter_count = parameter_count_attention(hyperparams)

  # Parameters for MLP component.
  mlp_parameter_count = parameter_count_mlp(hyperparams)

  # Parameters for two layer norm components.
  layer_norm_parameter_count = 2 * embedding_dim

  parameter_count = (
      mha_parameter_count + mlp_parameter_count + layer_norm_parameter_count
  )

  return parameter_count


def parameter_count_transformer(hyperparams: Dict[str, int]) -> int:
  """Computes parameters for an entire transformer model consisting of `num_blocks` blocks.

  Args:
    hyperparams: Model hyperparameters needed by the transformer block parameter
      count calculator, including `"embedding_dim"`, `"vocabulary_size",
      `"mlp_dim"`, `"num_blocks"`, and potentially others.

  Returns:
    Total number of trainable parameters for the transformer model.
  """
  num_blocks = hyperparams["num_blocks"]

  # Parameter count of embedding layer.
  embedding_parameter_count = parameter_count_embedding(hyperparams)

  # Parameter count of `num_blocks` transformer blocks.
  transformer_blocks_parameter_count = (
      num_blocks * parameter_count_transformer_block(hyperparams)
  )

  # Parameter count of output_layer.
  output_parameter_count = parameter_count_output_layer(hyperparams)

  parameter_count = (
      embedding_parameter_count
      + transformer_blocks_parameter_count
      + output_parameter_count
  )

  return parameter_count
