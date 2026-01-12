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

"""Functions for testing implementions of transformer submodule counts."""

from typing import Any, Callable, Dict, List
from ai_foundations.feedback.course_4 import counting_parameters
from IPython import display

TEST_MODEL_HYPERPARAMETERS = counting_parameters.TEST_MODEL_HYPERPARAMETERS


def _test_framework(
    learner_solutions: List[Any],
    correct_solutions: List[int],
    near_misses: List[Dict[Any, str]],
):
  """Framework for running tests related to parameter count exercises.

  Args:
    learner_solutions: The solutions provided by the learner.
    correct_solutions: The correct solutions.
    near_misses: The near misses for which specific error messages exist.
  """

  assert len(learner_solutions) == len(correct_solutions)
  assert len(learner_solutions) == len(near_misses)

  correct = True
  for learner_sol, correct_sol, near_miss_d in zip(
      learner_solutions, correct_solutions, near_misses
  ):

    near_miss_d[...] = (
        'You did not fully implement the function. It returns "..."'
    )
    near_miss_d[correct_solutions[0]] = (
        "You seem to have hard-coded"
        " the number of parameters in your implementation. Make sure to use"
        " the values in `hyperparams` to compute the number of parameters."
    )

    if learner_sol != correct_sol:
      if learner_sol in near_miss_d:
        error_message = near_miss_d[learner_sol]
      elif isinstance(learner_sol, str):
        if "is not defined" in learner_sol:
          learner_sol = (
              learner_sol
              + ". Make sure you have executed the"
              " previous cell before running the tests."
          )
        error_message = learner_sol
      else:
        error_message = (
            "Your solution did not return the correct number of parameters."
        )
      correct = False
      break

  if not correct:
    # Remove function defintion and leading indentation.
    display.display(
        display.HTML(
            "<h3>❌ Your implementation is incorrect!</h3>"
            f"<p><strong>Details:</strong><br>{error_message}</p>"
            "<p>Modify the code according to the"
            " error message above and run this cell again.</p>"
        )
    )
  else:
    print("✅ All tests passed. Your implementation is looking good.")


def test_parameter_count_attention(
    parameter_count_attention_fn: Callable[[Dict[str, int]], int],
):
  """Automatic tests for a learner implementation of `parameter_count_attention`.

  Args:
    parameter_count_attention_fn: A callable that takes model hyperparameters as
      a dictionary and returns the number of parameters in the multi-head
      attention component as an integer.
  """

  function_name = getattr(
      parameter_count_attention_fn, "__name__", "parameter_count_attention"
  )

  # No bias terms added.
  def incorrect_implementation1(hyperparams: Dict[str, int]) -> int:
    embedding_dim = hyperparams["embedding_dim"]
    # Parameters for query projection.
    q_parameter_count = (embedding_dim) * embedding_dim
    # Parameters for key projection.
    k_parameter_count = (embedding_dim) * embedding_dim

    # Parameters for value projection.
    v_parameter_count = (embedding_dim) * embedding_dim

    # Parameters for output projection.
    o_parameter_count = (embedding_dim) * embedding_dim

    # Parameters for layer normalization component.
    layer_norm_parameter_count = counting_parameters.parameter_count_layer_norm(
        hyperparams
    )
    parameter_count = (
        q_parameter_count
        + k_parameter_count
        + v_parameter_count
        + o_parameter_count
        + layer_norm_parameter_count
    )
    return parameter_count

  # Not considering parameters for layer normalization.
  def incorrect_implementation2(hyperparams: Dict[str, int]) -> int:
    embedding_dim = hyperparams["embedding_dim"]
    # Parameters for query projection.
    q_parameter_count = (embedding_dim + 1) * embedding_dim
    # Parameters for key projection.
    k_parameter_count = (embedding_dim + 1) * embedding_dim

    # Parameters for value projection.
    v_parameter_count = (embedding_dim + 1) * embedding_dim

    # Parameters for output projection.
    o_parameter_count = (embedding_dim + 1) * embedding_dim

    parameter_count = (
        q_parameter_count
        + k_parameter_count
        + v_parameter_count
        + o_parameter_count
    )
    return parameter_count

  learner_solutions = []
  correct_solutions = []
  near_misses = []
  for hyperparams in TEST_MODEL_HYPERPARAMETERS:
    try:
      learner_solutions.append(parameter_count_attention_fn(hyperparams))
    except Exception as e:  # pylint: disable=broad-except
      learner_solutions.append(f"Error executing `{function_name}`: {e}")

    correct_solutions.append(
        counting_parameters.parameter_count_attention(hyperparams)
    )

    near_misses.append({
        incorrect_implementation1(hyperparams): (
            "You have not added the bias terms for each projection layer. Note"
            " that each projection  using a dense layer includes a bias term "
            " and you need to add 1 to  the first dimension of the projection"
            " matrix to account for that."
        ),
        incorrect_implementation2(hyperparams): (
            "You have not added the parameter count fot the layer normalization"
            " component."
        ),
    })

  _test_framework(
      learner_solutions,
      correct_solutions,
      near_misses,
  )


def test_parameter_count_embedding(
    parameter_count_embedding_fn: Callable[[Dict[str, int]], int],
):
  """Automatic tests for a learner implementation of `parameter_count_embedding`.

  Args:
    parameter_count_embedding_fn: A callable that takes model hyperparameters as
      a dictionary and returns the number of parameters in the embedding
      component as an integer.
  """

  function_name = getattr(
      parameter_count_embedding_fn, "__name__", "parameter_count_embedding"
  )

  # Incorrectly added bias term.
  def incorrect_implementation1(hyperparams: Dict[str, int]) -> int:
    vocabulary_size = hyperparams["vocabulary_size"] + 1
    embedding_dim = hyperparams["embedding_dim"]
    parameter_count = vocabulary_size * embedding_dim
    return parameter_count

  learner_solutions = []
  correct_solutions = []
  near_misses = []
  for hyperparams in TEST_MODEL_HYPERPARAMETERS:
    try:
      learner_solutions.append(parameter_count_embedding_fn(hyperparams))
    except Exception as e:  # pylint: disable=broad-except
      learner_solutions.append(f"Error executing `{function_name}`: {e}")

    correct_solutions.append(
        counting_parameters.parameter_count_embedding(hyperparams)
    )

    near_misses.append({
        incorrect_implementation1(hyperparams): (
            "You have incorrectly added a bias term. Note that the embedding"
            " matrix is a lookup matrix and does not include a bias term."
        )
    })

  _test_framework(
      learner_solutions,
      correct_solutions,
      near_misses,
  )


def test_parameter_count_layer_norm(
    parameter_count_layer_norm_fn: Callable[[Dict[str, int]], int],
):
  """Automatic tests for a learner implementation of `parameter_count_layer_norm`.

  Args:
    parameter_count_layer_norm_fn: A callable that takes model hyperparameters
      as a dictionary and returns the number of parameters in the layer norm
      component as an integer.
  """

  function_name = getattr(
      parameter_count_layer_norm_fn, "__name__", "parameter_count_layer_norm"
  )

  # Only one set of parameters instead of both Beta and Gamma parameters.
  def incorrect_implementation1(hyperparams: Dict[str, int]) -> int:
    embedding_dim = hyperparams["embedding_dim"]
    parameter_count = embedding_dim
    return parameter_count

  learner_solutions = []
  correct_solutions = []
  near_misses = []
  for hyperparams in TEST_MODEL_HYPERPARAMETERS:
    try:
      learner_solutions.append(parameter_count_layer_norm_fn(hyperparams))
    except Exception as e:  # pylint: disable=broad-except
      learner_solutions.append(f"Error executing `{function_name}`: {e}")

    correct_solutions.append(
        counting_parameters.parameter_count_layer_norm(hyperparams)
    )

    near_misses.append({
        incorrect_implementation1(hyperparams): (
            "You've counted only one set of parameters. Make sure to consider"
            " both the scaling parameters `gamma` and the shifting parameters"
            " `beta`."
        )
    })

  _test_framework(
      learner_solutions,
      correct_solutions,
      near_misses,
  )


def test_parameter_count_mlp(
    parameter_count_mlp_fn: Callable[[Dict[str, int]], int],
):
  """Automatic tests for a learner implementation of `parameter_count_mlp`.

  Args:
    parameter_count_mlp_fn: A callable that takes model hyperparameters as a
      dictionary and returns the number of parameters in the MLP component as an
      integer.
  """

  function_name = getattr(
      parameter_count_mlp_fn, "__name__", "parameter_count_mlp"
  )

  # No bias terms added.
  def incorrect_implementation1(hyperparams: Dict[str, int]) -> int:
    embedding_dim = hyperparams["embedding_dim"]
    mlp_dim = hyperparams["mlp_dim"]

    # Parameters for first projection component.
    ffn_parameter_count = (embedding_dim) * mlp_dim
    # Parameters for second projection component.
    output_parameter_count = (mlp_dim) * embedding_dim

    layer_norm_parameter_count = counting_parameters.parameter_count_layer_norm(
        hyperparams
    )
    parameter_count = (
        ffn_parameter_count
        + output_parameter_count
        + layer_norm_parameter_count
    )
    return parameter_count

  learner_solutions = []
  correct_solutions = []
  near_misses = []
  for hyperparams in TEST_MODEL_HYPERPARAMETERS:
    try:
      learner_solutions.append(parameter_count_mlp_fn(hyperparams))
    except Exception as e:  # pylint: disable=broad-except
      learner_solutions.append(f"Error executing `{function_name}`:{e}")

    correct_solutions.append(
        counting_parameters.parameter_count_mlp(hyperparams)
    )

    near_misses.append({
        incorrect_implementation1(hyperparams): (
            "You have not added the bias terms for each projection layer. Note"
            " that each projection using a dense layer includes a bias term"
            " and you need to add 1 to the first dimension of the projection"
            " matrix to account for that."
        )
    })

  _test_framework(
      learner_solutions,
      correct_solutions,
      near_misses,
  )


def test_parameter_count_output_layer(
    parameter_count_output_layer_fn: Callable[[Dict[str, int]], int],
):
  """Automatic tests for a learner implementation of `parameter_count_output_layer`.

  Args:
    parameter_count_output_layer_fn: A callable that takes model hyperparameters
      as a dictionary and returns the number of parameters in the output
      projection layer as an integer.
  """

  function_name = getattr(
      parameter_count_output_layer_fn,
      "__name__",
      "parameter_count_output_layer",
  )

  # No bias terms added.
  def incorrect_implementation1(hyperparams: Dict[str, int]) -> int:
    embedding_dim = hyperparams["embedding_dim"]
    vocabulary_size = hyperparams["vocabulary_size"]

    # Parameters for output projection.
    output_parameter_count = (embedding_dim) * vocabulary_size

    # Only the projection component has parameters,
    # the activation function does not.
    parameter_count = output_parameter_count

    return parameter_count

  learner_solutions = []
  correct_solutions = []
  near_misses = []
  for hyperparams in TEST_MODEL_HYPERPARAMETERS:
    try:
      learner_solutions.append(parameter_count_output_layer_fn(hyperparams))
    except Exception as e:  # pylint: disable=broad-except
      learner_solutions.append(f"Error executing `{function_name}`: {e}")

    correct_solutions.append(
        counting_parameters.parameter_count_output_layer(hyperparams)
    )

    near_misses.append({
        incorrect_implementation1(hyperparams): (
            "You have not added  the bias terms for the projection layer. Note"
            " that each projection using a dense layer includes a bias term and"
            " you need  to add 1 to the first dimension of the projection"
            " matrix to account for that."
        ),
    })

  _test_framework(
      learner_solutions,
      correct_solutions,
      near_misses,
  )


def test_parameter_count_transformer_block(
    parameter_count_transformer_block_fn: Callable[[Dict[str, int]], int],
):
  """Automatic tests for a learner implementation of `parameter_count_transformer_block`.

  Args:
    parameter_count_transformer_block_fn: A callable that takes model
      hyperparameters as a dictionary and returns the number of parameters in a
      single Transformer block as an integer.
  """

  function_name = getattr(
      parameter_count_transformer_block_fn,
      "__name__",
      "parameter_count_transformer_block",
  )

  # No multi-head attention.
  def incorrect_implementation1(hyperparams: Dict[str, int]) -> int:
    # Parameters for MLP component.
    mlp_parameter_count = counting_parameters.parameter_count_mlp(hyperparams)
    parameter_count = mlp_parameter_count
    return parameter_count

  # No MLP.
  def incorrect_implementation2(hyperparams: Dict[str, int]) -> int:
    # Parameters for multi-head attention mechanism.
    mha_parameter_count = counting_parameters.parameter_count_attention(
        hyperparams
    )

    parameter_count = mha_parameter_count
    return parameter_count

  # No Layer-Norm.
  def incorrect_implementation3(hyperparams: Dict[str, int]) -> int:

    # Parameters for multi-head attention mechanism.
    mha_parameter_count = counting_parameters.parameter_count_attention(
        hyperparams
    )
    # Parameters for MLP component.
    mlp_parameter_count = counting_parameters.parameter_count_mlp(hyperparams)

    parameter_count = mha_parameter_count + mlp_parameter_count
    return parameter_count

  learner_solutions = []
  correct_solutions = []
  near_misses = []
  for hyperparams in TEST_MODEL_HYPERPARAMETERS:
    try:
      learner_solutions.append(
          parameter_count_transformer_block_fn(hyperparams)
      )
    except Exception as e:  # pylint: disable=broad-except
      learner_solutions.append(f"Error executing `{function_name}`: {e}")

    correct_solutions.append(
        counting_parameters.parameter_count_transformer_block(hyperparams)
    )

    near_misses.append({
        incorrect_implementation1(hyperparams): (
            "You have not added the parameter count for the multi-head"
            " attention component."
        ),
        incorrect_implementation2(
            hyperparams
        ): "You have not added the parameter count fot the MLP component.",
        incorrect_implementation3(hyperparams): (
            "You have not added the parameter count fot the Layer normalization"
            " components."
        ),
    })

  _test_framework(
      learner_solutions,
      correct_solutions,
      near_misses,
  )


def test_parameter_count_transformer(
    parameter_count_transformer_fn: Callable[[Dict[str, int]], int],
):
  """Automatic tests for a learner implementation of `parameter_count_transformer`.

  Args:
    parameter_count_transformer_fn: A callable that takes model hyperparameters
      as a dictionary and returns the total number of parameters in the entire
      transformer model as an integer.
  """

  function_name = getattr(
      parameter_count_transformer_fn, "__name__", "parameter_count_transformer"
  )

  # No embedding.
  def incorrect_implementation1(hyperparams: Dict[str, int]) -> int:
    num_blocks = hyperparams["num_blocks"]

    # Parameter count of `num_blocks` transformer blocks.
    transformer_blocks_parameter_count = (
        num_blocks
        * counting_parameters.parameter_count_transformer_block(hyperparams)
    )

    # Parameter count of output_layer.
    output_parameter_count = counting_parameters.parameter_count_output_layer(
        hyperparams
    )

    parameter_count = (
        transformer_blocks_parameter_count + output_parameter_count
    )
    return parameter_count

  # No output layer.
  def incorrect_implementation2(hyperparams: Dict[str, int]) -> int:
    num_blocks = hyperparams["num_blocks"]

    # Parameter count of embedding layer.
    embedding_parameter_count = counting_parameters.parameter_count_embedding(
        hyperparams
    )

    # Parameter count of `num_blocks` transformer blocks.
    transformer_blocks_parameter_count = (
        num_blocks
        * counting_parameters.parameter_count_transformer_block(hyperparams)
    )
    parameter_count = (
        embedding_parameter_count + transformer_blocks_parameter_count
    )

    return parameter_count

  # Only one transformer block.
  def incorrect_implementation3(hyperparams: Dict[str, int]) -> int:

    # Parameter count of embedding layer.
    embedding_parameter_count = counting_parameters.parameter_count_embedding(
        hyperparams
    )

    # Parameter count of 1 transformer blocks.
    transformer_blocks_parameter_count = (
        1 * counting_parameters.parameter_count_transformer_block(hyperparams)
    )
    # Parameter count of output_layer.
    output_parameter_count = counting_parameters.parameter_count_output_layer(
        hyperparams
    )

    parameter_count = (
        embedding_parameter_count
        + transformer_blocks_parameter_count
        + output_parameter_count
    )
    return parameter_count

  learner_solutions = []
  correct_solutions = []
  near_misses = []
  for hyperparams in TEST_MODEL_HYPERPARAMETERS:
    try:
      learner_solutions.append(parameter_count_transformer_fn(hyperparams))
    except Exception as e:  # pylint: disable=broad-except
      learner_solutions.append(f"Error executing `{function_name}`: {e}")

    correct_solutions.append(
        counting_parameters.parameter_count_transformer(hyperparams)
    )

    near_misses.append({
        incorrect_implementation1(
            hyperparams
        ): "You have not added the parameter count for the embedding layer.",
        incorrect_implementation2(
            hyperparams
        ): "You have not added the parameter count for the output layer.",
        incorrect_implementation3(hyperparams): (
            "You have only added the parameter count for one transformer block."
            " Make sure to account for all `n_blocks` transformer layers."
        ),
    })

  _test_framework(
      learner_solutions,
      correct_solutions,
      near_misses,
  )
