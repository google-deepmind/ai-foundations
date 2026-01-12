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

"""Feedback functions for the lab on MLPs in Course 3."""

from typing import Callable, List

from ai_foundations.feedback import utils
import keras


def test_construct_operations(
    construct_operations: Callable[[List[int], int], List[keras.Layer]],
):
  """Tests a student's implementation of a function that builds MLP layers.

  This function validates that the student's `construct_operations` function
  correctly generates a list of Keras layers for a Multi-Layer Perceptron (MLP)
  based on specified hidden layer dimensions and the number of output classes.

  It runs two primary test cases:
  1.  No hidden layers: A simple logistic regression-style model. It
      checks if the function produces exactly two layers: a `Dense` layer for
      the weighted sum and a `Softmax` layer for the output probabilities.
      It also verifies that the `Dense` layer has the correct output dimension
      (equal to `n_classes`).
  2.  Multiple hidden layers: A deeper MLP. It checks if the function
      correctly interleaves `Dense` and `ReLU` activation layers for each
      specified hidden dimension. It then verifies that the final output
      consists of a `Dense` layer and a `Softmax` layer. The output dimensions
      of all `Dense` layers are also checked for correctness.

  The function provides user-friendly feedback if any of the checks fail,
  guiding the student toward the correct implementation.

  Args:
    construct_operations: A function handle to the student's implementation.
      This function is expected to take two arguments:
      - hidden_dims: A list where each integer specifies the number of neurons
        in a hidden layer. An empty list signifies no hidden layers.
      - n_classes: The number of neurons in the final output layer,
        corresponding to the number of classes for classification.
      It should return a list of Keras `Layer` objects.
  """

  try:
    one_layer_nn_layers = construct_operations([], 3)
    if len(one_layer_nn_layers) != 2:
      if len(one_layer_nn_layers) > 2:
        raise AssertionError(
            "You seem to have implemented too many operations. Make sure that"
            " <code>running construct_operations([], 3)</code> only generates"
            " two operations (the weighted sums and the SoftMax)."
        )
      elif not one_layer_nn_layers:
        raise NotImplementedError(
            "You seem to not have implemented the output layer. Make sure that"
            " the final two operations consist of a weighted sum and the"
            " SoftMax operation."
        )
      elif not isinstance(one_layer_nn_layers[0], keras.layers.Dense):
        raise NotImplementedError(
            "You seem to not have implemented the weighted sum as part of the"
            " output. Make sure that the final two operations consist of a"
            " weighted sum and the SoftMax operation."
        )
      elif not isinstance(one_layer_nn_layers[1], keras.layers.Softmax):
        raise NotImplementedError(
            "You seem to not have implemented the SoftMax operation as part of"
            " the output. Make sure that the final two operations consist of a"
            " weighted sum and the SoftMax operation."
        )
    else:
      dense_layer_output_dim = one_layer_nn_layers[0].compute_output_shape((5,))

      assert dense_layer_output_dim[0] == 3, (
          "The output layer does not have the correct dimension. Make sure that"
          " the output layer's dimension is set to <code>n_classes</code>."
      )

    three_layer_nn_layers = construct_operations([5, 10], 7)
    if len(three_layer_nn_layers) != 6:
      if len(three_layer_nn_layers) > 6:
        raise AssertionError(
            "You seem to have implemented too many operations. Make sure that"
            " running <code>construct_operations([5, 10], 7)</code> only"
            " generates six oeprations (a weighted sum, a ReLU, another"
            " weighted sum, another ReLU, and weighted sum and the SoftMax"
            " operation for the output)."
        )
      elif len(three_layer_nn_layers) == 2:
        raise NotImplementedError(
            "You seem to not have implemented the hidden layers. Make sure that"
            " when running <code>construct_operations([5, 10], 7)</code>, the"
            " first four layers are a weighted sum, a ReLU, another weighted"
            " sum, and another ReLU."
        )
      else:
        raise AssertionError(
            "Incorrect number of operations. Make sure that running"
            " <code>construct_operations([5, 10], 7)</code> generates six"
            " layers (a weighted sum, a ReLU, another weighted sum, another"
            " ReLU, and the weighted sum and the SoftMax operation for the"
            " output)."
        )
    else:
      if not isinstance(
          three_layer_nn_layers[0], keras.layers.Dense
      ) or not isinstance(three_layer_nn_layers[2], keras.layers.Dense):
        raise NotImplementedError(
            "You seem to not have implemented the dense operations correctly."
            " Make sure that when running <code>construct_operations([5, 10],"
            " 7)</code>, the first four operations are a weighted sum, a ReLU,"
            " another weighted sum, and another ReLU."
        )
      elif not isinstance(
          three_layer_nn_layers[1], keras.layers.ReLU
      ) or not isinstance(three_layer_nn_layers[3], keras.layers.ReLU):
        raise NotImplementedError(
            "You seem to not have implemented the ReLU operations correctly. "
            " Make sure that when running <code>construct_operations([5, 10],"
            " 7)</code>, the first four operations are a weighted sum, a ReLU,"
            " another weighted sum, and another ReLU."
        )

      dense_layer_output_dim = three_layer_nn_layers[0].compute_output_shape((
          5,
      ))
      assert dense_layer_output_dim[0] == 5, (
          "The first dense layer does not have the correct dimension. Make sure"
          " that the first hidden layer's dimension is set to the first element"
          " in <code>hidden_dims</code>."
      )

      dense_layer_output_dim = three_layer_nn_layers[2].compute_output_shape((
          5,
      ))
      assert dense_layer_output_dim[0] == 10, (
          "The second dense layer does not have the correct dimension. Make"
          " sure that the second hidden layer's dimension is set to the second"
          " element in <code>hidden_dims</code>."
      )

      dense_layer_output_dim = three_layer_nn_layers[4].compute_output_shape((
          5,
      ))
      assert dense_layer_output_dim[0] == 7, (
          "The output layer does not have the correct dimension. Make sure that"
          " the output layer's dimension is set to <code>n_classes</code>."
      )

      print("✅ All tests passed. Your implementation is looking good.")

  except (
      AssertionError,
      RuntimeError,
      KeyError,
      NotImplementedError,
      SyntaxError,
      ReferenceError,
      NameError,
  ) as e:
    utils.render_feedback(e)
