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

"""Utitlity functions to test a learner's implementation of attention."""

from typing import Any, Callable, Dict, Tuple
from ai_foundations import attention
from ai_foundations.feedback.utils import render_feedback
from gemma import modules as _modules
import jax
import jax.numpy as jnp

K_MASK = _modules.K_MASK


def test_apply_attention_mask(
    apply_attention_mask: Callable[[jax.Array], jax.Array],
    qkv_dict: Dict[str, Any],
) -> None:
  """Main test entrypoint for apply_attention_mask.

  Args:
    apply_attention_mask: Student's function mapping raw logits to masked
      logits.
    qkv_dict: Dictionary containing query/key/value projection weights.
  """

  def _reference_implementation(logits_raw: jax.Array) -> jax.Array:
    """Computes reference masked logits using a causal triangular mask.

    Args:
      logits_raw: Unmasked attention logits with shape (heads, n_tokens,
        n_tokens).

    Returns:
      Masked attention logits with shape (heads, n_tokens, n_tokens).
    """
    _, n_tokens, _ = logits_raw.shape

    attention_mask = jnp.tri(n_tokens)
    logits_masked = jnp.where(attention_mask, logits_raw, K_MASK)

    return logits_masked

  def _test_apply_attention_mask(
      apply_attention_mask: Callable[[jax.Array], jax.Array],
      qkv_dict: Dict[str, Any],
  ):
    """Runs assertions comparing student's masked logits to reference.

    Args:
      apply_attention_mask: Student's function mapping raw logits to masked
        logits.
      qkv_dict: Dictionary containing query/key/value projection weights.

    Raises:
      RuntimeError: If the student's function returns Ellipsis.
      AssertionError: If the student's outputs don't match the reference.
    """
    # Test the first 10 layers.
    for layer in range(10):
      (query_proj_list, key_proj_list, _) = attention.get_qkv_matrices(
          qkv_dict, layer, head=None
      )

      # Stack the query, key, and value projection matrices.
      query_proj = jnp.stack(query_proj_list, axis=0)
      key_proj = jnp.stack(key_proj_list, axis=0)

      key_transposed = jnp.transpose(key_proj, (0, 2, 1))
      _, _, dim_k = key_proj.shape
      logits_raw = query_proj @ key_transposed / jnp.sqrt(dim_k)

      logits_ref = _reference_implementation(logits_raw)
      logits_stu = apply_attention_mask(logits_raw)
      if logits_stu == ...:
        raise RuntimeError(
            "Your implementation is incorrect!",
            "You are not computing the masked logits, the function returned"
            ' "...".',
        )

      assert jnp.allclose(logits_ref, logits_stu), (
          "Your implementation is incorrect!",
          (
              "Your implementation of the masked logits is incorrect. Check"
              " your implementation and try again."
          ),
      )

  try:
    _test_apply_attention_mask(apply_attention_mask, qkv_dict)

  except (
      AssertionError,
      KeyError,
      NameError,
      NotImplementedError,
      RuntimeError,
      SyntaxError,
      ValueError,
  ) as e:
    render_feedback(e)
  else:
    print("✅ All tests passed. Your implementation is looking good.")


def test_compute_attention_mask(
    compute_attention_mask: Callable[[int], jax.Array],
):
  """Main test function for `compute_attention_mask`.

  Args:
    compute_attention_mask: The student's implementation to test.
  """

  def _reference_implementation(n_tokens: int) -> jax.Array:
    """Creates a reference attention mask using triangular matrix.

    Args:
        n_tokens: Number of tokens in the sequence.

    Returns:
        A JAX array representing the attention mask as a triangular matrix.
    """
    attention_mask = jnp.tri(n_tokens)
    return attention_mask

  def _test_compute_attention_mask(
      compute_attention_mask: Callable[[int], jax.Array],
  ) -> None:
    """Tests the student's implementation of compute_attention_mask against reference.

    This function runs tests on the student's implementation by comparing it
    with the reference implementation for various sequence lengths.

    Args:
      compute_attention_mask: The student's implementation function to test.

    Raises:
      RuntimeError: If the student's function returns "..." or produces
      incorrect results.
      AssertionError: If the student's implementation doesn't match the
      reference.
    """

    for n_tokens in range(1, 20, 3):
      mask_ref = _reference_implementation(n_tokens)
      mask_stu = compute_attention_mask(n_tokens)

      if mask_stu == ...:
        raise RuntimeError(
            "Your implementation is incorrect!",
            "You are not computing the attention mask, the"
            ' function returned "...".',
        )

      assert jnp.allclose(mask_ref, mask_stu), (
          "Your implementation is incorrect!",
          (
              "Your implementation of the attention mask is incorrect."
              " Check your implementation and try again."
          ),
      )

  try:
    _test_compute_attention_mask(compute_attention_mask)

  except (
      AssertionError,
      KeyError,
      NameError,
      NotImplementedError,
      RuntimeError,
      SyntaxError,
      ValueError,
  ) as e:
    render_feedback(e)
  else:
    print("✅ All tests passed. Your implementation is looking good.")


def test_compute_attention_output(
    compute_attention_output: Callable[[jax.Array, jax.Array], jax.Array],
    qkv_dict: Dict[str, Any],
):
  """Main test entrypoint for compute_attention_output with feedback.

  Args:
    compute_attention_output: Student's function to compute Y from (alpha, V).
    qkv_dict: Dictionary containing query/key/value projection weights.
  """

  def _reference_implementation(
      alpha: jax.Array, value_proj: jax.Array
  ) -> jax.Array:
    """Computes reference attention output Y.

    Args:
      alpha: Attention weights of shape (heads, n_tokens, n_tokens).
      value_proj: Value projections of shape (heads, n_tokens, dim_v).

    Returns:
      Attention output Y of shape (heads, n_tokens, dim_v).
    """
    Y = alpha @ value_proj  # pylint: disable=invalid-name

    return Y

  def _test_compute_attention_output(
      compute_attention_output: Callable[[jax.Array, jax.Array], jax.Array],
      qkv_dict: Dict[str, Any],
  ) -> None:
    """Runs assertions comparing student's attention output to reference.

    Args:
      compute_attention_output: Student's function to compute Y from (alpha, V).
      qkv_dict: Dictionary containing query/key/value projection weights.

    Raises:
      NotImplementedError: If the student's function returns Ellipsis.
      AssertionError: If the student's outputs don't match the reference.
    """
    # Test the first 10 layers.
    for layer in range(10):
      (query_proj_list, key_proj_list, value_proj_list) = (
          attention.get_qkv_matrices(qkv_dict, layer, head=None)
      )

      # Stack the query, key, and value projection matrices.
      query_proj = jnp.stack(query_proj_list, axis=0)
      key_proj = jnp.stack(key_proj_list, axis=0)
      value_proj = jnp.stack(value_proj_list, axis=0)

      key_transposed = jnp.transpose(key_proj, (0, 2, 1))
      _, n_tokens, dim_k = key_proj.shape
      logits_raw = query_proj @ key_transposed / jnp.sqrt(dim_k)

      # Build a causal attention mask locally to avoid extra dependencies.
      attention_mask = jnp.tri(n_tokens)
      logits_masked = jnp.where(attention_mask, logits_raw, K_MASK)

      alpha = jax.nn.softmax(logits_masked)

      out_ref = _reference_implementation(alpha, value_proj)
      out_stu = compute_attention_output(alpha, value_proj)

      if out_stu == ...:
        raise NotImplementedError(
            "Your implementation is incorrect!",
            "You are not computing the attention output Y,"
            ' the function returned "...".',
        )

      assert jnp.allclose(out_ref, out_stu), (
          "Your implementation is incorrect!",
          (
              "Your implementation of the attention output Y is incorrect."
              "  Check your implementation and try again."
          ),
      )

  try:
    _test_compute_attention_output(compute_attention_output, qkv_dict)

  except (
      AssertionError,
      KeyError,
      NameError,
      NotImplementedError,
      RuntimeError,
      SyntaxError,
      ValueError,
  ) as e:
    render_feedback(e)
  else:
    print("✅ All tests passed. Your implementation is looking good.")


def test_compute_attention_weights(
    compute_attention_weights: Callable[[jax.Array], jax.Array],
    qkv_dict: Dict[str, Any],
):
  """Main test entrypoint for compute_attention_weights.

  Args:
      compute_attention_weights: Student's function to compute alpha from masked
        logits.
      qkv_dict: Dictionary containing query/key/value projection weights.
  """

  def _reference_implementation(logits_masked: jax.Array) -> jax.Array:
    """Computes reference attention weights via softmax.

    Args:
      logits_masked: Masked attention logits of shape (heads, n_tokens,
        n_tokens).

    Returns:
      Attention weights alpha of shape (heads, n_tokens, n_tokens).
    """
    # Apply the softmax.
    alpha = jax.nn.softmax(logits_masked)

    return alpha

  def _test_compute_attention_weights(
      compute_attention_weights: Callable[[jax.Array], jax.Array],
      qkv_dict: Dict[str, Any],
  ):
    """Runs assertions comparing student's attention weights to reference.

    Args:
      compute_attention_weights: Student's function to compute alpha from masked
        logits.
      qkv_dict: Dictionary containing query/key/value projection weights.

    Raises:
      NotImplementedError: If the student's function returns Ellipsis.
      AssertionError: If the student's outputs don't match the reference.
    """
    # Test the first 10 layers.
    for layer in range(10):
      (query_proj_list, key_proj_list, _) = attention.get_qkv_matrices(
          qkv_dict, layer, head=None
      )

      # Stack the query, key, and value projection matrices.
      query_proj = jnp.stack(query_proj_list, axis=0)
      key_proj = jnp.stack(key_proj_list, axis=0)

      key_transposed = jnp.transpose(key_proj, (0, 2, 1))
      _, n_tokens, dim_k = key_proj.shape
      logits_raw = query_proj @ key_transposed / jnp.sqrt(dim_k)

      # Build a causal attention mask locally to avoid extra dependencies.
      attention_mask = jnp.tri(n_tokens)
      logits_masked = jnp.where(attention_mask, logits_raw, K_MASK)

      alpha_ref = _reference_implementation(logits_masked)
      alpha_stu = compute_attention_weights(logits_masked)

      if alpha_stu == ...:
        raise NotImplementedError(
            "Your implementation is incorrect!",
            "You are not computing the attention weights alpha,"
            ' the function returned "...".',
        )

      assert jnp.allclose(alpha_ref, alpha_stu), (
          "Your implementation is incorrect!",
          (
              "Your implementation of the attention weights alpha is"
              "  incorrect. Check your implementation and try again."
          ),
      )

  try:
    _test_compute_attention_weights(compute_attention_weights, qkv_dict)

  except (
      AssertionError,
      KeyError,
      NameError,
      NotImplementedError,
      RuntimeError,
      SyntaxError,
      ValueError,
  ) as e:
    render_feedback(e)
  else:
    print("✅ All tests passed. Your implementation is looking good.")


def test_compute_attention(
    compute_attention: Callable[
        [Dict[str, Any], int], Tuple[jax.Array, jax.Array, jax.Array]
    ],
    qkv_dict: Dict[str, Any],
):
  """Main test entrypoint for compute_attention.

  Args:
    compute_attention: Student's function mapping (qkv_dict, layer) to (Y,
      alpha, logits).
    qkv_dict: Dictionary containing query/key/value projection weights.
  """

  def _reference_implementation(
      qkv_dict: Dict[str, Any], layer: int
  ) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Computes reference attention outputs (Y, alpha, logits) per layer.

    This follows the original code pattern using QK^T/sqrt(d_k) without masking.

    Args:
      qkv_dict: Dictionary containing layer-wise Q, K, and V projections.
      layer: Layer index to test.

    Returns:
      Y: Attention output of shape (heads, n_tokens, dim_v).
      alpha: Attention weights of shape (heads, n_tokens, n_tokens).
      logits: Raw attention logits of shape (heads, n_tokens, n_tokens).
    """
    query_proj, key_proj, value_proj = attention.get_qkv_matrices(
        qkv_dict, layer
    )

    # Extract d_k.
    _, dim_key = key_proj.shape  # type: ignore

    # Compute the logits = QK.T/sqrt(d_k)
    logits = (query_proj @ key_proj.T) / jnp.sqrt(dim_key)  # type: ignore

    # Compute the attention weights alpha.
    alpha = jax.nn.softmax(logits)

    # Compute the output Y = alpha @ V.
    Y = alpha @ value_proj  # pylint: disable=invalid-name

    return Y, alpha, logits

  def _test_compute_attention(
      compute_attention: Callable[
          [Dict[str, Any], int], Tuple[jax.Array, jax.Array, jax.Array]
      ],
      qkv_dict: Dict[str, Any],
  ):
    """Runs assertions comparing student's attention tuple to reference.

    Args:
        compute_attention: Student's function mapping (qkv_dict, layer) to (Y,
          alpha, logits).
        qkv_dict: Dictionary containing query/key/value projection weights.

    Raises:
      RuntimeError: If the student's function returns ... for any output.
      AssertionError: If the student's outputs don't match the reference.
    """
    # Test the first 10 layers.
    for layer in range(10):
      Y_ref, alpha_ref, logits_ref = _reference_implementation(qkv_dict, layer)  # pylint: disable=invalid-name
      Y_stu, alpha_stu, logits_stu = compute_attention(qkv_dict, layer)  # pylint: disable=invalid-name

      if logits_stu == ...:
        raise RuntimeError(
            "Your implementation is incorrect!",
            'You are not computing the logits, the function returned "...".',
        )
      if alpha_stu == ...:
        raise RuntimeError(
            "Your implementation is incorrect!",
            "You are not computing the attention weights alpha, the function"
            ' returned "...".',
        )
      if Y_stu == ...:
        raise RuntimeError(
            "Your implementation is incorrect!",
            'You are not computing the output Y, the function returned "...".',
        )

      assert jnp.allclose(logits_ref, logits_stu), (
          "Your implementation is incorrect!",
          (
              "Your implementation of the logits is incorrect. Check your"
              " implementation and try again."
          ),
      )
      assert jnp.allclose(alpha_ref, alpha_stu), (
          "Your implementation is incorrect!",
          (
              "Your implementation of the attention weight computation is"
              " incorrect. Check your implementation and try again."
          ),
      )
      assert jnp.allclose(Y_ref, Y_stu), (
          "Your implementation is incorrect!",
          (
              "Your implementation of the output Y is"
              " incorrect. Check your implementation and try again."
          ),
      )

  try:
    _test_compute_attention(compute_attention, qkv_dict)

  except (
      AssertionError,
      KeyError,
      NameError,
      NotImplementedError,
      RuntimeError,
      SyntaxError,
      ValueError,
  ) as e:
    render_feedback(e)
  else:
    print("✅ All tests passed. Your implementation is looking good.")


def test_compute_raw_logits(
    compute_raw_logits: Callable[[jax.Array, jax.Array], jax.Array],
    qkv_dict: Dict[str, Any],
):
  """Main test entrypoint for compute_raw_logits.

  Args:
    compute_raw_logits: Student's implementation to compute raw logits.
    qkv_dict: Dictionary containing query/key/value projection weights.
  """

  def _reference_implementation(
      query_proj: jax.Array, key_proj: jax.Array
  ) -> jax.Array:
    """Computes reference raw attention logits.

    Args:
      query_proj: Query projections with shape (heads, n_tokens, dim_k).
      key_proj: Key projections with shape (heads, n_tokens, dim_k).

    Returns:
      Raw attention logits with shape (heads, n_tokens, n_tokens).
    """
    # Transpose the last two dimensions of the key matrix for matrix
    # multiplication. The shape changes from (heads, n_tokens, dim_k) to
    # (heads, dim_k, n_tokens).
    key_transposed = jnp.transpose(key_proj, (0, 2, 1))

    _, _, dim_k = key_proj.shape
    logits = query_proj @ key_transposed / jnp.sqrt(dim_k)

    return logits

  def _test_compute_raw_logits(
      compute_raw_logits: Callable[[jax.Array, jax.Array], jax.Array],
      qkv_dict: Dict[str, Any],
  ):
    """Runs assertions comparing student's logits to reference implementation.

    Args:
      compute_raw_logits: Student's implementation to compute raw logits.
      qkv_dict: Dictionary containing query/key/value projection weights.

    Raises:
      RuntimeError: If the student's function returns Ellipsis.
      AssertionError: If the student's outputs don't match the reference.
    """
    # Test the first 10 layers.
    for layer in range(10):
      (query_proj_list, key_proj_list, _) = attention.get_qkv_matrices(
          qkv_dict, layer, head=None
      )

      # Stack the query, key, and value projection matrices.
      query_proj = jnp.stack(query_proj_list, axis=0)
      key_proj = jnp.stack(key_proj_list, axis=0)

      logits_ref = _reference_implementation(query_proj, key_proj)
      logits_stu = compute_raw_logits(query_proj, key_proj)
      if logits_stu == ...:
        raise RuntimeError(
            "Your implementation is incorrect!",
            "You are not computing the raw logits, the function returned"
            ' "...".',
        )

      assert jnp.allclose(logits_ref, logits_stu), (
          "Your implementation is incorrect!",
          (
              "Your implementation of the raw logits is"
              "  incorrect. Check your implementation and try again."
          ),
      )

  try:
    _test_compute_raw_logits(compute_raw_logits, qkv_dict)

  except (
      AssertionError,
      KeyError,
      NameError,
      NotImplementedError,
      RuntimeError,
      SyntaxError,
      ValueError,
  ) as e:
    render_feedback(e)
  else:
    print("✅ All tests passed. Your implementation is looking good.")


def test_stack_matrices(
    stack_matrices: Callable[
        [Dict[str, Any], int], Tuple[jax.Array, jax.Array, jax.Array]
    ],
    qkv_dict: Dict[str, Any],
):
  """Main test entrypoint for stack_matrices.

  Args:
    stack_matrices: Student's implementation to stack Q, K, V matrices.
    qkv_dict: Dictionary containing query/key/value projection weights.
  """

  def _reference_implementation(
      qkv_dict: Dict[str, Any], layer: int
  ) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Builds reference stacked Q, K, V tensors for a given layer.

    Args:
      qkv_dict: Dictionary containing query/key/value projection weights.
      layer: Layer index to extract matrices from.

    Returns:
      Tuple of (query_proj, key_proj, value_proj) stacked along head axis.
    """
    (query_proj_list, key_proj_list, value_proj_list) = (
        attention.get_qkv_matrices(qkv_dict, layer, head=None)
    )

    # Stack the query, key, and value projection matrices.
    query_proj = jnp.stack(query_proj_list, axis=0)
    key_proj = jnp.stack(key_proj_list, axis=0)
    value_proj = jnp.stack(value_proj_list, axis=0)

    return query_proj, key_proj, value_proj

  def _test_stack_matrices(
      stack_matrices: Callable[
          [Dict[str, Any], int], Tuple[jax.Array, jax.Array, jax.Array]
      ],
      qkv_dict: Dict[str, Any],
  ):
    """Runs assertions comparing student's outputs to the reference stacking.

    Args:
      stack_matrices: Student's implementation to stack Q, K, V matrices.
      qkv_dict: Dictionary containing query/key/value projection weights.

    Raises:
      RuntimeError: If any of the returned tensors are Ellipsis.
      AssertionError: If student's outputs don't match the reference.
    """
    # Test the first 10 layers.
    for layer in range(10):
      (query_proj_ref, key_proj_ref, value_proj_ref) = (
          _reference_implementation(qkv_dict, layer)
      )
      (query_proj_stu, key_proj_stu, value_proj_stu) = stack_matrices(
          qkv_dict, layer
      )
      if query_proj_stu == ...:
        raise RuntimeError(
            "Your implementation is incorrect!",
            "You are not computing the query projections,"
            ' the function returned "...".',
        )
      if key_proj_stu == ...:
        raise RuntimeError(
            "Your implementation is incorrect!",
            "You are not computing the key projections"
            ' alpha, the function returned "...".',
        )
      if value_proj_stu == ...:
        raise RuntimeError(
            "Your implementation is incorrect!",
            "You are not computing key value projections,"
            ' the function returned "...".',
        )

      assert jnp.allclose(query_proj_ref, query_proj_stu), (
          "Your implementation is incorrect!",
          (
              "Your implementation of the stacked query projections is"
              "  incorrect. Check your implementation and try again."
          ),
      )
      assert jnp.allclose(key_proj_ref, key_proj_stu), (
          "Your implementation is incorrect!",
          (
              "Your implementation of the stacked key projections is"
              "  incorrect. Check your implementation and try again."
          ),
      )
      assert jnp.allclose(value_proj_ref, value_proj_stu), (
          "Your implementation is incorrect!",
          (
              "Your implementation of the stacked value projections is"
              "  incorrect. Check your implementation and try again."
          ),
      )

  try:
    _test_stack_matrices(stack_matrices, qkv_dict)

  except (
      AssertionError,
      KeyError,
      NameError,
      NotImplementedError,
      RuntimeError,
      SyntaxError,
      ValueError,
  ) as e:
    render_feedback(e)
  else:
    print("✅ All tests passed. Your implementation is looking good.")
