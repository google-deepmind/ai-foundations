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

"""A class that implements a Byte Pair Encoding Tokenizer for Arabic text.

This class is specifically designed and factorized for the AI Research
Foundations curriculum's Challenge Lab 2.
"""

from collections.abc import Sequence
import string

from ai_foundations.tokenization.bpe_tokenizer import BPEWordTokenizer
import tqdm


class ArabicBPETokenizer(BPEWordTokenizer):
  """A Byte Pair Encoding (BPE) based subword tokenizer.

  This class supports encoding and decoding text to subword tokens using BPE.

  Attributes:
    vocabulary: List of subword tokens including special tokens.
    vocabulary_size: Total number of tokens in vocabulary.
    token_to_index: Mapping from tokens to indices.
    index_to_token: Mapping from indices to tokens.
    pad_token_id: Index of the padding token.
    unknown_token_id: Index of the unknown token.
    tokenized_corpus: Cached tokenized corpus after BPE training.
  """

  def __init__(
      self,
      texts: Sequence[str] | str,
      num_merges: int = 100,
  ):
    """Initializes the ArabicBPETokenizer.

    Extracts the unique tokens from the text corpus and learns the BPE merges.

    Args:
      texts: A sequence of strings or a string representing the text corpus.
      num_merges: Defines how many rounds of merges should be performed when
        learning the BPE merges.
    """
    # Create a parent object with minimal vocabulary.
    super().__init__(None, vocabulary=[self.PAD_TOKEN, self.UNKNOWN_TOKEN])

    # Normalize text to list of strings.
    if isinstance(texts, str):
      texts = [texts]

    # Learn BPE merges and derive vocabulary from tokenized corpus.
    self.merges = []
    self.vocabulary = None
    tokenized_corpus, vocabulary_set = self._learn_bpe(texts, num_merges)
    self.tokenized_corpus = tokenized_corpus
    self._update_vocabulary(vocabulary_set)

  def continue_bpe(
      self,
      corpus: Sequence[str] | str,
      num_merges: int,
  ) -> None:
    """Continues BPE training with additional merges on new corpus data.

    Extends the existing BPE model by learning additional merges from new
    training data and updating the vocabulary accordingly.

    Args:
      corpus: Sequence of input strings or a single string for continued
        training.
      num_merges: Number of additional merge operations to perform.
    """
    # Normalize text to list of strings.
    if isinstance(corpus, str):
      corpus = [corpus]

    tokenized_corpus, vocabulary_set = self._learn_bpe(corpus, num_merges)
    self.tokenized_corpus = tokenized_corpus
    self._update_vocabulary(vocabulary_set)

  def _learn_bpe(
      self,
      corpus: Sequence[str],
      num_merges: int,
  ) -> tuple[list[list[list[str]]], set[str]]:
    """Learns BPE merges from a corpus of texts.

    Args:
      corpus: Sequence of input texts.
      num_merges: Number of merge operations to perform.

    Returns:
      Tuple of (tokenized_corpus, vocabulary_set) where tokenized_corpus is a
      list of paragraphs (each tokenized as a list of word token lists) and
      vocabulary_set is the set of subword tokens after performing all merges.
    """
    tokenized_corpus, vocabulary_set = self._tokenize_corpus(corpus)

    # Unset vocabulary and tokenized_corpus attributes to force refresh after
    # merging operation.
    self.vocabulary = None
    self.tokenized_corpus = None

    for _ in (pbar := tqdm.tqdm(range(num_merges), unit="merges")):
      next_pair_to_merge = self._find_next_pair_to_merge(tokenized_corpus)

      if next_pair_to_merge is None:
        break

      self.merges.append(next_pair_to_merge)
      # Apply a merge operation to all occurrences of `next_pair_to_merge` in
      # the tokenized corpus.
      tokenized_corpus = self._apply_merge_to_tokenized_corpus(
          tokenized_corpus=tokenized_corpus,
          next_pair_to_merge=next_pair_to_merge,
      )
      vocabulary_set.add(next_pair_to_merge[0] + next_pair_to_merge[1])

      pbar.set_postfix(num_merges=f"{len(self.merges):,}")

    return tokenized_corpus, vocabulary_set

  def _apply_merge_to_tokenized_corpus(
      self,
      tokenized_corpus: list[list[list[str]]],
      next_pair_to_merge: tuple[str, str],
  ) -> list[list[list[str]]]:
    """Applies a merge operation to all occurrences in the tokenized corpus.

    Args:
      tokenized_corpus: List of paragraphs containing word token lists.
      next_pair_to_merge: Tuple of two tokens to merge throughout the corpus.

    Returns:
      Updated tokenized corpus with the specified pair merged.
    """
    # Apply merge to each token in each paragraph.
    new_tokenized_corpus = []
    for para_tokens in tokenized_corpus:
      new_para_tokens = []
      for word_tokens in para_tokens:
        new_para_tokens.append(
            self._merge_pairs_in_word(word_tokens, next_pair_to_merge)
        )
      new_tokenized_corpus.append(new_para_tokens)
    return new_tokenized_corpus

  def _update_vocabulary(self, vocabulary_set: set[str]) -> None:
    """Updates the vocabulary with required characters and special tokens.

    This method ensures essential Arabic characters are included in the
    tokenizer's vocabulary, adds special tokens (PAD and UNKNOWN), and builds
    the token-to-index mappings. It finalizes the vocabulary after BPE training
    is complete and updates the instance attributes (vocabulary,
    vocabulary_size, token_to_index, index_to_token, pad_token_id, and
    unknown_token_id).

    Args:
      vocabulary_set: Set of unique tokens discovered during BPE training.
    """
    # Ensure that basic characters in Arabic are always included in
    # the vocabulary.
    required_chars = set(
        "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"  # Arabic letters
        + "٠١٢٣٤٥٦٧٨٩"  # Arabic-Indic digits
        + string.digits   # Western digits
        + "،؛؟!"        # Essential punctuation and space
    )
    vocabulary_set.update(required_chars)

    # Add special tokens to the vocabulary.
    self.vocabulary = (
        [self.PAD_TOKEN, self.UNKNOWN_TOKEN] + sorted(vocabulary_set)
    )

    # Build mappings and set IDs of special tokens.
    self.vocabulary_size = len(self.vocabulary)
    self.token_to_index = {tok: i for i, tok in enumerate(self.vocabulary)}
    self.index_to_token = {i: tok for i, tok in enumerate(self.vocabulary)}
    self.pad_token_id = self.token_to_index[self.PAD_TOKEN]
    self.unknown_token_id = self.token_to_index[self.UNKNOWN_TOKEN]

  def _split_text(self, text: str) -> list[str]:
    """Splits text into subword tokens using learned BPE merges.

    Splits a string that possibly contains space separators into subword tokens
    using learned BPE merges.

    Args:
      text: String to split into subword tokens.

    Returns:
      List of subword tokens that together form the original text.
    """
    tokens = []
    for word in text.strip().split():
      tokens.extend(self._split_word(word))
    return tokens

  def _split_word(self, word: str) -> list[str]:
    """Splits a word into subword tokens using learned BPE merges.

    Splits a word (string without space separators) into subword tokens using
    learned BPE merges.

    Args:
      word: String to split into subword tokens.

    Returns:
      List of subword tokens that together form the original word.
    """
    # Split the string into characters and add special END_WORD token.
    subwords = list(word) + [self.END_WORD]

    # Merge individual characters according to learned BPE merges.
    for pair in self.merges:
      subwords = self._merge_pairs_in_word(subwords, pair)
    return subwords

  def _find_next_pair_to_merge(
      self,
      tokenized_corpus: list[list[list[str]]],
  ) -> tuple[str, str] | None:
    """Finds the most frequent adjacent token pair in the corpus.

    Args:
      tokenized_corpus: List of paragraphs, each containing lists of word
        tokens.

    Returns:
      Tuple of the most frequent adjacent token pair, or None if no valid pairs
      are found.
    """
    # Flatten corpus to single paragraph of words as tokens.
    flat_corpus = []
    for tokenized_paragraph in tokenized_corpus:
      flat_corpus.extend(tokenized_paragraph)

    # Find the most frequent pair of adjacent tokens.
    pair_freqs = self._get_pair_frequencies(flat_corpus)
    if not pair_freqs:
      return None
    most_freq_pair, freq = pair_freqs.most_common(1)[0]
    if freq < 1:
      return None
    return most_freq_pair

  def _tokenize_corpus(
      self,
      corpus: Sequence[str] | str,
  ) -> tuple[list[list[list[str]]], set[str]]:
    """Tokenizes a raw text corpus using current BPE merges and builds vocabulary.

    Splits each paragraph into words and applies learned BPE merges (or splits
    into characters with end-of-word markers when no merges exist yet).

    Args:
      corpus: A sequence of strings or a single string for tokenization.

    Returns:
      Tuple of (tokenized_corpus, vocabulary_set) where tokenized_corpus
      is a list of paragraphs containing word token lists, and
      vocabulary_set contains all unique tokens found.
    """
    # Normalize text to list of strings.
    if isinstance(corpus, str):
      corpus = [corpus]

    # List of lists of lists to store tokenized text corpus.
    tokenized_corpus = []

    if self.vocabulary is None:
      vocabulary_set = {self.END_WORD}
    else:
      # Skip the first two special tokens (<PAD> and <UNK>) which are placed at
      # the beginning of the vocabulary.
      vocabulary_set = set(self.vocabulary[2:])

    for paragraph in corpus:
      sentence_raw_tokens = []
      for word in paragraph.strip().split():
        # Split the word into characters and add word boundary marker.
        sentence_raw_tokens.append(self._split_word(word))
        vocabulary_set.update(word)
      tokenized_corpus.append(sentence_raw_tokens)
    return tokenized_corpus, vocabulary_set
