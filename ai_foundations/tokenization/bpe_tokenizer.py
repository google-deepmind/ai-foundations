# Copyright 2025 Google LLC
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

"""Defines a class that implements a Byte Pair Encoding Tokenizer."""

import collections
import pickle
import string
from typing import List, Optional, Set, Tuple
from urllib import request

import tqdm

Counter = collections.Counter


class BPEWordTokenizer:
  """A Byte Pair Encoding (BPE) based subword tokenizer.

  This class supports encoding and decoding text to subword tokens using BPE.
  It can learn merge rules from a corpus or be initialized with a pre-built
  vocabulary.

  Attributes:
    vocabulary: List of subword tokens including special tokens.
    vocabulary_size: Total number of tokens in vocabulary.
    token_to_index: Mapping from tokens to indices.
    index_to_token: Mapping from indices to tokens.
    merges: List of byte pair merges learned during training.
    pad_token_id: Index of the padding token.
    unknown_token_id: Index of the unknown token.
    tokenized_corpus: Cached tokenized corpus after BPE training.
  """

  UNKNOWN_TOKEN = "<UNK>"
  PAD_TOKEN = "<PAD>"
  END_WORD = "</w>"
  CLS = "<CLS>"
  SEP = "<SEP>"
  start_of_turn = "<start_of_turn>"
  end_of_turn = "<end_of_turn>"
  newline = "\n"

  def __init__(
      self,
      texts: List[str] | str,
      vocabulary: Optional[List[str]] = None,
      num_merges: int = 100,
  ):
    """Initializes the BPEWordTokenizer.

    If no vocabulary is specified, it extracts the unique tokens from the
    text corpus and learns the BPE merges.

    Args:
      texts: A list of strings or a string representing the text corpus.
      vocabulary: Optional list of strings with unique tokens.
      num_merges: Defines the how many rounds of merges should be performed when
        learning the BPE merges.
    """

    # Normalize to list of strings.
    if isinstance(texts, str):
      texts = [texts]

    if vocabulary is None:
      # Learn BPE merges and derive vocabulary from tokenized corpus.
      self.merges, vocabulary_set = self._learn_bpe(
          texts, num_merges
      )

      # Ensure that basic alphanumeric characters are always included in the
      # vocabulary.
      required_chars = set(
          string.ascii_lowercase + string.ascii_uppercase + string.digits
      )

      vocabulary_set.update(required_chars)

      # Add special tokens to the vocabulary.
      self.vocabulary = (
          [self.PAD_TOKEN]
          + sorted(vocabulary_set)
          + [self.UNKNOWN_TOKEN]
          + [self.CLS]
          + [self.SEP]
          + [self.start_of_turn]
          + [self.end_of_turn]
          + [self.newline]
      )

    else:
      self.vocabulary = vocabulary
      self.merges = []  # Skip merge logic when a vocabulary is provided.

    # Build mappings and set IDs of special tokens.
    self.vocabulary_size = len(self.vocabulary)
    self.token_to_index = {tok: i for i, tok in enumerate(self.vocabulary)}
    self.index_to_token = {i: tok for i, tok in enumerate(self.vocabulary)}
    self.pad_token_id = self.token_to_index[self.PAD_TOKEN]
    self.unknown_token_id = self.token_to_index[self.UNKNOWN_TOKEN]

  def _split_text(self, text: str) -> List[str]:
    """Split a string into subword tokens using learned BPE merges.

    Args:
      text: String to split into subword tokens.

    Returns:
      List of subword tokens that together form the original text.
    """
    tokens = []
    for word in text.strip().split():
      # Split the string into characters and add special END_WORD token.
      chars = list(word) + [self.END_WORD]

      # Merge individual characters according to learned BPE merges.
      for pair in self.merges:
        chars = self._merge_pairs_in_word(chars, pair)
      tokens.extend(chars)
    return tokens

  def join_text(self, tokens: List[str]) -> str:
    """Join subword tokens into full string, preserving word boundaries.

    Args:
      tokens: List of subword tokens to be joined.

    Returns:
      String obtained from joining the subword tokens.
    """
    words = []
    current_word = []
    for token in tokens:
      # Check whether token ends with a word boundary marker.
      if token.endswith(self.END_WORD):
        current_word.append(token.replace(self.END_WORD, ""))
        words.append("".join(current_word))
        current_word = []
      else:
        current_word.append(token)
    if current_word:
      words.append("".join(current_word))
    return " ".join(words).strip()

  def encode(self, text: str) -> List[int]:
    """Encode a string into list of token indices.

    Args:
      text: Input text.

    Returns:
      List of integers corresponding to tokens.
    """
    token_ids = []
    for token in self._split_text(text):
      token_id = self.token_to_index.get(token, self.unknown_token_id)
      token_ids.append(token_id)
    return token_ids

  def decode(self, token_ids: int | List[int]) -> str:
    """Decode list of token IDs back to original text.

    Args:
      token_ids: Single index or list of token IDs.

    Returns:
      Decoded text string.
    """
    # Covert to list if a single token index is specified.
    if isinstance(token_ids, int):
      token_ids = [token_ids]

    tokens = []
    for token_id in token_ids:
      tokens.append(
          self.index_to_token.get(token_id, self.UNKNOWN_TOKEN + self.END_WORD)
      )

    return self.join_text(tokens)

  def _get_pair_frequencies(
      self, corpus: List[List[str]]
  ) -> Counter[Tuple[str, str]]:
    """Count all adjacent token pairs in corpus.

    Args:
      corpus: A list of lists of strings representing subword tokens.

    Returns:
      Counter mapping adjacent pairs of subword tokens to their frequencies.
    """
    pairs = Counter()
    for word in corpus:
      for i in range(len(word) - 1):
        pair = (word[i], word[i + 1])
        # Increase the count by 1.
        pairs[pair] += 1
    return pairs

  def _merge_pairs_in_word(
      self, word: List[str], pair_to_merge: Tuple[str, str]
  ) -> List[str]:
    """Merge all occurrences of a token pair inside a word.

    Args:
      word: A list of subword tokens representing one space separated word.
      pair_to_merge: A pair of two subword tokens that should be merged into one
        subword token.

    Returns:
      New list of subword tokens representing the word after applying the merge.
    """

    merged_symbol = "".join(pair_to_merge)
    if pair_to_merge[0] not in word or pair_to_merge[1] not in word:
      return word

    i = 0
    new_word = []
    while i < len(word):
      if i < len(word) - 1 and (word[i], word[i + 1]) == pair_to_merge:
        new_word.append(merged_symbol)
        i += 2
      else:
        new_word.append(word[i])
        i += 1
    return new_word

  def _learn_bpe(
      self, corpus: List[str], num_merges: int
  ) -> Tuple[List[Tuple[str, str]], Set[str]]:
    """Learn BPE merges from a corpus of texts.

    Args:
      corpus: List of input texts.
      num_merges: Number of merge operations to perform.

    Returns:
      merges: List of merges in order they are learned to be performed.
      vocabulary_set: Set of subword tokens after performing all merges.
    """
    # List of lists of lists to store tokenized text corpus.
    tokenized_corpus = []
    vocabulary = set([self.END_WORD])
    for paragraph in corpus:
      sentence_raw_tokens = []
      for word in paragraph.strip().split():
        # Split the word into characters and add word boundary marker.
        sentence_raw_tokens.append(list(word) + [self.END_WORD])
        vocabulary.update(list(word))
      tokenized_corpus.append(sentence_raw_tokens)

    merges = []
    for _ in (pbar := tqdm.tqdm(range(num_merges), unit="merges")):
      # Build a one-dimensional list of all tokens in the corpus.
      flat_corpus = []
      for tokenized_paragraph in tokenized_corpus:
        flat_corpus.extend(tokenized_paragraph)

      # Find the most frequent pair of adjecent tokens.
      pair_freqs = self._get_pair_frequencies(flat_corpus)
      if not pair_freqs:
        break
      most_freq_pair, freq = pair_freqs.most_common(1)[0]
      if freq < 1:
        break
      merges.append(most_freq_pair)

      # Apply merge to each token in each paragraph.
      new_tokenized_corpus = []
      for para_tokens in tokenized_corpus:
        new_para_tokens = []
        for word_tokens in para_tokens:
          new_para_tokens.append(
              self._merge_pairs_in_word(word_tokens, most_freq_pair)
          )
        new_tokenized_corpus.append(new_para_tokens)
      tokenized_corpus = new_tokenized_corpus
      vocabulary.add(most_freq_pair[0] + most_freq_pair[1])
      pbar.set_postfix(vocabulary_size=f"{len(vocabulary):,}")

    return merges, vocabulary

  @classmethod
  def from_url(cls, url: str) -> "BPEWordTokenizer":
    """Loads a pickled tokenizer from a url.

    Args:
      url: The URL where the pickled tokenizer is stored.

    Returns:
      An instance of BPEWordTokenizer intialized with the vocabulary and merge
        rules of the original tokenizer.
    """
    with request.urlopen(url) as response:
      tokenizer = pickle.load(response)
    print(
        "Loaded pretrained tokenizer with vocabulary size"
        f" {tokenizer.vocabulary_size:,}."
    )
    return tokenizer
