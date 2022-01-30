from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from logzero import logger
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm


class WordEmbedding:
    """Measuring the similarity of word embeddings"""

    def __init__(self, file_path: str) -> None:
        self.__load_vectors(file_path)

    def __load_vectors(self, file_path: str) -> None:
        """Load embedding file
        Args:
            file_path: Path to a file with a word and its vector on one line (GloVe or Word2Vec format)
                ```
                てすと 0.31882 0.89289 0.90071 0.45753 0.37083 0.64955 0.34075 0.70048 0.89085 0.13621
                サンプル 0.79375 0.44464 0.07644 0.35242 0.03996 0.68827 0.97103 0.77324 0.72781 0.69158
                ...
                ```
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"not found: {file_path}")
        logger.debug(f"load embedding: {file_path}")

        fi = open(file_path)
        line = next(fi)
        # In Word2Vec format, the first line is the number of words and the number of dimensions
        if len(line.split()) != 2:
            fi.seek(0)  # GloVe format

        word2index = {}
        index2word = {}
        matrix = []

        # read file
        for idx, line in tqdm(enumerate(fi)):
            word, *vec = line.rstrip("\n").split(" ")
            word2index[word] = idx
            index2word[idx] = word
            matrix.append(list(map(float, vec)))

        # create embeddings
        self.__word2index = word2index
        self.__index2word = index2word
        self.__matrix = np.array(matrix)
        self.__n_vocab, self.__dim = self.__matrix.shape

    @property
    def weights(self) -> np.ndarray:
        """Returns word embedding matrix"""
        return self.__matrix

    @property
    def dim(self) -> int:
        """Returns the dimension of embeddings"""
        return self.__dim

    @property
    def shape(self) -> Tuple[int, int]:
        """Returns the shape of embeddings"""
        return self.__n_vocab, self.__dim

    @property
    def vocab(self) -> Set[str]:
        """Returns the vocabulary"""
        return set(self.__word2index.keys())

    def to_index(self, word: str) -> int:
        """Convert the word to the index"""
        if not self.is_known(word):
            raise ValueError(f"unknown word: '{word}'")
        return self.__word2index[word]

    def to_word(self, index: int) -> str:
        """Convert the index to the word"""
        if index > self.__n_vocab:
            raise ValueError(
                f"index is over the max length of matrix: {index} > {self.__n_vocab}"
            )
        return self.__index2word[index]

    def is_known(self, word: str) -> bool:
        """judge whether the word is known"""
        return word in self.__word2index

    def __len__(self) -> int:
        """Returns the number of vocabulary"""
        return self.__n_vocab

    def __call__(self, words: Union[str, List[str]]) -> np.ndarray:
        """Convert tokens to embeddings

        Args:
            words: List of words or word

        Returns:
            embeddings: Embeddings of given words
        """
        if isinstance(words, str):
            index = self.to_index(words)
            return self.weights[index]

        indices = [self.to_index(word) for word in words]
        return self.weights[indices]

    def __compute_best_n(
        self, vec: np.ndarray, top_n: int = 10, ignore_top: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Args:
            vec: Target vector
            top_n:  Top N

        Return:
            best_n: [(word, score), ...] (Length of list = top_n)

        """
        similar_scores = self.compute_cosine_similarity(vec)
        v = 1 if ignore_top else 0
        sort_ids = np.argsort(-similar_scores)[v : v + top_n]
        best_n = [(self.to_word(index), similar_scores[index]) for index in sort_ids]

        return best_n

    def compute_cosine_similarity(self, vec: np.ndarray) -> np.ndarray:
        """
        Args:
            vec: Target vector

        Returns:
            similar_score: Results of computing cosine similarity
        """
        if len(vec.shape) == 1:
            vec = vec.reshape(1, -1)
        similar_scores = np.squeeze(cosine_similarity(vec, self.weights))

        return similar_scores

    def compute_similar_words_from_word(
        self, word: str, top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Args:
            word: The target word
            top_n: Top N

        Return:
            best_n: [(word, score), ...] (Length of list = top_n)

        """
        vec = self.weights[self.to_index(word)]
        return self.__compute_best_n(vec, top_n, ignore_top=True)

    def compute_similar_words_from_vec(
        self, vec: np.ndarray, top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Args:
            vec: The target vector
            top_n: Top N

        Return:
            best_n: [(word, score), ...], (Length of list = top_n)

        """
        if vec.shape[-1] != self.dim:
            raise ValueError(
                f"the dimensions of the vectors don't match: {vec.shape[-1]} != {self.dim}"
            )
        if len(vec.shape) > 1:
            raise ValueError(f"unsupported shape: {vec.shape}")
        return self.__compute_best_n(vec, top_n, ignore_top=False)


@dataclass
class SIF:
    """Create sentence embeddings (paper: https://openreview.net/pdf?id=SyK00v5xx)"""

    def __init__(
        self,
        embeddings: np.ndarray,
        word2index: Dict[str, int],
        word_frequency: Optional[Dict[str, int]] = None,
        alpha: float = 1e-03,
        min_freq: Optional[int] = None,
        normalize_by_word: bool = False,
        normalize_by_sent: bool = True,
    ):
        """
        Args:
            embeddings:        Word embeddings
            word2index:        Dictionary to convert from a word to an index
            word_frequency:    Word frequency
            alpha:             Parameter to compute SIF embeddings
            min_freq:          The minimum number of the frequency to use for computing sentence vector
            normalize_by_word: If true, each word is normalized with L2 norm (default: False)
            normalize_by_sent: If true, each sentence is normalized with L2 norm (default: True)

        """
        self.__embeddings = embeddings
        self.__word2index = word2index
        self.__normalize_by_sent = normalize_by_sent

        self.__use_sif = False
        self.__n_vocab, self.__dim = embeddings.shape
        self.__zero_vector = np.zeros(self.__dim)
        self.__sif_weights = np.ones(self.__n_vocab)

        if word_frequency:
            self.__use_sif = True
            self.update_sif_weights(
                word_freq=word_frequency, alpha=alpha, min_freq=min_freq
            )

        if normalize_by_word:
            self.__embeddings = normalize(self.__embeddings, norm="l2")
            logger.info("word embeddings are normalized with L2 norm")

    def __call__(self, tokens: List[str]) -> np.ndarray:
        """Create sentence vector

        Args:
            tokens: the target tokens
        """
        if isinstance(tokens, str):
            raise ValueError("tokens must be list of string, not string.")

        ids = [
            self.__word2index[token] for token in tokens if token in self.__word2index
        ]

        if len(ids) == 0:
            logger.warning(f"all words are unknown: {', '.join(tokens)}")
            logger.warning(
                f"please confirm that the vocabulary of tokenizer and that of embedding are the same"
            )
            return self.__zero_vector

        # create vector
        embeds = self.embeds[ids]
        if self.__use_sif:
            sif_weights = self.__sif_weights[ids]
            sent_vec = np.dot(sif_weights, embeds)
        else:
            sent_vec = np.sum(embeds, axis=0)

        # normalize
        if self.__normalize_by_sent:
            sent_vec = normalize(sent_vec.reshape(1, -1), norm="l2").reshape(-1)

        return sent_vec

    @property
    def embeds(self) -> np.ndarray:
        """Returns embedding matrix"""
        return self.__embeddings

    def use_sum(self) -> None:
        """Set the flag to sum vectors (do not use the SIF weighting)"""
        self.__use_sif = False

    def use_sif(self) -> None:
        """Set the flag to use the SIF weighting"""
        self.__use_sif = True

    def get_sif_weight(self, word: str) -> float:
        """Returns the SIF weight for the word"""
        idx = self.__word2index.get(word)
        return 0.0 if idx is None else float(self.__sif_weights[idx])

    def update_sif_weights(
        self, word_freq: Dict[str, int], alpha: float, min_freq: Optional[int]
    ) -> None:
        """Create weights to compute SIF embeddings

        Args:
            word_freq: Word frequency
            alpha:     Parameter to compute SIF embeddings
            min_freq:  If this value is set, words with a frequency lower than this number will not be used.

        """
        word_freq_vecs = np.zeros(self.__n_vocab)
        for word, freq in word_freq.items():
            # If the frequency is smaller than min_freq, don't use it
            if min_freq is not None and int(freq) < min_freq:
                continue
            if word in self.__word2index:
                idx = self.__word2index[word]
                word_freq_vecs[idx] = float(freq)

        total_freq = sum(word_freq.values())
        word_probabilities = word_freq_vecs / total_freq
        self.__sif_weights = alpha / (alpha + word_probabilities)

        if not self.__use_sif:
            logger.warn(
                "the flag to use SIF is currently false. please set the flag to true with 'use_sif()'"
            )
