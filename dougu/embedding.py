from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from logzero import logger
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .read import read_line


class WordEmbedding:
    """Measuring the similarity of word embeddings"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.n_vocab = None
        self.dim = None
        self.embeds = None
        self.word2index = None
        self.on_display = False
        self._load_vectors()

    def _load_vectors(self) -> None:
        """Load embedding file"""
        self.word2index = {}
        self.embeds = []

        # read file
        is_word2vec_format = None
        for idx, line in enumerate(read_line(self.model_path), -1):
            if idx == -1:
                # model fileの先頭に"vocab size", "embed dim"が書いてある場合の分岐
                try:  # 書いてある (Word2Vec Format)
                    self.n_vocab, self.dim = map(int, line.rstrip("\n").split())
                    is_word2vec_format = True
                    continue
                except ValueError:  # 書いてない (GloVe Format)
                    is_word2vec_format = False

            word, *vec = line.rstrip("\n").split(" ")
            self.word2index[word] = idx + (0 if is_word2vec_format else 1)
            self.embeds.append(list(map(float, vec)))

        # create embeddings
        self.embeds = np.array(self.embeds)

        # check params
        if is_word2vec_format:
            assert self.embeds.shape == (self.n_vocab, self.dim)
        else:
            self.n_vocab, self.dim = self.embeds.shape

    def __call__(self, words: List[str]) -> np.ndarray:
        """Convert tokens to embeddings

        Args:
            tokens (List[str]): The list of tokens

        Returns:
            np.ndarray: Embeddings
        """
        indices = [self.word2index[word] for word in words]
        embeddings = self.embeds[indices]
        assert embeddings.shape == (len(words), self.dim)

        return embeddings

    def compute_similar_words(self, word: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Args:
            word (str): The target word
            top_n (int): Top N

        Return:
            best_n: [(word, score), ...], (Length of list = top_n)
        """
        if word not in self.word2index:
            logger.warning("{} is UNK".format(word))
        else:
            scores = cosine_similarity(self.embeds[self.word2index[word]].reshape(1, -1), self.embeds)[0]
            index2word = {idx: word for word, idx in self.word2index.items()}
            best_n = [(index2word[index], scores[index]) for index in np.argsort(-scores)[1:top_n + 1]]

            return best_n

    def compute_similar_words_with_vec(self, vec: np.ndarray, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Args:
            vec: The target vector
            top_n (int): Top N

        Return:
            best_n: [(word, score), ...], (Length of list = top_n)
        """
        assert vec.shape == (self.dim,), f"The dimensions of the vectors don't match ({vec.shape} != {self.dim})."
        scores = cosine_similarity(vec.reshape(1, -1), self.embeds)[0]
        index2word = {idx: word for word, idx in self.word2index.items()}
        best_n = [(index2word[index], scores[index]) for index in np.argsort(-scores)[1:top_n + 1]]

        return best_n

    def get_known_words(self, words: List[str]) -> List[str]:
        known_words = [word for word in words if word in self.word2index]

        return known_words


@dataclass
class SIF:
    """
    Args:
        - embeddings: Word embeddings
        - word2index: Dictionary to convert from a word to an index
        - word_freq_file: Path to word frequency file.
                        Each line contains a word and its frequency separated by tab.
                        This file is assumed to be sorted in descending order by frequency.
        - normalize_by_word: If true, each word is normalized with L2 norm.
        - normalize_by_sent: If true, each sentence is normalized with L2 norm.
    """
    embeddings: np.ndarray
    word2index: Dict[str, int]
    word_freq_file: str = None
    normalize_by_word: bool = False
    normalize_by_sent: bool = True

    def __post_init__(self):
        self.a = 1e-3
        self.use_sif = True
        self.n_vocab, self.dim = self.embeddings.shape
        self.zero_vector = np.zeros(self.dim)
        self.word_prob = self.create_wordprob_from_wordfreq(
            file_path=self.word_freq_file,
        )

        logger.info(f"Number of vocab: {self.n_vocab}, dim = {self.dim}")

        if self.normalize_by_word:
            self.embeddings = normalize(self.embeddings, norm='l2')
            logger.info("Each word is normalized with L2 norm.")

    def __call__(self, tokens: List[str]) -> np.ndarray:
        """Create sentence vector"""
        known_tokens = [token for token in tokens if token in self.word2index]

        # If there are only unknown words
        if len(known_tokens) == 0:
            logger.warning("Length of words is zero")
            logger.warning(" ".join(tokens))
            return self.zero_vector

        # create vector
        ids = [self.word2index[token] for token in known_tokens]
        embeds = self.embeddings[ids]
        if self.use_sif:
            word_weights = np.array([self.get_sif_weight(token) for token in known_tokens])
            sent_vec = np.dot(word_weights, embeds)
        else:
            sent_vec = np.sum(embeds, axis=0)

        # normalize
        if self.normalize_by_sent:
            sent_vec = normalize(sent_vec.reshape(1, -1), norm='l2').reshape(-1)

        return sent_vec

    def get_sif_weight(self, word: str) -> float:
        w_prob = self.word_prob[word] if word in self.word_prob else 0
        weight = self.a / (self.a + w_prob)

        return weight

    @staticmethod
    def create_wordprob_from_wordfreq(file_path: str, min_freq: int = None) -> Dict[str, float]:
        """
        Args:
            file_path: Path to word frequency file (each line contains a word and its frequency).
            min_freq: If this value is set, words with a frequency lower than this number will not be used.
        Return:
            word_prob: Dictionary for word probability
        """
        logger.info(f"Loading: {file_path}")

        word_freq = {}
        for line in read_line(file_path):
            token, freq = line.split("\t")
            if min_freq is not None and int(freq) < min_freq:
                break
            word_freq[token] = int(freq)
        logger.info("done")

        all_freq = sum(freq for freq in word_freq.values())
        word_prob = {token: freq / all_freq for token, freq in word_freq.items()}

        return word_prob
