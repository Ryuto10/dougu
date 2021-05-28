from typing import List, Tuple

import numpy as np
from logzero import logger
from sklearn.metrics.pairwise import cosine_similarity

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
            word (str): The target vector
            top_n (int): Top N

        Return:
            best_n: [(word, score), ...], (Length of list = top_n)
        """
        assert vec.shape == (self.dim, ), f"The dimensions of the vectors don't match ({vec.shape} != {self.dim})."
        scores = cosine_similarity(vec.reshape(1, -1), self.embeds)[0]
        index2word = {idx: word for word, idx in self.word2index.items()}
        best_n = [(index2word[index], scores[index]) for index in np.argsort(-scores)[1:top_n + 1]]

        return best_n

    def get_known_words(self, words: List[str]) -> List[str]:
        known_words = [word for word in words if word in self.word2index]

        return known_words
