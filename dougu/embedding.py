import numpy as np
from logzero import logger
from sklearn.metrics.pairwise import cosine_similarity

from .read import read_file


class WordVecSimilar:
    """Measuring the similarity of word embeddings"""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.n_vocab = None
        self.dim = None
        self.matrix = None
        self.word2index = None
        self.on_display = False
        self._load_vectors()

    def __call__(self, word: str, n: int = 10):
        """
        Args:
            word: Target word
            n: Top n
        Return:
            best_n: [(word, score), ...], (Length of list = n)
        """
        if word not in self.word2index:
            logger.warning("{} is UNK".format(word))
        else:
            scores = cosine_similarity(self.matrix[self.word2index[word]].reshape(1, -1), self.matrix)[0]
            index2word = {idx: word for word, idx in self.word2index.items()}
            best_n = [(index2word[index], scores[index]) for index in np.argsort(scores)[::-1][1:n + 1]]

            if self.on_display:
                self.display(best_n)  # returnする代わりに見やすく表示する
            else:
                return best_n

    def _load_vectors(self):
        """Load embedding file"""
        self.word2index = {}
        self.matrix = []
        n = 0

        for idx, line in enumerate(read_file(self.model_path), -1):
            if idx == -1:
                # model fileの先頭に"vocab size", "embed dim"が書いてある場合の分岐
                try:  # 書いてある
                    n_vocab, dim = line.rstrip("\n").split()
                    self.n_vocab, self.dim = int(n_vocab), int(dim)
                    continue
                except:  # 書いてない
                    n = 1

            word, vec = line.rstrip("\n").split(" ", 1)
            self.word2index[word] = idx + n
            self.matrix.append([float(v) for v in vec.split()])

        self.matrix = np.array(self.matrix)
        if n == 1:
            self.n_vocab, self.dim = self.matrix.shape
        assert self.matrix.shape == (self.n_vocab, self.dim)

    @staticmethod
    def display(best_n):
        for w, s in best_n:
            print("{}: {}".format(w, s))
