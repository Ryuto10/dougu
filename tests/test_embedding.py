import unittest
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

from dougu.embedding import SIF, WordEmbedding


class TestWordEmbedding(unittest.TestCase):
    def test_load_word2vec_format_file(self) -> None:
        file_path = str(
            Path(__file__).resolve().parent / "samples" / "sample.word2vec.txt"
        )
        word_embeds = WordEmbedding(file_path)
        self.assertIsInstance(word_embeds, WordEmbedding)

    def test_load_glove_format_file(self) -> None:
        file_path = str(
            Path(__file__).resolve().parent / "samples" / "sample.glove.txt"
        )
        word_embeds = WordEmbedding(file_path)
        self.assertIsInstance(word_embeds, WordEmbedding)

    def test_extract_word_embeddings(self) -> None:
        file_path = str(
            Path(__file__).resolve().parent / "samples" / "sample.word2vec.txt"
        )
        word_embeds = WordEmbedding(file_path)

        words = ["てすと", "サンプル"]
        actual = word_embeds(words)
        expected = np.array(
            [
                [
                    0.31882,
                    0.89289,
                    0.90071,
                    0.45753,
                    0.37083,
                    0.64955,
                    0.34075,
                    0.70048,
                    0.89085,
                    0.13621,
                ],
                [
                    0.79375,
                    0.44464,
                    0.07644,
                    0.35242,
                    0.03996,
                    0.68827,
                    0.97103,
                    0.77324,
                    0.72781,
                    0.69158,
                ],
            ]
        )
        np.testing.assert_almost_equal(actual, expected)

        word = "てすと"
        actual = word_embeds(word)
        expected = np.array(
            [
                0.31882,
                0.89289,
                0.90071,
                0.45753,
                0.37083,
                0.64955,
                0.34075,
                0.70048,
                0.89085,
                0.13621,
            ]
        )
        np.testing.assert_almost_equal(actual, expected)

    def test_error_unknown_words(self) -> None:
        file_path = str(
            Path(__file__).resolve().parent / "samples" / "sample.word2vec.txt"
        )
        word_embeds = WordEmbedding(file_path)

        words = ["てすと", "unk"]
        with self.assertRaises(ValueError, msg=f"unknown word: 'unk'"):
            word_embeds(words)

    def test_compute_similar_words_from_word(self) -> None:
        file_path = str(
            Path(__file__).resolve().parent / "samples" / "sample.word2vec.txt"
        )
        word_embeds = WordEmbedding(file_path)

        word = "てすと"
        expected = [("サンプル", 0.7506074093675397)]
        actual = word_embeds.compute_similar_words_from_word(word)
        self.assertEqual(actual, expected)

    def test_compute_similar_words_from_vec(self) -> None:
        file_path = str(
            Path(__file__).resolve().parent / "samples" / "sample.word2vec.txt"
        )
        word_embeds = WordEmbedding(file_path)

        embed = np.array(
            [
                0.29902,
                0.90019,
                0.89964,
                0.50753,
                0.38001,
                0.59495,
                0.29175,
                0.69909,
                0.90185,
                0.09687,
            ]
        )
        expected = [("てすと", 0.9987114080207757), ("サンプル", 0.7286216119815097)]
        actual = word_embeds.compute_similar_words_from_vec(embed)
        self.assertEqual(actual, expected)

    def test_compute_cosine_similarity(self) -> None:
        file_path = str(
            Path(__file__).resolve().parent / "samples" / "sample.word2vec.txt"
        )
        word_embeds = WordEmbedding(file_path)

        embed = np.array(
            [
                0.29902,
                0.90019,
                0.89964,
                0.50753,
                0.38001,
                0.59495,
                0.29175,
                0.69909,
                0.90185,
                0.09687,
            ]
        )
        expected = np.array([0.9987114080207757, 0.7286216119815097])
        actual = word_embeds.compute_cosine_similarity(embed)
        np.testing.assert_almost_equal(actual, expected)

    def test_property_weight(self) -> None:
        file_path = str(
            Path(__file__).resolve().parent / "samples" / "sample.word2vec.txt"
        )
        word_embeds = WordEmbedding(file_path)

        actual = word_embeds.weights
        expected = np.array(
            [
                [
                    0.31882,
                    0.89289,
                    0.90071,
                    0.45753,
                    0.37083,
                    0.64955,
                    0.34075,
                    0.70048,
                    0.89085,
                    0.13621,
                ],
                [
                    0.79375,
                    0.44464,
                    0.07644,
                    0.35242,
                    0.03996,
                    0.68827,
                    0.97103,
                    0.77324,
                    0.72781,
                    0.69158,
                ],
            ]
        )
        np.testing.assert_almost_equal(actual, expected)

    def test_property_shape(self) -> None:
        file_path = str(
            Path(__file__).resolve().parent / "samples" / "sample.word2vec.txt"
        )
        word_embeds = WordEmbedding(file_path)
        self.assertEqual(len(word_embeds), 2)
        self.assertEqual(word_embeds.dim, 10)
        self.assertEqual(word_embeds.shape, (2, 10))

    def test_property_vocab(self) -> None:
        file_path = str(
            Path(__file__).resolve().parent / "samples" / "sample.word2vec.txt"
        )
        word_embeds = WordEmbedding(file_path)

        self.assertSetEqual(word_embeds.vocab, {"てすと", "サンプル"})
        self.assertEqual(word_embeds.to_word(1), "サンプル")
        self.assertEqual(word_embeds.to_index("てすと"), 0)
        self.assertTrue(word_embeds.is_known("てすと"))
        self.assertFalse(word_embeds.is_known("test"))


class TestSIF(unittest.TestCase):
    embeddings = np.array(
        [
            [
                0.31882,
                0.89289,
                0.90071,
                0.45753,
                0.37083,
                0.64955,
                0.34075,
                0.70048,
                0.89085,
                0.13621,
            ],
            [
                0.79375,
                0.44464,
                0.07644,
                0.35242,
                0.03996,
                0.68827,
                0.97103,
                0.77324,
                0.72781,
                0.69158,
            ],
        ]
    )
    word2index = {"てすと": 0, "サンプル": 1}
    word_freq = {"てすと": 20, "サンプル": 80}

    def test_get_sif_weights(self) -> None:
        sif = SIF(
            embeddings=self.embeddings,
            word2index=self.word2index,
            word_frequency=self.word_freq,
            alpha=0.2,
            normalize_by_word=False,
            normalize_by_sent=False,
        )
        actual = sif.get_sif_weight("てすと")
        expected = 0.5  # 0.2 / (0.2 + 0.2)
        self.assertEqual(actual, expected)

        actual = sif.get_sif_weight("サンプル")
        expected = 0.2  # 0.2 / (0.2 + 0.8)
        self.assertEqual(actual, expected)

    def test_create_sif_embeds_without_normalize(self) -> None:
        sif = SIF(
            embeddings=self.embeddings,
            word2index=self.word2index,
            word_frequency=self.word_freq,
            alpha=0.2,
            normalize_by_word=False,
            normalize_by_sent=False,
        )

        actual = sif(["てすと", "サンプル"])
        expected = np.dot(np.array([0.5, 0.2]), self.embeddings)

        np.testing.assert_almost_equal(actual, expected)

    def test_create_sif_embeds_with_sent_normalize(self) -> None:
        sif = SIF(
            embeddings=self.embeddings,
            word2index=self.word2index,
            word_frequency=self.word_freq,
            alpha=0.2,
            normalize_by_word=False,
            normalize_by_sent=True,
        )

        actual = sif(["てすと", "サンプル"])

        embed = np.dot(np.array([0.5, 0.2]), self.embeddings)
        expected = embed / np.linalg.norm(
            embed, ord=2
        )  # instead of sklearn.preprocessing.normalize

        np.testing.assert_almost_equal(actual, expected)

    def test_create_sum_embeds_with_sent_and_word_normalize(self) -> None:
        sif = SIF(
            embeddings=self.embeddings,
            word2index=self.word2index,
            word_frequency=self.word_freq,
            alpha=0.2,
            normalize_by_word=True,
            normalize_by_sent=True,
        )
        sif.use_sum()

        actual = sif(["てすと", "サンプル"])

        normalized_embeds = normalize(self.embeddings, norm="l2")
        np.testing.assert_almost_equal(normalized_embeds, sif.embeds)

        sum_vec = np.sum(normalized_embeds, axis=0)
        expected = normalize(sum_vec.reshape(1, -1), norm="l2").reshape(-1)

        np.testing.assert_almost_equal(actual, expected)
