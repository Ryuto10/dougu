import unittest
from pathlib import Path

from dougu.tokenizers import (
    MecabTokenizer,
    Morph,
    SentencepieceTokenizer,
    Tokenizer,
    normalize,
)


class TestNormalize(unittest.TestCase):
    def test_normalizing_with_unicodedata_NFKC(self) -> None:
        test_instances = [
            ("①1１", "111"),
            ("ｶﾞガが", "ガガが"),
        ]
        for input_text, expected in test_instances:
            actual = normalize(text=input_text, newline_char="\n", space_char=" ")
            self.assertEqual(actual, expected)

    def test_replacing_newline_to_specified_char(self) -> None:
        # pattern 1
        input_text = "first\nsecond\rthird"
        expected = "first\nsecond\nthird"
        actual = normalize(text=input_text, newline_char="\n", space_char=" ")
        self.assertEqual(actual, expected)

        # pattern 2
        input_text = "first\nsecond\rthird"
        expected = "first\\nsecond\\nthird"
        actual = normalize(text=input_text, newline_char="\\n", space_char=" ")
        self.assertEqual(actual, expected)

    def test_replacing_space_to_specified_char(self) -> None:
        # pattern 1
        input_text = "space　is\treplaced to '<>'"
        expected = "space<>is<>replaced<>to<>'<>'"
        actual = normalize(text=input_text, newline_char="\n", space_char="<>")
        self.assertEqual(actual, expected)

        # pattern 2
        input_text = "space　is\treplaced to '　'"
        expected = "space　is　replaced　to　'　'"
        actual = normalize(text=input_text, newline_char="\n", space_char="　")
        self.assertEqual(actual, expected)


class TestMecabTokenizer(unittest.TestCase):
    newline = "\\n"
    space = "　"

    def test_instantiating_class(self) -> None:
        tokenizer = MecabTokenizer()
        self.assertIsInstance(tokenizer, Tokenizer)
        self.assertIsInstance(tokenizer, MecabTokenizer)

    def test_tokenize_wakati(self) -> None:
        tokenizer = MecabTokenizer(
            tokenize_type="wakati",
            newline_char=self.newline,
            space_char=self.space,
        )

        input_text = "テストの文です\n空白も あります"
        expected = [
            "テスト",
            "の",
            "文",
            "です",
            self.newline,
            "空白",
            "も",
            self.space,
            "あり",
            "ます",
        ]
        actual = tokenizer.tokenize(input_text)
        self.assertEqual(actual, expected)

    def test_tokenize_noun(self) -> None:
        tokenizer = MecabTokenizer(
            tokenize_type="noun",
            newline_char=self.newline,
            space_char=self.space,
        )

        input_text = "テストの文です\n空白も あります"
        expected = ["テスト", "文", "空白"]
        actual = tokenizer.tokenize(input_text)
        self.assertEqual(actual, expected)

    def test_tokenize_morph(self) -> None:
        tokenizer = MecabTokenizer(
            tokenize_type="morph",
            newline_char=self.newline,
            space_char=self.space,
        )

        input_text = "テストの文です\n空白も あります"
        expected = [
            Morph(
                surface_form="テスト",
                pos="名詞",
                pos_detail="サ変接続",
                yomi_hira="てすと",
                yomi_kata="テスト",
            ),
            Morph(
                surface_form="の",
                pos="助詞",
                pos_detail="連体化",
                yomi_hira="の",
                yomi_kata="ノ",
            ),
            Morph(
                surface_form="文",
                pos="名詞",
                pos_detail="一般",
                yomi_hira="ぶん",
                yomi_kata="ブン",
            ),
            Morph(
                surface_form="です",
                pos="助動詞",
                pos_detail="*",
                yomi_hira="です",
                yomi_kata="デス",
            ),
            Morph(
                surface_form=self.newline,
                pos="記号",
                pos_detail="改行",
                yomi_hira=self.newline,
                yomi_kata=self.newline,
            ),
            Morph(
                surface_form="空白",
                pos="名詞",
                pos_detail="一般",
                yomi_hira="くうはく",
                yomi_kata="クウハク",
            ),
            Morph(
                surface_form="も",
                pos="助詞",
                pos_detail="係助詞",
                yomi_hira="も",
                yomi_kata="モ",
            ),
            Morph(
                surface_form=self.space,
                pos="記号",
                pos_detail="空白",
                yomi_hira=self.space,
                yomi_kata=self.space,
            ),
            Morph(
                surface_form="あり",
                pos="動詞",
                pos_detail="自立",
                yomi_hira="あり",
                yomi_kata="アリ",
            ),
            Morph(
                surface_form="ます",
                pos="助動詞",
                pos_detail="*",
                yomi_hira="ます",
                yomi_kata="マス",
            ),
        ]
        actual = tokenizer.tokenize(input_text)
        self.assertEqual(actual, expected)

    def test_tokenize_yomi(self) -> None:
        tokenizer = MecabTokenizer(
            tokenize_type="yomi",
            newline_char=self.newline,
            space_char=self.space,
        )

        input_text = "テストの文です\n空白も あります"
        expected = [
            "てすと",
            "の",
            "ぶん",
            "です",
            self.newline,
            "くうはく",
            "も",
            self.space,
            "あり",
            "ます",
        ]
        actual = tokenizer.tokenize(input_text)
        self.assertEqual(actual, expected)

    def test_detokenize(self) -> None:
        tokenizer = MecabTokenizer(
            tokenize_type="wakati",
            newline_char=self.newline,
            space_char=self.space,
        )
        input_tokens = [
            "テスト",
            "の",
            "文",
            "です",
            self.newline,
            "空白",
            "も",
            self.space,
            "あります",
        ]
        expected = "テストの文です\n空白も あります"
        actual = tokenizer.detokenize(input_tokens)
        self.assertEqual(actual, expected)

    def test_unsupported_tokenize_type(self) -> None:
        tokenizer = MecabTokenizer(
            tokenize_type="hoge",
            newline_char=self.newline,
            space_char=self.space,
        )

        with self.assertRaises(ValueError, msg="unsupported value: hoge"):
            tokenizer.tokenize("てすと")

    def test_unsupported_mecab_option(self) -> None:
        with self.assertRaises(ValueError, msg="please remove the option '-Owakati'"):
            _ = MecabTokenizer(
                mecab_option='-r "/path/to/mecabrc" -d "/path/to/dicdir" -Owakati',
                tokenize_type="hoge",
                newline_char=self.newline,
                space_char=self.space,
            )


class TestSentencepieceTokenizer(unittest.TestCase):
    newline = "\\n"
    spm_path = str(Path(__file__).resolve().parent / "samples" / "sample.spm")

    def test_instantiating_class(self) -> None:
        tokenizer = SentencepieceTokenizer(self.spm_path, self.newline)
        self.assertIsInstance(tokenizer, Tokenizer)
        self.assertIsInstance(tokenizer, SentencepieceTokenizer)

    def test_tokenize(self) -> None:
        tokenizer = SentencepieceTokenizer(self.spm_path, self.newline)
        input_text = "テストの文です\n空白も あります"
        expected = ["テスト", "の", "文", "です", self.newline, "空", "白", "も", "▁", "あります"]
        actual = tokenizer.tokenize(input_text)
        self.assertEqual(actual, expected)

    def test_detokenize(self) -> None:
        tokenizer = SentencepieceTokenizer(self.spm_path, self.newline)
        input_tokens = ["テスト", "の", "文", "です", self.newline, "空", "白", "も", "▁", "あります"]
        expected = "テストの文です\n空白も あります"
        actual = tokenizer.detokenize(input_tokens)
        self.assertEqual(actual, expected)
