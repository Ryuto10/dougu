import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, List, Optional, TypedDict, Union

import ipadic
import jaconv
import MeCab
import sentencepiece
from logzero import logger

NEWLINE_CHAR = "\\n"
FULL_WIDTH_SPACE = "　"


def normalize(text: str, newline_char: str, space_char: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for s in ["\n", "\r", "\v", "\f"]:
        text = text.replace(s, newline_char)
    text = text.replace(" ", space_char)
    text = text.replace("\t", space_char)

    return text


class Morph(TypedDict):
    """Morpheme"""

    surface_form: str
    pos: str
    pos_detail: str
    yomi_hira: str
    yomi_kata: str


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> Union[List[str], List[Morph]]:
        """Tokenize text

        Args:
            text: Target text to be tokenized

        Returns:
            tokens: Tokenized tokens or morphemes

        """
        ...

    @abstractmethod
    def detokenize(self, tokens: List[str]) -> str:
        """Restore tokens to their original sentence

        Args:
            tokens: Tokenized tokens

        Returns:
            text: Reconstructed text

        """
        ...


class MecabTokenizer(Tokenizer):
    """Tokenizer using MeCab"""

    def __init__(
        self,
        mecab_option: Optional[str] = None,
        tokenize_type: str = "wakati",
        newline_char: str = NEWLINE_CHAR,
        space_char: str = FULL_WIDTH_SPACE,
    ) -> None:
        """
        Args:
            mecab_option:  MeCab option (e.g. '-r "/path/to/mecabrc" -d "/path/to/dicdir"')
            tokenize_type: Choose from ['wakati', 'noun', 'morph', 'yomi']
            newline_char:  Characters to be considered as a newline character after tokenize
            space_char:    Characters to be considered as a space character after tokenize

        """
        self.__mecab_option = mecab_option
        self.__tokenize_type = tokenize_type
        self.newline_char = newline_char
        self.space_char = space_char

        self.__set_mecab_option()
        self.__mecab_tagger = MeCab.Tagger(self.__mecab_option)

    def tokenize(self, text: str) -> Union[List[str], List[Morph]]:
        """Tokenize text according to 'self.__tokenized_type'"""
        text = normalize(
            text=text.rstrip("\n"),
            newline_char=self.newline_char,
            space_char=FULL_WIDTH_SPACE,
        )

        tokenize_func_name = "tokenize_" + self.__tokenize_type

        if hasattr(self, tokenize_func_name):
            tokens = getattr(self, tokenize_func_name)(text)
        else:
            raise ValueError(f"unsupported value: {self.__tokenize_type}")

        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        """Restore tokens to their original sentence
        (spaces and newline are completely unrecoverable)
        """
        text = (
            "".join(tokens)
            .replace(self.newline_char, "\n")
            .replace(self.space_char, " ")
        )

        return text

    def __set_mecab_option(self) -> None:
        """Set mecab option"""
        if self.__mecab_option is None:
            self.__mecab_option = ipadic.MECAB_ARGS

        if "-Owakati" in self.__mecab_option:
            raise ValueError("please remove the option '-Owakati'")

    def __read_mecab_line(self, text: str) -> Generator[Morph, None, None]:
        """
        Args:
            text: Target text to be tokenized

        Yields:
            morph: Morpheme

        """
        for idx, sentence in enumerate(text.split(self.newline_char)):
            if idx != 0:
                morph = Morph(
                    surface_form=self.newline_char,
                    pos="記号",
                    pos_detail="改行",
                    yomi_hira=self.newline_char,
                    yomi_kata=self.newline_char,
                )
                yield morph

            for line in self.__mecab_tagger.parse(sentence).split("\n"):
                if not line or line == "EOS":
                    continue
                surface_form, rest = line.split("\t")
                pos, pos_detail, *_, yomi_kata, _ = rest.split(",")
                if yomi_kata == "*":
                    yomi_kata = surface_form
                yomi_hira = jaconv.kata2hira(yomi_kata)

                if pos_detail == "空白":
                    surface_form = self.space_char
                    yomi_hira = self.space_char
                    yomi_kata = self.space_char

                morph = Morph(
                    surface_form=surface_form,
                    pos=pos,
                    pos_detail=pos_detail,
                    yomi_hira=yomi_hira,
                    yomi_kata=yomi_kata,
                )

                yield morph

    def tokenize_wakati(self, text: str) -> List[str]:
        """Split text into words:
            e.g. "テストの文です" -> ["テスト", "の", "文", "です"]

        Args:
            text: Target text to be tokenized

        Returns:
            tokens: Tokenized tokens

        """
        tokens = [morph["surface_form"] for morph in self.__read_mecab_line(text)]

        return tokens

    def tokenize_noun(self, text: str) -> List[str]:
        """Split text into words and select only nouns:
            e.g. "テストの文です" -> ["テスト", "文"]

        Args:
            text: Target text to be tokenized

        Returns:
            tokens: Tokenized tokens

        """
        tokens = [
            morph["surface_form"]
            for morph in self.__read_mecab_line(text)
            if morph["pos"] == "名詞"
        ]

        return tokens

    def tokenize_morph(self, text: str) -> List[Morph]:
        """Split text into morphemes:
            e.g. "テストの文です" -> [
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
                        pos_detail="連体詞",
                        yomi_hira="の",
                        yomi_kata="ノ",
                    ),
                    ...
                 ]

        Args:
            text: Target text to be tokenized

        Returns:
            morphemes: Tokenized tokens (morphemes)

        """

        morphemes = [morph for morph in self.__read_mecab_line(text)]

        return morphemes

    def tokenize_yomi(self, text: str) -> List[str]:
        """split text into words and extract reading kana
        e.g. "テストの文です" -> ["てすと", "の", "ぶん", "です"]

        Args:
            text: Target text to be tokenized

        Returns:
            tokens: Tokenized tokens (reading kana)

        """
        tokens = [morph["yomi_hira"] for morph in self.__read_mecab_line(text)]

        return tokens


class SentencepieceTokenizer(Tokenizer):
    def __init__(self, spm_path: str, newline_char: str = NEWLINE_CHAR) -> None:
        """
        Args:
            spm_path:     Path to the sentencepiece model
            newline_char: Characters to be considered as a newline character after tokenize

        """
        if not Path(spm_path).exists():
            raise FileNotFoundError(f"not found: {spm_path}")

        logger.debug(f"loading sentencepiece model: {spm_path}")
        self.__spm = sentencepiece.SentencePieceProcessor()
        self.__spm.Load(spm_path)
        self.newline_char = newline_char

    def tokenize(self, text: str) -> List[str]:
        """Split text into subwords using sentencepiece model"""
        text = normalize(
            text.rstrip("\n"),
            newline_char=self.newline_char,
            space_char=" ",
        )
        tokens = []
        for idx, sentence in enumerate(text.split(self.newline_char)):
            if idx != 0:
                tokens.append(self.newline_char)
            tokens.extend(self.__spm.EncodeAsPieces(sentence))

        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        """Restore tokens to their original sentence"""
        text = ""
        for idx, sentence in enumerate(" ".join(tokens).split(self.newline_char)):
            if idx != 0:
                text += "\n"
            text += self.__spm.DecodePieces(sentence.split(" "))

        return text
