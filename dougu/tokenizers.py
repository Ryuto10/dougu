import argparse
import string
import unicodedata
from os import path

import MeCab
import sentencepiece
from logzero import logger

NEW_LINE_CHAR = "\\n"


def create_parser():
    parser = argparse.ArgumentParser(description="tokenize")
    parser.add_argument('--in', dest="input", type=path.abspath, help='Path to input file.')
    parser.add_argument('--mecab', action='store_true', help='If true, text is tokenized by mecab.')
    parser.add_argument('--spm', type=path.abspath, help='Path to sentencepiece model file.')

    return parser


# 生文の正規化
def normalize(text: str) -> str:
    # 改行を統一
    string.whitespace
    for s in ['\n', '\r', '\v', '\f']:
        text = text.replace(s, "\\n")
    # 空白を統一
    text = text.replace("\t", " ")
    text = unicodedata.normalize("NFKC", text)

    return text


class Detokenizer:
    """tokenizeしたtextを元に戻す"""
    def __init__(self, spm_path: str = None):
        self.spm_path: str = spm_path
        self.spm = None
        self.space_char = "▁"

        # sentencepiece
        if self.spm_path is not None:
            assert path.exists(self.spm_path)
            self.spm = sentencepiece.SentencePieceProcessor()
            self.spm.Load(self.spm_path)
            logger.info(f'Load: {self.spm_path}')

    def __call__(self, text: str) -> str:
        if self.spm is not None:
            sentences = [self.spm.DecodePieces(sentence.split(" ")) for sentence in text.split(NEW_LINE_CHAR)]
            return f"{NEW_LINE_CHAR}".join(sentences)
        else:
            detokenized_text = text.replace(" ", "")
            detokenized_text = detokenized_text.replace(self.space_char, " ")
            return detokenized_text


class MecabTokenizer:
    def __init__(self, mecab_option: str = None, wakati: bool = True) -> None:
        if self.mecab_option is None:
            import ipadic
            mecab_option = ipadic.MECAB_ARGS
            if wakati:
                mecab_option += " -Owakati"

        self.wakati = wakati
        self.mecab_option = mecab_option
        self.mecab_tagger = MeCab.Tagger(self.mecab_option)

    def __call__(self, text: str) -> str:
        text = normalize(text.rstrip("\n"))
        text = text.replace(" ", "▁")  # 空白を残す
        if self.wakati:
            sentences = [self.mecab_tagger.parse(sentence).rstrip("\n ") for sentence in text.split(NEW_LINE_CHAR)]

            return f" {NEW_LINE_CHAR} ".join(sentences)
        else:
            self._customize_tokenization(text)

    def _customize_tokenization(self, text: str) -> str:
        """Please customize"""
        # for line in self.mecab.parse(text).split('\n'):
        #     if line == 'EOS':
        #         pass
        #     token, rest = line.split('\t')

        return text


class SentencepieceTokenizer:
    def __init__(self, spm_path: str = None) -> None:
        assert path.exists(spm_path), f"Not found: {spm_path}"
        self.spm_path: str = spm_path

        self.spm = sentencepiece.SentencePieceProcessor()
        self.spm.Load(self.spm_path)

    def __call__(self, text: str) -> str:
        text = normalize(text.rstrip("\n"))
        sentences = [" ".join(self.spm.EncodeAsPieces(sentence)) for sentence in text.split(NEW_LINE_CHAR)]

        return f" {NEW_LINE_CHAR} ".join(sentences)
