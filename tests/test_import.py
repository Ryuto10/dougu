class TestImport:
    def test_bq_import(self) -> None:
        import pandas as pd
        from logzero import logger

    def test_embedding_import(self) -> None:
        import numpy
        from logzero import logger
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import normalize

    def test_gcs_import(self) -> None:
        from google.auth import exceptions
        from google.cloud import storage
        from logzero import logger

    def test_read_import(self) -> None:
        import yaml
        from logzero import logger
        from tqdm import tqdm

    def test_s3_import(self) -> None:
        import boto3
        from logzero import logger

    def test_timer_import(self) -> None:
        from logzero import logger

    def test_tokenizer_import(self) -> None:
        import ipadic
        import jaconv
        import MeCab
        import sentencepiece
        from logzero import logger

    def test_visualize_import(self) -> None:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        from logzero import logger
