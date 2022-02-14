import os
import unittest
from pathlib import Path

import pandas as pd

from dougu.visualize import save_barplot


class TestSaveBarPlot(unittest.TestCase):
    def test_saving_bar_plot_as_pdf(self) -> None:
        df = pd.DataFrame(
            {
                "x_label": [1, 2, 3],
                "y_label": [21, 34, 56],
            }
        )
        out_path = str(Path(__file__).resolve().parent / "samples" / "sample.pdf")
        save_barplot(
            df,
            out_path=out_path,
            overwrite=True,
            title="sample",
            x_name="x_label",
            y_name="y_label",
        )

        self.assertTrue(Path(out_path).exists())
        os.remove(out_path)
