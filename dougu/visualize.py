from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from logzero import logger


def save_barplot(
    df: pd.DataFrame,
    out_path: str,
    overwrite: bool = False,
    title: str = "title",
    x_name: str = "x_label",
    y_name: str = "y_label",
) -> None:
    """
    Args:
        df: Target dataframe to plot the bar
        out_path: Path to an output file
        overwrite: If true, overwrite 'out_path'
        title: Title name
        x_name: x label name
        y_name: y label name

    Usage:
        ```
        >>> from visualize import save_barplot
        >>> from pandas as pd
        >>> df = pd.DataFrame({"x_label": [1, 2, 3], "y_label": [21, 34, 56]})
        >>> save_barplot(df, out_path='/path/to/file', title='sample', x_name="x_label", y_name="y_label")
        ```

    """

    if Path(out_path).exists() and not overwrite:
        raise FileExistsError(
            "'{out_path}' already exists. Please set 'overwrite' to true."
        )

    sns.set()
    sns.set_style("ticks")
    plt.figure(figsize=(10, 5))
    sns.set_context("paper", 1.5)

    ax = sns.barplot(
        data=df, x=x_name, y=y_name, palette=sns.color_palette("coolwarm", 24)
    )
    sns.despine()
    ax.tick_params(axis="both", length=0)

    plt.title(title, fontsize=15)
    plt.savefig(out_path, bbox_inches="tight", format="pdf", transparent=True)
    plt.close("all")

    logger.info(f"save to '{out_path}'")
