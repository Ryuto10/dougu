from pathlib import Path
from typing import Union

import pandas as pd
from logzero import logger


def download_from_bq(
    project_id: str,
    sql_text: Union[str] = None,
    sql_file: Union[str] = None,
    table_name: Union[str] = None,
    local_out_path: Union[str] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Execute the SQL and load the results from BigQuery.
    Please specify only one of the following arguments: 'sql_text', 'sql_file', or 'table_name'

    Args:
        project_id:     project id
        sql_text:       SQL script
        sql_file:       Path to sql scripts
        table_name:     Target table name
        local_out_path: Path to output file
        overwrite:      Whether to overwrite the output file when 'out_file_path' is specified

    Return:
        DataFrame: Loaded dataset

    """
    # error handler
    n_arguments = sum(map(bool, [sql_file, sql_text, table_name]))
    if n_arguments != 1:
        raise ValueError("please specify only one of the arguments")

    if local_out_path and Path(local_out_path).exists() and not overwrite:
        raise FileExistsError(f"'{local_out_path}' already exists")

    if sql_file and not Path(sql_file).exists():
        raise FileNotFoundError(f"'{sql_file}' is not found")

    # set sql text
    if sql_file:
        logger.debug(f"sql file: {sql_file}")
        sql_text = open(sql_file).read()

    elif table_name:
        logger.debug(f"table name: {table_name}")
        sql_text = f"select * from `{table_name}`"

    # execute SQL and load the dataset from bq
    logger.debug(f"project_id: {project_id}")
    logger.debug(f"sql text:\n{sql_text}")
    logger.debug(f"loading dataset from BQ...")
    df = pd.read_gbq(
        sql_text,
        project_id=project_id,
        dialect="standard",
        use_bqstorage_api=False,  # if True, it will cost you money instead of speeding up the execution time
    )
    logger.debug("done")

    # save as jsonl format
    if local_out_path:
        df.to_json(local_out_path, orient="records", force_ascii=False, lines=True)
        logger.debug(f"save to '{local_out_path}'")

    return df


def upload_to_bq(
    df: pd.DataFrame,
    project_id: str,
    dataset_id: str,
    table_id: str,
    if_exists: str = "replace",
) -> None:
    """Upload the data to BigQuery

    Args:
        df:         Dataframe to upload
        dataset_id: e.g. "kp_jln_bot"
        table_id:   e.g. "sample_table"
        project_id: e.g. "cet-stg"
        if_exists:  Behavior when the destination table exists. Value can be one of:
            ``'replace'`` (default)
                If table exists, drop it, recreate it, and insert data.
            ``'append'``
                If table exists, insert data. Create if does not exist.
            ``'fail'``
                If table exists raise pandas_gbq.gbq.TableCreationError.

    """
    if if_exists not in ("replace", "append", "fail"):
        raise ValueError(
            f"unsupported value: '{if_exists}'. please choose from 'replace', 'append', or 'fail'"
        )

    logger.info(f"uploading to '{project_id}:{dataset_id}.{table_id}' ...")
    df.to_gbq(f"{dataset_id}.{table_id}", project_id=project_id, if_exists=if_exists)
    logger.debug("done")
