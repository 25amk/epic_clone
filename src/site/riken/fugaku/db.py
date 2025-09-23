"""
src/site/ornl/frontier/db_test.py - Frontier (ORNL) system specific data access methods
"""
import os, csv
from typing import List
from textwrap import dedent
from pathlib import Path
from sqlalchemy import create_engine, Engine
from langchain_core.prompts import ChatPromptTemplate
from src.config import get_settings
from src.tools.sql import JSONSQLDatabase
from src.sql.sql import make_table_description
import pandas as pd

settings = get_settings()

DATA_BASE = Path(settings.data_base)
META_BASE = Path(__file__).parent / "metadata"
FUGAKU_DATA = DATA_BASE / "riken/fugaku"


def import_all(conn):
    conn.exec_driver_sql(f"CREATE VIEW IF NOT EXISTS workload AS SELECT * FROM '{FUGAKU_DATA}/*.parquet';")


def create_sql_database():
    db_connect_args = {
        'read_only': False,
        'config': {
            'memory_limit': '2000mb'
        }
    }
    db_filepath = Path(settings.scratch_base) / "riken/fugaku/sql.db"
    db_filepath.parent.mkdir(parents=True, exist_ok=True)
    db_uri = f"duckdb:///{db_filepath}"

    engine = create_engine(db_uri, connect_args=db_connect_args)

    with engine.connect() as conn:
        with conn.begin():
            _ = import_all(conn)

    return JSONSQLDatabase(engine,
        max_result_length = settings.query_output_limit,
        view_support = True,
    )


def get_all_table_descriptions() -> str:
    result = []
    tables = [
        {
           'table_name': 'workload',
           'relationships': [],
        },
    ]
    for args in tables:
        metadata_file = META_BASE / f"desc_{args['table_name']}.csv"
        result.append(make_table_description(**args, metadata_file=metadata_file))
    return "\n".join(result)


def create_sql_prompt_template():
    prompt_template = ChatPromptTemplate([
        ("system", dedent("""
            You are an expert {dialect} SQL programmer.
        """).strip()),
        ("user", dedent("""
            ### Task
            Generate a SQL query to answer [question]{question}[/question]

            ### Instructions 
            - Create a syntactically correct {dialect} query to run.
            - If the question cannot be answered given the database schema, return "I do not know".
            - The given database schema is of Fugaku system.
            - Recall the date format is YYYY-MM-DD
            - Please do not use "TO_CHAR" or "DATE" sql function, instead use "DATE_TRUNC" funciton.
            - For AVG function remember it can not be executed for time interval data, instead first use DATEDIFF function and then then use AVG fucntion.
            - Timezone info of datetime fields are: tzinfo=<DstTzInfo 'America/New_York' EST-1 day, 19:00:00 STD>. Use CONVERT_TZ function to convert to UTC when comparing datetime fields.
            - Be careful of the upper bound and lower bound when comparing datetime fields.
            - Return only 'SQL' statement, no explanantion.

            ### Database Schema
            The query will run on a database with the following schema:
            {create_table_statements}

            ### Task
            Given the database schema, write a {dialect} SQL query that answers: {question}
        """).strip()),
        ("placeholder", "{conversation}"),
    ])

    return prompt_template.partial(
        create_table_statements = get_all_table_descriptions(),
        dialect = 'duckdb'
    )
