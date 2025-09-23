"""
src/site/ornl/frontier/db_test.py - Frontier (ORNL) system specific data access methods
"""
import os
import pytest
from .sql import create_sql_qna_chain, make_table_description
from src.site.ornl.frontier.db import JOBSUMMARY_DATA, META_BASE
from src.models import mock_model


def no_datasets():
    """Condition that tests whether we have all datasets"""
    return not os.access(JOBSUMMARY_DATA, os.F_OK)


@pytest.mark.skipif(no_datasets(), reason="all datasets are not accessible")
def test_sql_qna_chain():
    model = mock_model({})
    chain = create_sql_qna_chain(model=model, db_module = "src.site.ornl.frontier.db")
    result = chain.invoke({"question": "What is the answer to life, the universe, and everything?"})


def test_make_table_description():
    """
    Test whether we can run the telemetry inference on a question
    """
    inputs = [
        {
           'table_name': 'jobstat',
           'relationships': ["table: 'project_description', key: 'account'"]
        },
        {
           'table_name': 'project_description',
           'relationships': []
        },
        {
           'table_name': 'scheduling_policy',
           'relationships': []
        },
    ]

    for args in inputs:
        metadata_file = META_BASE / f"desc_{args['table_name']}.csv"
        table_desc = make_table_description(**args, metadata_file = metadata_file)
        table_desc_lines = table_desc.split('\n')
        assert table_desc_lines is not None
        assert args['table_name'] in table_desc_lines[0]
        assert 'PRIMARY' in table_desc
        if args['relationships']:
            assert 'REFERENCES' in table_desc
        else:
            assert 'REFERENCES' not in table_desc

