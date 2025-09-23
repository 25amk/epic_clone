"""
main_cmd.py - tests for src.cmd.*
"""
import pytest
import argparse
from unittest.mock import ANY



@pytest.mark.parametrize("loglevel", [
    ('TRACE'), ('DEBUG'), ('INFO'), ('SUCCESS'), ('WARNING'), ('ERROR'), ('CRITICAL'),
])
def test_main_parser_loglevel(mocker, argparse_setup, loglevel):
    """Test the existance of the loglevel setting option"""
    parse_args, subparsers = argparse_setup

    # Patch the command handler
    mock_hnd = mocker.patch('src.cmd.main.main_handler')
    parse_args([
        f"--loglevel={loglevel}"
    ])

    # Check the mock calls
    mock_hnd.assert_called_once_with(ANY, loglevel=loglevel)
