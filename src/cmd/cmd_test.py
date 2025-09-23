"""
Tests for src.cmd.*

Template for the testing approach for subcommands.
Unless this is an integration test, we only test the behavior of parsing
Converting CLI calls from the shell to the command handler calls
"""
from unittest.mock import ANY
from src.cmd import create_main_parsers


def test_handle_command(mocker, argparse_setup):
    """Tests the handle_command"""
    parse_args, subparsers = argparse_setup

    from src.cmd.cmd import create_command_cmd
    create_command_cmd(subparsers)

    # Patch the command handler
    mock_hnd = mocker.patch('src.cmd.cmd.cmd_handler')
    parse_args([
        "cmd", "commands"
    ])

    # Check the mock calls
    mock_hnd.assert_called_once_with(settings=ANY, var="commands")