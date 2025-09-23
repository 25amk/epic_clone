"""
Tests for src.cmd.chat
"""
from unittest.mock import ANY
from src.cmd import create_main_parsers


def test_handle_command(mocker, argparse_setup):
    """Tests the handle_command"""
    parse_args, subparsers = argparse_setup

    from src.cmd.chat import create_command_chat
    create_command_chat(subparsers)

    # Patch the command handler
    mock_hnd = mocker.patch('src.cmd.chat.chat_handler')
    parse_args([
        "chat", "--host=some_host_name", "--port=some_port",
    ])

    # Check the mock calls
    mock_hnd.assert_called_once_with(settings=ANY, host="some_host_name", port="some_port", loglevel="INFO")
