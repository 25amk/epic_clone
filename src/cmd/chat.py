"""
src.cmd.chat

The <CMD> command that prints out helpful documentation for each subcommand
"""
import os
from loguru import logger
import argparse
from src.config import Settings
from typing import Optional


def chat_handler(settings: Optional[Settings]=None, host="127.0.0.1", port="3000", loglevel="INFO"):
    """Start the chat UI server via uvicorn for development"""

    # Check the current location
    from pathlib import Path
    if not (Path.cwd() / ".env").exists():
        logger.critical("Cannot find the environment file from the current working directory")
        exit(-1)
    if not (Path.cwd() / ".chainlit").exists():
        logger.critical("Cannot find the chainlit dot environment the current working directory")
        exit(-1)
    if not (Path.cwd() / "public").exists():
        logger.critical("Cannot find the public assets directory")
        exit(-1)

    # Run the server
    import uvicorn
    from src.server import app
    log_level = loglevel.lower()
    uvicorn.run(
        app=app,
        host=host,
        port=int(port),
        log_level=log_level,
    )


def create_command_chat(subparsers):
    """Define the 'cmd' subcommand"""

    # Preamble
    parser = subparsers.add_parser(
        "chat",
        help="Starts the chat server",
    )

    # Configure the args bridge
    def bridge(args: argparse.Namespace):
        """Bridge to convert args to a command call"""
        # TODO: implement the command handler
        chat_handler(settings=args.settings, host=args.host, port=args.port, loglevel=args.loglevel)
    parser.set_defaults(func=bridge)

    # Host and ports
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="3000")
