"""
src.cmd.cmd

The <CMD> command that prints out helpful documentation for each subcommand
"""
import argparse
from typing import Optional
from src.config import Settings


def cmd_handler(settings: Optional[Settings], var: str="default"):
    """Handler for the subcommand"""
    print(var)


def create_command_cmd(subparsers):
    """Define the 'cmd' subcommand"""

    # Preamble
    parser = subparsers.add_parser(
        "cmd",
        help="Execute the help subcommand",
    )

    def bridge(args: argparse.Namespace):
        """Bridge to convert args to a command call"""
        # TODO: implement the command handler
        cmd_handler(settings=args.settings, var=args.var)

    parser.set_defaults(func=bridge)

    # TODO: Define options and arguments
    parser.add_argument("var", default="default")
