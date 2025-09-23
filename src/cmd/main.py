"""
src/cmd/main.py - Common CLI handling
"""

import sys
import argparse
from loguru import logger
from src.config import Settings


def main_handler(settings: Settings, loglevel: str="INFO"):
    """Configure the main context"""

    # Configure the log level
    if loglevel not in ('TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'):
        logger.critical(f"Invalid log level {loglevel}")
        return -1
    logger.remove()
    logger.add(sys.stderr, level=loglevel)
    logger.debug(f"Log level set to {loglevel}")
    return 0


def create_main_parsers():
    """The primary factory for parsers
    """

    # The topmost parser
    parser = argparse.ArgumentParser(
        prog='epc',
        description="epc: The main entry point for EPIC",
        epilog="""""",
    )

    # args to handler bridge
    def bridge(parser, args: argparse.Namespace) -> int:
        # Reading the settings
        from src.config import get_settings
        settings = get_settings()
        setattr(args, "settings", settings)

        # Handle args
        ret = main_handler(
            settings,
            loglevel=args.loglevel,
        )
        if ret != 0:
            return ret

        # Handle the subcommand if provided
        if 'func' in args.__dir__() and args.func != None:
            return args.func(args)

        # Print help and exit
        parser.print_usage()
        return 0
    parser.set_defaults(main_func=bridge)

    # Top level arguments and options
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0'
    )

    # Set log level
    parser.add_argument(
        '--loglevel',
        default="INFO",
        help="set log level [TRACE|DEBUG|INFO|WARNING|ERROR|CRITICAL]"
    )

    #
    # The main subparser object
    #

    subparsers = parser.add_subparsers(
        title="Subcommands",
        description="Available subcommands",
        #help="use subcommand --help for details on each command"
    )

    # Return the main parsers
    return parser, subparsers


def register_subcommands(subparsers):
    """Register subcommands to the subparsers object"""
    #
    # TODO: Register all subcommands from this place
    # It should be simply importing "create_command_***" functions
    # and passing the "subparsers" object to the sub factories.
    # For convenience, we import each subcommand modules on the fly
    #

    # subcommand: 'chat'
    from .chat import create_command_chat
    create_command_chat(subparsers)

    return subparsers


def create_parser():
    """Creates the entirety of the argparse object"""
    parser, subparsers = create_main_parsers()
    register_subcommands(subparsers)
    return parser


def process_args():
    """Process arguments"""
    parser = create_parser()
    args = parser.parse_args()
    return args.main_func(parser, args)


def main():
    """The main entrypoint"""
    sys.exit(process_args())
