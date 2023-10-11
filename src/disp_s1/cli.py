#!/usr/bin/env python
import argparse
from typing import TYPE_CHECKING, Any, Optional, Sequence

from disp_s1 import __version__

if TYPE_CHECKING:
    _SubparserType = argparse._SubParsersAction[argparse.ArgumentParser]
else:
    _SubparserType = Any


def run(
    config_file: str,
    debug: bool = False,
) -> None:
    """Run the displacement workflow.

    Parameters
    ----------
    config_file : str
        YAML file containing the workflow options.
    debug : bool, optional
        Enable debug logging, by default False.
    """
    # rest of imports here so --help doesn't take forever
    from disp_s1.main import run
    from disp_s1.pge_runconfig import RunConfig

    pge_runconfig = RunConfig.from_yaml(config_file)
    cfg = pge_runconfig.to_workflow()
    run(cfg, pge_runconfig=pge_runconfig, debug=debug)


def get_parser(
    subparser: Optional[_SubparserType] = None, subcommand_name: str = "run"
) -> argparse.ArgumentParser:
    """Set up the command line interface."""
    metadata = dict(
        description="Run a displacement workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    if subparser:
        # Used by the subparser to make a nested command line interface
        parser = subparser.add_parser(subcommand_name, **metadata)  # type: ignore
    else:
        parser = argparse.ArgumentParser(**metadata)  # type: ignore

    # add a version command
    parser.add_argument("--version", action="version", version=__version__)

    # Main cli arguments
    parser.add_argument(
        "config_file",
        help="Name of YAML configuration file describing workflow options.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug messages to the log.",
    )
    parser.set_defaults(run_func=run)
    return parser


def main(args: Optional[Sequence[str]] = None) -> None:
    """Get the command line arguments and run the workflow."""
    from . import validate_product

    parser = argparse.ArgumentParser(
        prog=__package__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparser = parser.add_subparsers(title="subcommands")
    parser.add_argument("--version", action="version", version=__version__)

    get_parser(subparser, subcommand_name="run")
    # parser = get_parser()
    validate_product.get_parser(subparser)

    parsed_args = parser.parse_args(args)

    run(parsed_args.config_file, debug=parsed_args.debug)


if __name__ == "__main__":
    main()
