"""Command-line interface for syncscope."""

from __future__ import annotations

import argparse

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="syncscope",
        description="Audio-visual synchronization and active-speaker detection.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_subparsers(dest="command")
    # subcommands are registered here as they are implemented
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 1
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
