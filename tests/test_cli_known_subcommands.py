"""Regression: every argparse subcommand must be excluded from natural-language
routing (misterdev/cli.py's main()), or its first arg is treated as a build goal
instead of reaching argparse — caught when `misterdev evolve --help` silently
routed through nl_cli and attempted a real build."""

from misterdev import cli


def test_evolve_is_excluded_from_natural_language_routing():
    assert "evolve" in cli.KNOWN_SUBCOMMANDS


def test_known_subcommands_matches_actual_cli_registration():
    import re

    src = open(cli.__file__, encoding="utf-8").read()
    registered = set(re.findall(r'subparsers\.add_parser\(\s*"([^"]+)"', src))
    assert registered, "no subparsers found in cli.py — parsing regex is stale"
    assert registered <= cli.KNOWN_SUBCOMMANDS, (
        f"subcommand(s) {registered - cli.KNOWN_SUBCOMMANDS} registered via "
        "add_parser but missing from KNOWN_SUBCOMMANDS — their first arg will "
        "be swallowed by natural-language routing instead of reaching argparse"
    )
