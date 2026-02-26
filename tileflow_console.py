from __future__ import annotations

import shlex
import sys
from typing import Sequence

from tileflow import cli as tileflow_cli


MENU_ITEMS: dict[str, str] = {
    "1": "list",
    "2": "tune",
    "3": "pull ",
    "4": "run ",
    "5": "serve ",
    "6": "backend show",
}


def _print_menu() -> None:
    print("\nTileFlow Console")
    print("1) list")
    print("2) tune")
    print("3) pull <repo_id>")
    print("4) run <model>")
    print("5) serve <model>")
    print("6) backend show")
    print("Type a TileFlow command directly, or type: help, menu, exit")


def _run_tileflow_command(args: Sequence[str]) -> int:
    original_argv = sys.argv[:]
    sys.argv = ["tileflow", *args]
    try:
        tileflow_cli.main()
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        return 1
    finally:
        sys.argv = original_argv
    return 0


def _interactive_loop() -> int:
    _print_menu()
    while True:
        raw = input("\ntileflow> ").strip()
        if not raw:
            continue
        lowered = raw.lower()
        if lowered in {"exit", "quit"}:
            return 0
        if lowered in {"help", "menu"}:
            _print_menu()
            continue

        mapped = MENU_ITEMS.get(raw)
        command_text = mapped if mapped is not None else raw

        if command_text.endswith(" "):
            print("Please complete the command arguments.")
            continue

        try:
            parsed = shlex.split(command_text)
        except ValueError as exc:
            print(f"Invalid command syntax: {exc}")
            continue

        code = _run_tileflow_command(parsed)
        if code != 0:
            print(f"Command exited with code {code}")


def main() -> int:
    if len(sys.argv) > 1:
        return _run_tileflow_command(sys.argv[1:])
    return _interactive_loop()


if __name__ == "__main__":
    raise SystemExit(main())
