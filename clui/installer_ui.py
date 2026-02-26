#!/usr/bin/env python3
"""
CLUI dependency-free installer UI.
Double-click to run, answers are collected in-terminal.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import clui as ui
from clui import C


ROOT = Path(__file__).resolve().parent
DEFAULT_TARGET = ROOT / "src"


def _file_url(path: Path) -> str:
    return path.resolve().as_uri()


def _copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def main() -> int:
    ui.Term.clear()
    banner = f"{C.CYN_B}CLUI Installer{C.R}\n{C.D}Dependency-free setup UI{C.R}"
    print(ui.box(banner, width=60, style=ui.Border.DOUBLE, align=ui.Align.CENTER))
    print()

    print(f"{C.B}Install path:{C.R}")
    print(f"{C.D}Press Enter for default: {DEFAULT_TARGET}{C.R}")
    target_input = input("> ").strip()
    target_path = Path(target_input) if target_input else DEFAULT_TARGET

    print()
    print(f"{C.B}Textual integration?{C.R}")
    print(f"{C.D}If enabled, this will install dependencies using pip.{C.R}")
    use_textual = ui.confirm("Install Textual integration?", default=False).run()

    # Copy core files
    existing_install = (target_path / "clui.py").exists()
    if existing_install:
        print(f"{C.YLW}Existing CLUI install detected. Reinstalling...{C.R}")
    target_path.mkdir(parents=True, exist_ok=True)
    try:
        _copy_file(ROOT / "clui.py", target_path / "clui.py")
        if (ROOT / "GUIDE.md").exists():
            _copy_file(ROOT / "GUIDE.md", target_path / "GUIDE.md")
    except Exception as e:
        ui.error(f"Failed to copy files: {e}")
        return 1

    # Optional: pip install textual into local folder
    if use_textual:
        print()
        print(f"{C.D}Installing Textual into {target_path}...{C.R}")
        try:
            subprocess.run(
                [
                    sys.executable, "-m", "pip", "install",
                    "textual", "-t", str(target_path / "vendor"),
                    "--upgrade", "--force-reinstall"
                ],
                check=True
            )
            print(f"{C.GRN}Textual installed.{C.R}")
        except Exception as e:
            ui.error(f"Textual install failed: {e}")

    print()
    print(ui.divider(" INSTALL COMPLETE ", color=C.GRN))

    api_summary = "\n".join([
        "Core UI: box, table, cols, divider, hline",
        "Layouts: Grid, Dashboard, tabs, breadcrumbs",
        "Data viz: sparkline, barchart, progress",
        "Live: ProgressBar, Spinner, Live",
        "Prompts: select, confirm, prompt",
        "Utilities: markdown, tree, kvlist"
    ])
    print()
    print(ui.box(api_summary, title=" API Summary ", width=70, color=C.CYN))

    guide_path = target_path / "GUIDE.md"
    if guide_path.exists():
        link = C.link("Click here to view the full API docs", _file_url(guide_path))
        print()
        print(f"{C.B}Full API Docs:{C.R} {link}")
        print(f"{C.D}{guide_path}{C.R}")
    else:
        print(f"{C.YLW}GUIDE.md was not found to display.{C.R}")

    print()
    input(f"{C.D}Press Enter to exit...{C.R}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
