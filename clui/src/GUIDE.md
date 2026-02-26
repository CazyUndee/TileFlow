# CLUI: The Modern Python Toolkit for Terminal UI (v3.0)

CLUI is a comprehensive, feature-rich library designed for building beautiful, responsive, and interactive terminal interfaces in Python. It bridges the gap between simple script output and complex text-based user interfaces (TUI) by providing a high-level API for layout, styling, and data visualization.

---

## 📋 Table of Contents

1.  [Core Philosophy](#-core-philosophy)
2.  [Installation & Setup](#-installation--setup)
3.  [Getting Started: Your First App](#-getting-started-your-first-app)
4.  [Core Modules](#-core-modules)
    -   [Terminal Core (Term)](#terminal-core-term)
    -   [Colors & Styling (C)](#colors--styling-c)
    -   [Symbols & Icons (Sym)](#symbols--icons-sym)
5.  [Layout System](#-layout-system)
    -   [Boxes & Containers](#boxes--containers)
    -   [Advanced Layouts (Grid, Dashboard)](#advanced-layouts)
    -   [Columns & Responsiveness](#columns--responsiveness)
6.  [Data Visualization](#-data-visualization)
7.  [Interactive Components](#-interactive-components)
8.  [Live Updates & Animations](#-live-updates--animations)
9.  [Theming](#-theming)
10. [Utilities & Helpers](#-utilities--helpers)
11. [API Reference Index](#-api-reference-index)

---

## 🌟 Core Philosophy

-   **ANSI-Awareness**: All components understand ANSI escape codes. Length calculations (`Term.vlen`) and text wrapping (`Term.wrap`) ignore non-printable characters, ensuring pixel-perfect layouts even with heavy styling.
-   **Responsiveness**: Built-in support for percentage-based widths (e.g., `width="50%"`) and adaptive layouts that change based on terminal size.
-   **No Dependencies**: Core functionality requires only the Python standard library. Optional features like clipboard support in prompts use `pyperclip`.
-   **Theming First**: Change the entire look and feel of your application with a single line of code.

---

## 🚀 Installation & Setup

### Requirements
- Python 3.8+
- `pyperclip` (Optional, for clipboard support in interactive prompts)

### Installation
```bash
pip install clui-toolkit
```

---

## 🏁 Getting Started: Your First App

Building a basic dashboard with CLUI is straightforward:

```python
import clui as ui
from clui import C, Term, Align

def main():
    Term.clear()
    
    # 1. Header
    print(ui.divider(" SYSTEM MONITOR v1.0 ", color=C.CYN_B))
    
    # 2. Stats in columns
    stats = [
        ("CPU", "45%", C.GRN),
        ("MEM", "2.4GB", C.YLW),
        ("DISK", "80%", C.RED)
    ]
    
    cols_data = [
        ui.box(f"{C.B}{val}{C.R}", title=f" {label} ", color=color, width="100%")
        for label, val, color in stats
    ]
    print(ui.cols(cols_data))
    
    # 3. Data Table
    headers = ["ID", "Process", "Status"]
    rows = [
        [1, "nginx", f"{C.GRN}Running{C.R}"],
        [2, "redis", f"{C.GRN}Running{C.R}"],
        [3, "postgres", f"{C.RED}Stopped{C.R}"]
    ]
    print(ui.table(headers, rows, color=C.BLU))

if __name__ == "__main__":
    main()
```

---

## 🖥️ Core Modules

### Terminal Core (Term)
The `Term` class handles terminal detection, ANSI stripping, and raw input.

-   `Term.width()` / `Term.height()`: Current dimensions.
-   `Term.vlen(text)`: Visible length (ignores ANSI).
-   `Term.strip(text)`: Removes all ANSI codes.
-   `Term.wrap(text, width)`: Smart wrapping that preserves ANSI state.
-   `Term.get_key()`: Blocking call to read a single key (supports arrows, esc, enter, etc.).
-   `Term.cursor_up(n)`, `Term.clear_line()`, `Term.hide_cursor()`: Cursor control.

### Colors & Styling (C)
The `C` class provides easy access to colors and advanced effects.

-   **Standard**: `C.RED`, `C.GRN`, `C.BLU`, `C.YLW`, `C.MGT`, `C.CYN`, `C.WHT`, `C.BLK`.
-   **Bright**: `C.RED_B`, `C.GRN_B`, etc.
-   **Backgrounds**: `C.BG_RED`, `C.BG_GRN`, etc.
-   **Modifiers**: `C.B` (Bold), `C.D` (Dim), `C.I` (Italic), `C.U` (Underline), `C.R` (Reset).
-   **RGB**: `C.rgb(r, g, b)` for 24-bit color.
-   **Gradients**: `C.gradient("Hello", (255,0,0), (0,0,255))`.
-   **Links**: `C.link("GitHub", "https://github.com")`.

### Symbols & Icons (Sym)
Predefined ASCII/Unicode symbols for consistent UI design.

-   `Sym.CHECK`, `Sym.CROSS`, `Sym.WARN`, `Sym.INFO`
-   `Sym.BULLET`, `Sym.TRIANGLE`, `Sym.ARROW_R`
-   `Sym.DOTS` (Spinner frames), `Sym.BLOCKS` (Progress bar frames)

---

## 🏗️ Layout System

CLUI uses a flexible layout system that supports nesting and responsive behavior.

### Boxes & Containers
The `box` function is the foundation of the UI.

```python
print(ui.box(
    "Content goes here",
    title=" BOX TITLE ",
    width="50%",
    color=C.CYN,
    style=Border.ROUNDED,
    align=Align.CENTER,
    padding=1
))
```

-   **Styles**: `Border.ROUNDED`, `Border.SHARP`, `Border.DOUBLE`, `Border.HEAVY`, `Border.BLOCK`, `Border.NONE`.
-   **Containers**:
    -   `ui.shadow(content)`: Adds a drop-shadow effect.
    -   `ui.overlay(base, overlay, x, y)`: Places one piece of content over another.
    -   `ui.floating_box(content)`: Centers a box both horizontally and vertically.

### Columns & Responsiveness
-   **`cols(columns, spacing=2, widths=None)`**: Side-by-side components. `widths` can be a list of percentages like `["30%", "70%"]`.
-   **`responsive_cols(columns, threshold=80)`**: Automatically stacks columns vertically if the terminal width is less than `threshold`.

### Advanced Layouts
-   **`Grid(columns=2, padding=2)`**: Automatically arranges multiple items into a grid.
-   **`Dashboard(mode=Layout.HORIZONTAL)`**: A high-level layout manager.
    ```python
    db = ui.Dashboard()
    db.add(component_a, title="Stats", weight=1)
    db.add(component_b, title="Logs", weight=2)
    print(db.render())
    ```

---

## 📊 Data Visualization

### Tables
Powerful tables with auto-wrapping and column alignment.

```python
headers = ["Item", "Price", "Stock"]
rows = [["Apple", "$1.00", "50"], ["Banana", "$0.50", "100"]]
print(ui.table(
    headers, 
    rows, 
    alignments=[Align.LEFT, Align.RIGHT, Align.CENTER],
    color=C.CYN
))
```

### Trees & Lists
-   **`tree(data, label="root")`**: Renders a dictionary or list as a hierarchical tree.
-   **`kvlist(items)`**: Formats a dictionary into an aligned key-value list.
-   **`tabs(options, selected_index)`**: Renders a horizontal tab bar.

### Charts
-   **`barchart(data, width=40)`**: Renders horizontal bars for numeric data.
-   **`sparkline(data)`**: A tiny inline trend line (e.g., `▂▄▆█`).

---

## ⌨️ Interactive Components

### Menus & Selections
-   **`select(options, title, multi=False, search=True)`**: An interactive menu with:
    -   Arrow and Vim-key (`j`/`k`) navigation.
    -   Type-ahead fuzzy filtering.
    -   Multi-select support (space to toggle).

### Prompts
-   **`confirm(message, default=True)`**: Simple Yes/No prompt.
-   **`prompt(message, validator=None, password=False)`**: Advanced text input with:
    -   Real-time validation.
    -   Input history (Up/Down arrows).
    -   Clipboard support (`Ctrl+V` on Linux/macOS, built-in on Windows).

### Notifications & Tooltips
-   **`notification(msg, title)`**: A temporary floating message that disappears after a duration.
-   **`tooltip(text, target_text)`**: Wraps text with a dim hint.

---

## 🔄 Live Updates & Animations

### The `Live` Context
The `Live` class is used for flicker-free updates of multiple components.

```python
with ui.Live() as live:
    for i in range(100):
        live.update(ui.cols([
            ui.ProgressBar(100).render(i),
            ui.Spinner().render(f"Step {i}")
        ]))
        time.sleep(0.05)
```

### Specialized Components
-   **`Spinner(label, type="DOTS")`**: Animated loading indicators.
-   **`ProgressBar(total, label)`**: Visual progress tracking.
-   **`Fullscreen()`**: A context manager that switches to the alternate terminal buffer and restores the original state on exit.

---

## 🎨 Theming

CLUI supports global theming via the `Theme` class.

### Built-in Themes
Switch themes with `ui.set_theme("Neon")`. Available: `"Default"`, `"Neon"`, `"Ocean"`, `"Monochrome"`, `"Sunset"`.

### Custom Themes
```python
my_theme = ui.Theme(
    primary=C.rgb(200, 100, 255),
    border=C.D,
    success=C.GRN_B
)
ui.set_theme(my_theme)
```

---

## 🛠️ Utilities & Helpers

### Structured Logging
The `Logger` class provides timestamped, color-coded logs.
```python
log = ui.Logger("API")
log.info("Request started", url="https://api.com")
log.success("Data received")
```

### CLI Integration
The `CLI` class simplifies command-line argument handling.
```python
app = ui.CLI(description="My App")

@app.command("run", help_text="Start the service")
def run_cmd():
    print("Running...")

app.run()
```

### Formatters
-   `fsize(bytes)`: Formats bytes to "1.2 MB".
-   `ftime(seconds)`: Formats duration to "2h 15m".
-   `fnum(n)`: Adds thousands separators.
-   `fpercent(v, total)`: Formats as "45.0%".

---

## 📚 API Reference Index

| Category | Functions / Classes |
| :--- | :--- |
| **Terminal** | `Term`, `Key`, `Fullscreen`, `Align`, `Layout`, `Border` |
| **Styling** | `C`, `Sym`, `Theme`, `set_theme`, `markdown`, `gradient` |
| **Containers** | `box`, `shadow`, `overlay`, `floating_box`, `divider`, `hline` |
| **Layout** | `cols`, `responsive_cols`, `Grid`, `Dashboard` |
| **Data Viz** | `table`, `tree`, `barchart`, `sparkline`, `kvlist`, `tabs` |
| **Interactive** | `select`, `confirm`, `prompt`, `notification`, `tooltip` |
| **Live** | `Live`, `Spinner`, `ProgressBar` |
| **App Logic** | `CLI`, `Logger`, `exception_box`, `success`, `error`, `warn`, `info`, `step`, `bullet` |
| **Formatters** | `fnum`, `ftime`, `fsize`, `fpercent`, `ftrunc` |
