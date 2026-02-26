# CLUI v3.0 – Developer Guide

CLUI is a dependency‑free terminal UI toolkit for Python with optional Textual integration for draggable windows and mouse interactions. This guide is concise and developer‑focused.

---

## Install

### 1) Drop‑in (no pip)
Copy `clui.py` anywhere and import it directly.

### 2) Optional Textual (pip)
```bash
pip install clui[textual]
```

### 3) Custom installer (no pip, interactive)
```bash
python installer_ui.py
```
The installer defaults to `src/`, asks for a custom path, and can optionally install Textual into `src/vendor/`.

---

## Quick Start

```python
from clui import ui, C

print(ui.divider(" SYSTEM MONITOR ", color=C.CYN_B))
print(ui.panel("All systems nominal", title=" STATUS ", color=C.GRN, width="50%"))

headers = ["ID", "Service", "State"]
rows = [["1", "nginx", f"{C.GRN}Running{C.R}"], ["2", "redis", f"{C.GRN}Running{C.R}"]]
print(ui.table(headers, rows, color=C.CYN, align=["left","left","right"]))
```

## Method‑Based API (Recommended)

The `ui` object exposes fluent, method‑based access to the entire library. Each component has a `.render()` method and also implements `__str__`, so `print(component)` just works.

```python
from clui import ui, C

panel = ui.panel("Hello", title=" Greeting ", color=C.CYN).shadow()
print(panel)

cols = ui.columns([ui.panel("Left"), ui.panel("Right")], widths=["50%","50%"])
print(cols)
```

Additional fluent helpers:
- `ui.kvlist(items)`
- `ui.tree(data)`
- `ui.tabs(options)`
- `ui.breadcrumbs(path)`
- `ui.sparkline(data)`
- `ui.barchart(data)`
- `ui.progress(current, total)`
- `ui.markdown(text)`
- `ui.select(options).run()`
- `ui.confirm(message).run()`
- `ui.prompt(message).run()`
- `ui.notification(msg).run()`
- `ui.tooltip(text)`
- `ui.spinner(text)`
- `ui.progress_bar(total)`
- `ui.live()`
- `ui.fullscreen()`
- Class accessors: `ui.term()`, `ui.key()`, `ui.colors()`, `ui.symbols()`
- Constructors: `ui.theme(...)`, `ui.skin(name, ...)`
- Enums: `ui.border("rounded")`, `ui.align("center")`, `ui.layout("grid")`
- Textual: `ui.window(title)` / `ui.textual_app()` / `ui.run_textual_demo()`
- `ui.overlay(base, overlay, x, y)`
- `ui.shadow(content)` / `ui.shadow_box(content)`
- `ui.floating_box(content)`
- `ui.calendar(year=..., month=...)`
- `ui.grid(...)` / `ui.dashboard(...)`
- `ui.menu(options)`
- `ui.logger(name)` / `ui.cli(description)`
- `ui.set_theme(name)` / `ui.set_skin(name)` / `ui.set_unicode(bool)`
- `ui.fnum(...)` / `ui.ftime(...)` / `ui.fsize(...)` / `ui.fpercent(...)`

---

## Core Concepts

### Terminal Core (`Term`)
- `Term.width()`, `Term.height()`
- `Term.strip(text)`, `Term.vlen(text)`
- `Term.wrap(text, width)`
- `Term.get_key()` (single‑keypress input)
- Cursor utilities: `cursor_up`, `clear_line`, `hide_cursor`, `show_cursor`
- Raw input mode: `Term.raw_mode()`
- Unicode handling: `Term.supports_unicode()`, `set_unicode()`
- Performance: `Term.enable_vlen_cache()`, `Term.clear_vlen_cache()`

### Styling (`C`, `Sym`, `Theme`, `Skin`)
- Colors: `C.RED`, `C.GRN`, `C.CYN`, `C.WHT`, etc.
- Modifiers: `C.B`, `C.D`, `C.I`, `C.U`, `C.R`
- RGB: `C.rgb(r,g,b)` and `C.gradient(text, start, end)`
- Hyperlinks: `C.link(text, url)`
- Symbols: `Sym.CHECK`, `Sym.BULLET`, `Sym.ARROW_R`, `Sym.DOTS`, etc.
- Themes: `Theme`, `set_theme("Neon")`
- Skins (visual symbols/borders): `Skin`, `set_skin("ASCII")`, `register_skin(...)`

---

## Layout

### Box & Containers
```python
print(ui.panel("Content", title=" BOX ", width="50%", color=C.CYN).render())
print(ui.panel("Content", gradient=((255,0,0),(0,0,255)), gradient_dir="horizontal").render())
print(ui.panel("Shadowed", color=C.MGT).shadow().render())
```

Useful helpers:
- `box`, `box_gradient`, `shadow`, `shadow_box`
- `overlay`, `floating_box`
- `divider`, `hline`

### Columns & Responsive
- `ui.columns([...], widths=["30%","70%"]).render()`
- `responsive_cols(columns, threshold=80)`

### Advanced Layouts
- `Grid(columns=2, padding=2)`
- `Dashboard(mode=Layout.HORIZONTAL | Layout.VERTICAL | Layout.GRID)`

---

## Data & Visuals

### Tables
```python
print(ui.table(headers, rows, footers=footers, row_sep=True).render())
```

### Trees, Lists, Tabs
- `tree(data, label="root")`
- `kvlist(items)`
- `tabs(options, active_idx=0)`
- `breadcrumbs(path)`

### Charts
- `sparkline(data)`
- `barchart(data, width=40)`

---

## Interactive

### Prompts
- `select(options, multi=False, search=True)`
- `confirm(message, default=False)`
- `prompt(message, validator=None, password=False)`
- `Menu(options).run()`

### Notifications
- `notification(msg, title="Notification")`
- `tooltip(text, target_text="")`

---

## Live Components

- `Spinner(...)`
- `ProgressBar(...)`
- `progress(...)` (string helper)
- `Live()` (multi‑component updates)
- `Fullscreen()` (alternate buffer)
- `spinner_task()` (context manager)

---

## Utilities

- `markdown()` / `markdown_to_ansi()`
- `exception_box(e)`
- `success()`, `error()`, `warn()`, `info()`, `step()`, `bullet()`
- Formatters: `fsize`, `ftime`, `fnum`, `fpercent`, `ftrunc`
- `Logger`, `CLI`

---

## Optional Textual Integration

If Textual is installed, you can launch the demo:
```python
from clui import run_textual_demo
run_textual_demo()
```
This uses `DraggableWindow` for a draggable, buttoned window widget.

---

## API Index (Quick Reference)

Terminal:
- `Term`, `Key`, `Align`, `Layout`, `Border`, `Fullscreen`

Styling:
- `C`, `Sym`, `Theme`, `Skin`, `set_theme`, `set_skin`, `markdown`

Layout:
- Method-based: `ui` (fluent entry point)
- `ui.panel`, `ui.columns`, `ui.table`, `ui.divider`, `ui.hr`
- `ui.kvlist`, `ui.tree`, `ui.tabs`, `ui.breadcrumbs`
- `ui.sparkline`, `ui.barchart`, `ui.progress`, `ui.markdown`
 - `ui.overlay`, `ui.shadow`, `ui.shadow_box`, `ui.floating_box`, `ui.calendar`
 - `ui.grid`, `ui.dashboard`, `ui.menu`, `ui.logger`, `ui.cli`
 - `ui.set_theme`, `ui.set_skin`, `ui.set_unicode`
 - `ui.fnum`, `ui.ftime`, `ui.fsize`, `ui.fpercent`
- Functional: `box`, `box_gradient`, `shadow`, `shadow_box`, `overlay`, `floating_box`
- `cols`, `responsive_cols`, `Grid`, `Dashboard`

Data:
- `table`, `tree`, `kvlist`, `tabs`, `breadcrumbs`, `barchart`, `sparkline`

Interactive:
- `select`, `Menu`, `confirm`, `prompt`, `notification`, `tooltip`

Live:
- `Live`, `Spinner`, `ProgressBar`, `progress`, `spinner_task`

Utilities:
- `Logger`, `CLI`, `exception_box`, `success`, `error`, `warn`, `info`, `step`, `bullet`
- `fsize`, `ftime`, `fnum`, `fpercent`, `ftrunc`
