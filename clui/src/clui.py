#!/usr/bin/env python3
"""
Ultimate CLI Library v3.0
A comprehensive toolkit for building beautiful terminal interfaces.
"""

import sys
import os
import shutil
import time
import re
import threading
import itertools
import signal
import datetime
import traceback
import calendar
import functools
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from enum import Enum, auto

try:
    from textual.app import App, ComposeResult
    from textual.containers import Container
    from textual.widgets import Static, Button
    from textual.events import MouseDown, MouseMove, MouseUp
    _TEXTUAL_AVAILABLE = True
except Exception:
    _TEXTUAL_AVAILABLE = False

# Platform specific imports for raw input
if os.name == 'nt':
    import msvcrt
else:
    import tty
    import termios

# ------------------------------------------------------------------------------
# 1. TERMINAL CORE
# ------------------------------------------------------------------------------

class Key:
    """Key constants for input handling."""
    UP        = "up"
    DOWN      = "down"
    LEFT      = "left"
    RIGHT     = "right"
    HOME      = "home"
    END       = "end"
    PAGE_UP   = "page_up"
    PAGE_DOWN = "page_down"
    ENTER     = "enter"
    ESC       = "esc"
    TAB       = "tab"
    BACKSPACE = "backspace"
    SPACE     = "space"
    DELETE    = "delete"
    
    # Vim keys
    H = "h"
    J = "j"
    K = "k"
    L = "l"

class Term:
    """Terminal detection, ANSI handling, and core utilities."""
    
    _ANSI_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    
    # Initialize at import time
    is_tty: bool = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    colors: bool = is_tty
    unicode: bool = True
    
    # Windows ANSI support
    if is_tty and os.name == 'nt':
        try:
            import ctypes
            _k32 = ctypes.windll.kernel32
            _k32.SetConsoleMode(_k32.GetStdHandle(-11), 7)
        except Exception:
            colors = False
    
    _resize_handlers: List[Callable[[int, int], None]] = []
    _vlen_cache_enabled: bool = True

    @staticmethod
    @functools.lru_cache(maxsize=4096)
    def _strip_cached(text: str) -> str:
        return Term._ANSI_RE.sub('', str(text))

    @classmethod
    def supports_unicode(cls) -> bool:
        """Best-effort check for Unicode output support."""
        enc = getattr(sys.stdout, "encoding", None) or ""
        return "UTF" in enc.upper()

    @classmethod
    def width(cls) -> int:
        """Terminal width in columns."""
        return shutil.get_terminal_size().columns

    @classmethod
    def height(cls) -> int:
        """Terminal height in rows."""
        return shutil.get_terminal_size().lines

    @classmethod
    def on_resize(cls, handler: Callable[[int, int], None]):
        """Register a callback for terminal resize events."""
        if not cls._resize_handlers:
            if os.name != 'nt':
                def signal_handler(sig, frame):
                    w, h = cls.width(), cls.height()
                    for hnd in cls._resize_handlers:
                        hnd(w, h)
                signal.signal(signal.SIGWINCH, signal_handler)
            # Windows doesn't have SIGWINCH, but we can't easily poll here
            # without a main loop. We'll just register it for now.
        cls._resize_handlers.append(handler)
    
    @classmethod
    def strip(cls, text: str) -> str:
        """Remove all ANSI escape codes from text."""
        if cls._vlen_cache_enabled:
            return cls._strip_cached(str(text))
        return cls._ANSI_RE.sub('', str(text))
    
    @classmethod
    def vlen(cls, text: str) -> int:
        """Visible length of text (excluding ANSI codes)."""
        return len(cls.strip(text))

    @classmethod
    def enable_vlen_cache(cls, enabled: bool = True):
        """Enable or disable ANSI-aware length caching."""
        cls._vlen_cache_enabled = enabled

    @classmethod
    def clear_vlen_cache(cls):
        """Clear cached ANSI-aware length calculations."""
        cls._strip_cached.cache_clear()
    
    @classmethod
    def truncate(cls, text: str, max_len: int, suffix: str = "...") -> str:
        """Truncate text to max visible length, preserving ANSI codes."""
        if cls.vlen(text) <= max_len:
            return text
        
        result = []
        visible = 0
        target = max(0, max_len - len(suffix))
        i = 0
        
        while i < len(text) and visible < target:
            match = cls._ANSI_RE.match(text[i:])
            if match:
                result.append(match.group())
                i += len(match.group())
            else:
                result.append(text[i])
                visible += 1
                i += 1
        
        return ''.join(result) + suffix + ("\033[0m" if cls.colors else "")
    
    @classmethod
    def pad(cls, text: str, width: int, align: 'Align' = None, char: str = " ") -> str:
        """Pad text to width, accounting for ANSI codes."""
        if align is None:
            align = Align.LEFT
        visible = cls.vlen(text)
        padding = max(0, width - visible)
        
        if align == Align.LEFT:
            return text + char * padding
        elif align == Align.RIGHT:
            return char * padding + text
        else:  # CENTER
            left = padding // 2
            right = padding - left
            return char * left + text + char * right
    
    @classmethod
    def wrap(cls, text: str, width: int) -> List[str]:
        """
        Word-wrap text to width while preserving ANSI codes.
        Maintains style state across line breaks.
        """
        if width <= 0:
            return [text] if text else [""]
        
        lines = []
        
        for paragraph in text.split('\n'):
            if not paragraph:
                lines.append("")
                continue
            
            words = paragraph.split(' ')
            current_line = []
            current_len = 0
            active_codes = []
            
            for word in words:
                word_visible = cls.vlen(word)
                
                # Extract ANSI codes from word to track state
                for match in cls._ANSI_RE.finditer(word):
                    code = match.group()
                    if code == "\033[0m":
                        active_codes = []
                    else:
                        active_codes.append(code)
                
                # Handle words longer than width
                if word_visible > width:
                    if current_line:
                        lines.append(''.join(current_line))
                        current_line = []
                        current_len = 0
                    
                    # Split the word into chunks of width
                    prefix = ''.join(active_codes)
                    temp_word = word
                    while cls.vlen(temp_word) > width:
                        # Find cut point in ANSI-aware way
                        cut = 0
                        vis = 0
                        while vis < width and cut < len(temp_word):
                            m = cls._ANSI_RE.match(temp_word[cut:])
                            if m:
                                cut += len(m.group())
                            else:
                                cut += 1
                                vis += 1
                        lines.append(prefix + temp_word[:cut])
                        temp_word = temp_word[cut:]
                    
                    current_line = [prefix + temp_word] if prefix else [temp_word]
                    current_len = cls.vlen(temp_word)
                    continue

                if current_len + word_visible + (1 if current_line else 0) <= width:
                    if current_line:
                        current_line.append(' ')
                        current_len += 1
                    current_line.append(word)
                    current_len += word_visible
                else:
                    if current_line:
                        lines.append(''.join(current_line))
                    # Start new line with active ANSI codes
                    prefix = ''.join(active_codes)
                    current_line = [prefix + word] if prefix else [word]
                    current_len = word_visible
            
            if current_line:
                lines.append(''.join(current_line))
        
        return lines if lines else [""]
    
    @classmethod
    def clear(cls):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @classmethod
    def clear_line(cls):
        """Clear current line and move cursor to start."""
        if cls.is_tty:
            sys.stdout.write("\033[2K\r")
            sys.stdout.flush()
    
    @classmethod
    def cursor_up(cls, n: int = 1):
        """Move cursor up n lines."""
        if cls.is_tty:
            sys.stdout.write(f"\033[{n}A")
            sys.stdout.flush()
    
    @classmethod
    def hide_cursor(cls):
        """Hide the cursor."""
        if cls.is_tty:
            sys.stdout.write("\033[?25l")
            sys.stdout.flush()
    
    @classmethod
    def show_cursor(cls):
        """Show the cursor."""
        if cls.is_tty:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()

    @classmethod
    def alt_buffer(cls, enable: bool):
        """Switch to alternate terminal buffer."""
        if cls.is_tty:
            sys.stdout.write("\033[?1049h" if enable else "\033[?1049l")
            sys.stdout.flush()

    @classmethod
    def move_cursor(cls, row: int, col: int):
        """Move cursor to specific position (1-indexed)."""
        if cls.is_tty:
            sys.stdout.write(f"\033[{row};{col}H")
            sys.stdout.flush()

    @classmethod
    def parse_width(cls, width: Union[int, str], total_width: int = None) -> int:
        """Parse width as fixed integer or percentage string."""
        if total_width is None:
            total_width = cls.width()

        if isinstance(width, str):
            w = width.strip().lower()
            if w in ("auto", "fit", "min"):
                return 0
            if w.endswith('%'):
                try:
                    percent = float(w[:-1]) / 100.0
                    return int(total_width * percent)
                except ValueError:
                    return total_width
        
        if isinstance(width, str) and width.endswith('%'):
            try:
                percent = float(width[:-1]) / 100.0
                return int(total_width * percent)
            except ValueError:
                return total_width
        
        try:
            val = int(width)
            if val <= 0:
                return 0 # Auto-width signal
            return val
        except (ValueError, TypeError):
            return 0

    @classmethod
    def get_key(cls) -> str:
        """Read a single keypress from the user."""
        if not cls.is_tty:
            return ""

        if os.name == 'nt':
            ch = msvcrt.getch()
            if ch in (b'\x00', b'\xe0'):  # Function or arrow key
                ch2 = msvcrt.getch()
                mapping = {
                    b'H': Key.UP, b'P': Key.DOWN, b'K': Key.LEFT, b'M': Key.RIGHT,
                    b'G': Key.HOME, b'O': Key.END, b'I': Key.PAGE_UP, b'Q': Key.PAGE_DOWN,
                    b'S': Key.DELETE
                }
                return mapping.get(ch2, "")
            
            mapping = {
                b'\r': Key.ENTER, b'\x1b': Key.ESC, b'\t': Key.TAB,
                b'\x08': Key.BACKSPACE, b' ': Key.SPACE, b'\x16': "\x16"
            }
            if ch in mapping:
                return mapping[ch]
            
            try:
                return ch.decode('utf-8')
            except UnicodeDecodeError:
                return ""
        else:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                if ch == '\x1b':
                    # Read next two chars for escape sequences
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        ch3 = sys.stdin.read(1)
                        mapping = {
                            'A': Key.UP, 'B': Key.DOWN, 'C': Key.RIGHT, 'D': Key.LEFT,
                            'H': Key.HOME, 'F': Key.END, '5': Key.PAGE_UP, '6': Key.PAGE_DOWN,
                            '3': Key.DELETE
                        }
                        # Handle ~ in some sequences like 5~, 6~, 3~
                        if ch3 in ('5', '6', '3'):
                            sys.stdin.read(1) 
                        return mapping.get(ch3, "")
                    return Key.ESC
                
                mapping = {
                    '\r': Key.ENTER, '\n': Key.ENTER, '\t': Key.TAB,
                    '\x7f': Key.BACKSPACE, ' ': Key.SPACE, '\x16': "\x16"
                }
                return mapping.get(ch, ch)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    @classmethod
    @contextmanager
    def raw_mode(cls):
        """
        Context manager for raw input mode (POSIX).
        Lets power users manage their own input loops.
        """
        if not cls.is_tty or os.name == 'nt':
            yield
            return
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            yield
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


Term.unicode = Term.supports_unicode()


# Helper for color initialization
def _c(code: str) -> str:
    return code if Term.colors else ""


class C:
    """ANSI Color Palette - all codes auto-disable if terminal doesn't support colors."""
    
    # Reset & Modifiers
    R   = _c("\033[0m")   # Reset all
    B   = _c("\033[1m")   # Bold
    D   = _c("\033[2m")   # Dim
    I   = _c("\033[3m")   # Italic
    U   = _c("\033[4m")   # Underline
    S   = _c("\033[9m")   # Strikethrough
    
    # Standard Foreground (30-37)
    BLK = _c("\033[30m")
    RED = _c("\033[31m")
    GRN = _c("\033[32m")
    YLW = _c("\033[33m")
    BLU = _c("\033[34m")
    MGT = _c("\033[35m")
    CYN = _c("\033[36m")
    WHT = _c("\033[37m")
    
    # Bright Foreground (90-97)
    GRY   = _c("\033[90m")  # Bright black = gray
    RED_B = _c("\033[91m")
    GRN_B = _c("\033[92m")
    YLW_B = _c("\033[93m")
    BLU_B = _c("\033[94m")
    MGT_B = _c("\033[95m")
    CYN_B = _c("\033[96m")
    WHT_B = _c("\033[97m")
    
    # Backgrounds (40-47)
    BG_BLK = _c("\033[40m")
    BG_RED = _c("\033[41m")
    BG_GRN = _c("\033[42m")
    BG_YLW = _c("\033[43m")
    BG_BLU = _c("\033[44m")
    BG_MGT = _c("\033[45m")
    BG_CYN = _c("\033[46m")
    BG_WHT = _c("\033[47m")
    
    @staticmethod
    def rgb(r: int, g: int, b: int, bg: bool = False) -> str:
        """Generate 24-bit RGB color code."""
        if not Term.colors:
            return ""
        return f"\033[{48 if bg else 38};2;{r};{g};{b}m"
    
    @staticmethod
    def c256(n: int, bg: bool = False) -> str:
        """Generate 256-color palette code (0-255)."""
        if not Term.colors:
            return ""
        return f"\033[{48 if bg else 38};5;{n}m"

    @staticmethod
    def gradient(text: str, start_rgb: Tuple[int, int, int], end_rgb: Tuple[int, int, int]) -> str:
        """Apply a linear RGB gradient to text."""
        if not Term.colors:
            return text
        
        chars = list(text)
        if not chars:
            return ""
        
        result = []
        n = len(chars)
        for i, char in enumerate(chars):
            r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * i / (n - 1)) if n > 1 else start_rgb[0]
            g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * i / (n - 1)) if n > 1 else start_rgb[1]
            b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * i / (n - 1)) if n > 1 else start_rgb[2]
            result.append(f"{C.rgb(r, g, b)}{char}")
        
        return "".join(result) + C.R

    @staticmethod
    def link(text: str, url: str) -> str:
        """Create a terminal hyperlink (if supported)."""
        if not Term.is_tty:
            return text
        return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"


def _gradient_segment(text: str, start_rgb: Tuple[int, int, int], end_rgb: Tuple[int, int, int], start_pos: int, total_len: int) -> str:
    """Apply a horizontal gradient to a text segment using global line position."""
    if not Term.colors:
        return text
    if total_len <= 1:
        return f"{C.rgb(start_rgb[0], start_rgb[1], start_rgb[2])}{text}{C.R}"
    result = []
    for i, ch in enumerate(text):
        pos = start_pos + i
        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * pos / (total_len - 1))
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * pos / (total_len - 1))
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * pos / (total_len - 1))
        result.append(f"{C.rgb(r, g, b)}{ch}")
    return "".join(result) + C.R


class Theme:
    """
    Color themes for the library.
    """
    def __init__(
        self,
        primary: str = C.CYN,
        secondary: str = C.MGT,
        success: str = C.GRN,
        warning: str = C.YLW,
        error: str = C.RED,
        info: str = C.CYN,
        dim: str = C.D,
        bold: str = C.B,
        border: str = C.D,
        title: str = C.CYN_B
    ):
        self.primary = primary
        self.secondary = secondary
        self.success = success
        self.warning = warning
        self.error = error
        self.info = info
        self.dim = dim
        self.bold = bold
        self.border = border
        self.title = title

# Predefined Themes
THEMES = {
    "Default": Theme(),
    "Neon": Theme(
        primary=C.rgb(255, 0, 255), 
        secondary=C.rgb(0, 255, 255),
        success=C.rgb(0, 255, 0),
        title=C.rgb(255, 255, 0) + C.B
    ),
    "Ocean": Theme(
        primary=C.BLU,
        secondary=C.CYN,
        success=C.GRN,
        title=C.BLU_B + C.B
    ),
    "Monochrome": Theme(
        primary=C.WHT_B,
        secondary=C.WHT,
        success=C.WHT_B,
        warning=C.WHT,
        error=C.WHT_B,
        info=C.WHT,
        title=C.WHT_B + C.U
    ),
    "Sunset": Theme(
        primary=C.rgb(255, 87, 34),
        secondary=C.rgb(255, 193, 7),
        success=C.rgb(76, 175, 80),
        title=C.rgb(255, 152, 0) + C.B
    ),
    "Dracula": Theme(
        primary=C.rgb(189, 147, 249),
        secondary=C.rgb(255, 121, 198),
        success=C.rgb(80, 250, 123),
        warning=C.rgb(241, 250, 140),
        error=C.rgb(255, 85, 85),
        info=C.rgb(139, 233, 253),
        border=C.rgb(68, 71, 90),
        title=C.rgb(189, 147, 249) + C.B
    ),
    "Gruvbox": Theme(
        primary=C.rgb(215, 153, 33),
        secondary=C.rgb(184, 187, 38),
        success=C.rgb(152, 151, 26),
        warning=C.rgb(250, 189, 47),
        error=C.rgb(204, 36, 29),
        info=C.rgb(131, 165, 152),
        border=C.rgb(146, 131, 116),
        title=C.rgb(215, 153, 33) + C.B
    ),
    "Nord": Theme(
        primary=C.rgb(136, 192, 208),
        secondary=C.rgb(129, 161, 193),
        success=C.rgb(163, 190, 140),
        warning=C.rgb(235, 203, 139),
        error=C.rgb(191, 97, 106),
        info=C.rgb(143, 188, 187),
        border=C.rgb(76, 86, 106),
        title=C.rgb(136, 192, 208) + C.B
    ),
    "Solarized": Theme(
        primary=C.rgb(38, 139, 210),
        secondary=C.rgb(42, 161, 152),
        success=C.rgb(133, 153, 0),
        warning=C.rgb(181, 137, 0),
        error=C.rgb(220, 50, 47),
        info=C.rgb(38, 139, 210),
        border=C.rgb(88, 110, 117),
        title=C.rgb(38, 139, 210) + C.B
    ),
    "Aycue": Theme(
        primary=C.rgb(255, 173, 70),
        secondary=C.rgb(240, 113, 120),
        success=C.rgb(195, 232, 141),
        warning=C.rgb(255, 203, 107),
        error=C.rgb(255, 102, 114),
        info=C.rgb(57, 186, 230),
        border=C.rgb(71, 71, 71),
        title=C.rgb(255, 173, 70) + C.B
    )
}

CURRENT_THEME = THEMES["Default"]

def set_theme(name: str):
    global CURRENT_THEME
    if name in THEMES:
        CURRENT_THEME = THEMES[name]


# ------------------------------------------------------------------------------
# 2. SYMBOLS & ICONS
# ------------------------------------------------------------------------------

class Sym:
    """Symbols and icons for CLI interfaces."""
    
    # Status (ASCII to avoid emojis)
    CHECK    = "v"
    CROSS    = "x"
    WARN     = "!"
    INFO     = "i"
    QUESTION = "?"
    
    # Bullets & Markers
    BULLET   = "•"
    DIAMOND  = "◆"
    STAR     = "★"
    STAR_O   = "☆"
    CIRCLE   = "●"
    CIRCLE_O = "○"
    SQUARE   = "■"
    SQUARE_O = "□"
    TRIANGLE = "▶"
    
    # Arrows
    ARROW_R = "→"
    ARROW_L = "←"
    ARROW_U = "↑"
    ARROW_D = "↓"
    ARROW_DR = "└─"
    
    # Lines (Unicode for connected look)
    H_LINE     = "─"
    H_LINE_DBL = "═"
    H_LINE_HVY = "━"
    V_LINE     = "│"
    V_LINE_DBL = "║"
    V_LINE_HVY = "┃"
    
    # Other
    ELLIPSIS = "…"
    MIDDOT   = "·"
    
    # Spinner Frames
    DOTS   = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    LINES  = ("│", "╱", "─", "╲")
    ARROWS = ("←", "↖", "↑", "↗", "→", "↘", "↓", "↙")
    BLOCKS = ("░", "▒", "▓", "█")
    BOUNCE = ("⠁", "⠂", "⠄", "⠂")

    _UNICODE = {
        "CHECK": CHECK,
        "CROSS": CROSS,
        "WARN": WARN,
        "INFO": INFO,
        "QUESTION": QUESTION,
        "BULLET": BULLET,
        "DIAMOND": DIAMOND,
        "STAR": STAR,
        "STAR_O": STAR_O,
        "CIRCLE": CIRCLE,
        "CIRCLE_O": CIRCLE_O,
        "SQUARE": SQUARE,
        "SQUARE_O": SQUARE_O,
        "TRIANGLE": TRIANGLE,
        "ARROW_R": ARROW_R,
        "ARROW_L": ARROW_L,
        "ARROW_U": ARROW_U,
        "ARROW_D": ARROW_D,
        "ARROW_DR": ARROW_DR,
        "H_LINE": H_LINE,
        "H_LINE_DBL": H_LINE_DBL,
        "H_LINE_HVY": H_LINE_HVY,
        "V_LINE": V_LINE,
        "V_LINE_DBL": V_LINE_DBL,
        "V_LINE_HVY": V_LINE_HVY,
        "ELLIPSIS": ELLIPSIS,
        "MIDDOT": MIDDOT,
        "DOTS": DOTS,
        "LINES": LINES,
        "ARROWS": ARROWS,
        "BLOCKS": BLOCKS,
        "BOUNCE": BOUNCE
    }

    _ASCII = {
        "CHECK": "v",
        "CROSS": "x",
        "WARN": "!",
        "INFO": "i",
        "QUESTION": "?",
        "BULLET": "*",
        "DIAMOND": "+",
        "STAR": "*",
        "STAR_O": "*",
        "CIRCLE": "o",
        "CIRCLE_O": "o",
        "SQUARE": "#",
        "SQUARE_O": "#",
        "TRIANGLE": ">",
        "ARROW_R": ">",
        "ARROW_L": "<",
        "ARROW_U": "^",
        "ARROW_D": "v",
        "ARROW_DR": "\\-",
        "H_LINE": "-",
        "H_LINE_DBL": "=",
        "H_LINE_HVY": "=",
        "V_LINE": "|",
        "V_LINE_DBL": "|",
        "V_LINE_HVY": "|",
        "ELLIPSIS": "...",
        "MIDDOT": ".",
        "DOTS": (".", "o", "O", "o"),
        "LINES": ("|", "/", "-", "\\"),
        "ARROWS": ("<", "^", ">", "v"),
        "BLOCKS": (" ", ".", ":", "#"),
        "BOUNCE": (".", "o", ".", "o")
    }

    @classmethod
    def apply(cls, symbols: Dict[str, Any]):
        """Apply a symbol mapping to the Sym class."""
        for key, val in symbols.items():
            setattr(cls, key, val)

    @classmethod
    def use_ascii(cls, enabled: bool = True):
        cls.apply(cls._ASCII if enabled else cls._UNICODE)


class Align(Enum):
    """Text alignment options."""
    LEFT   = auto()
    CENTER = auto()
    RIGHT  = auto()


class Layout(Enum):
    """Dashboard layout modes."""
    VERTICAL   = auto()
    HORIZONTAL = auto()
    GRID       = auto()


# ------------------------------------------------------------------------------
# 3. FORMATTERS
# ------------------------------------------------------------------------------

def fnum(n: Union[int, float], decimals: int = 0) -> str:
    """Format number with thousands separator: 1234567 -> 1,234,567"""
    if isinstance(n, float) and decimals > 0:
        return f"{n:,.{decimals}f}"
    return f"{int(n):,}"


def ftime(seconds: float, compact: bool = True) -> str:
    """Format duration: 3665 -> '1h 01m' or '1 hour 1 minute'"""
    s = max(0, int(seconds))
    
    if s < 60:
        return f"{s}s" if compact else f"{s} second{'s' if s != 1 else ''}"
    
    if s < 3600:
        m, sec = divmod(s, 60)
        if compact:
            return f"{m}m {sec:02d}s"
        return f"{m} min {sec} sec"
    
    h, rem = divmod(s, 3600)
    m = rem // 60
    
    if s < 86400:
        if compact:
            return f"{h}h {m:02d}m"
        return f"{h} hour{'s' if h != 1 else ''} {m} min"
    
    d, rem = divmod(s, 86400)
    h = rem // 3600
    if compact:
        return f"{d}d {h:02d}h"
    return f"{d} day{'s' if d != 1 else ''} {h} hr"


def fsize(size_bytes: int, precision: int = 1) -> str:
    """Format file size: 1536000 -> '1.5MB'"""
    if size_bytes <= 0:
        return "0B"
    
    units = ("B", "KB", "MB", "GB", "TB", "PB", "EB")
    size = float(size_bytes)
    unit_idx = 0
    
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1
    
    if unit_idx == 0:
        return f"{int(size)}B"
    return f"{size:.{precision}f}{units[unit_idx]}"


def fpercent(value: float, total: float = 100, decimals: int = 1, colored: bool = False) -> str:
    """Format as percentage: fpercent(25, 100) -> '25.0%'"""
    if total == 0:
        return "0%"
    pct = (value / total) * 100
    res = f"{pct:.{decimals}f}%" if decimals else f"{int(pct)}%"
    if colored:
        # Default threshold: <50 green, <80 yellow, else red
        if pct < 50: clr = C.GRN
        elif pct < 80: clr = C.YLW
        else: clr = C.RED
        return f"{clr}{res}{C.R}"
    return res


def ftrunc(text: str, max_len: int, suffix: str = "...") -> str:
    """Truncate text with suffix (non-ANSI aware version)."""
    if len(text) <= max_len:
        return text
    return text[:max_len - len(suffix)] + suffix


def markdown(text: str) -> str:
    """
    Enhanced markdown-to-ANSI parser.
    Supports: 
    - Headers: # H1, ## H2, ### H3
    - Inline: **bold**, *italic*, __underline__, ~strikethrough~, `code`
    - Links: [text](url)
    - Lists: - item or * item
    - Blockquotes: > text
    - Code Blocks: ```code```
    - Horizontal Rule: ---
    """
    if not Term.colors:
        return text
    
    lines = text.split('\n')
    result_lines = []
    in_code_block = False
    code_lines = []
    
    for line in lines:
        # Code Blocks: ```
        if line.strip().startswith('```'):
            if in_code_block:
                # End of block
                block_content = '\n'.join(code_lines)
                # Render code block in a box or with background
                result_lines.append(box(block_content, color=CURRENT_THEME.dim, padding=0, style=Border.ASCII))
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            continue
            
        if in_code_block:
            code_lines.append(line)
            continue

        # Horizontal Rule: ---
        if re.match(r'^---+$', line.strip()):
            result_lines.append(hline())
            continue
            
        # Headers
        h_match = re.match(r'^(#{1,3})\s+(.*)', line)
        if h_match:
            level = len(h_match.group(1))
            content = h_match.group(2)
            if level == 1:
                result_lines.append(f"{C.B}{C.U}{C.CYN_B}{content.upper()}{C.R}")
            elif level == 2:
                result_lines.append(f"{C.B}{C.CYN}{content}{C.R}")
            else:
                result_lines.append(f"{C.B}{C.D}{content}{C.R}")
            continue

        # Blockquotes: > text
        bq_match = re.match(r'^>\s+(.*)', line)
        if bq_match:
            content = bq_match.group(1)
            bq_char = "┃" if Term.unicode else "|"
            result_lines.append(f"{CURRENT_THEME.primary}{bq_char}{C.R} {C.I}{content}{C.R}")
            continue

        # Lists
        l_match = re.match(r'^[\-\*]\s+(.*)', line)
        if l_match:
            content = l_match.group(1)
            result_lines.append(f"  {C.CYN}{Sym.BULLET}{C.R} {content}")
            continue

        # Inline styles
        # Links: [text](url)
        line = re.sub(r'\[(.*?)\]\((.*?)\)', lambda m: C.link(m.group(1), m.group(2)), line)
        # Bold: **text**
        line = re.sub(r'\*\*(.*?)\*\*', f'{C.B}\\1{C.R}', line)
        # Italic: *text*
        line = re.sub(r'\*(.*?)\*', f'{C.I}\\1{C.R}', line)
        # Underline: __text__
        line = re.sub(r'__(.*?)__', f'{C.U}\\1{C.R}', line)
        # Strikethrough: ~text~
        line = re.sub(r'~(.*?)~', f'{C.S}\\1{C.R}', line)
        # Code: `text`
        line = re.sub(r'`(.*?)`', f'{C.BG_BLK}{C.WHT}\\1{C.R}', line)
        
        result_lines.append(line)
    
    return '\n'.join(result_lines)


# ------------------------------------------------------------------------------
# 4. STATUS MESSAGES
# ------------------------------------------------------------------------------

def _print_status(symbol: str, color: str, msg: str, prefix: str = ""):
    """Internal helper for status messages."""
    print(f"{prefix}{color}{symbol}{C.R} {msg}")


def success(msg: str, prefix: str = ""):
    """Print success message with checkmark."""
    _print_status(Sym.CHECK, CURRENT_THEME.success, msg, prefix)


def error(msg: str, prefix: str = ""):
    """Print error message with cross."""
    _print_status(Sym.CROSS, CURRENT_THEME.error, msg, prefix)


def warn(msg: str, prefix: str = ""):
    """Print warning message with symbol."""
    _print_status(Sym.WARN, CURRENT_THEME.warning, msg, prefix)


def info(msg: str, prefix: str = ""):
    """Print info message with symbol."""
    _print_status(Sym.INFO, CURRENT_THEME.info, msg, prefix)


def step(msg: str, num: int = None, prefix: str = ""):
    """Print step indicator."""
    marker = f"{C.CYN}[{num}]{C.R}" if num else f"{C.CYN}{Sym.TRIANGLE}{C.R}"
    print(f"{prefix}{marker} {msg}")


def bullet(msg: str, indent: int = 0, color: str = ""):
    """Print bulleted item with optional indentation."""
    pad = "  " * indent
    clr = color or C.D
    print(f"{pad}{clr}{Sym.BULLET}{C.R} {msg}")


# ------------------------------------------------------------------------------
# 5. BORDER STYLES
# ------------------------------------------------------------------------------

class Border:
    """
    Border style tuples: (TL, H, TR, V, BR, B, BL)
    TL=top-left, H=horizontal, TR=top-right, V=vertical,
    BR=bottom-right, B=bottom, BL=bottom-left
    """
    ROUNDED = ("╭", "─", "╮", "│", "╯", "─", "╰")
    SHARP   = ("┌", "─", "┐", "│", "┘", "─", "└")
    DOUBLE  = ("╔", "═", "╗", "║", "╝", "═", "╚")
    HEAVY   = ("┏", "━", "┓", "┃", "┛", "━", "┗")
    ASCII   = ("+", "-", "+", "|", "+", "-", "+")
    BLOCK   = ("▛", "▀", "▜", "▌", "▟", "▄", "▙")
    NONE    = (" ", " ", " ", " ", " ", " ", " ")


class Skin:
    """
    Visual skin (symbols + default border style). Purely aesthetic.
    """
    def __init__(
        self,
        name: str,
        border_style: Tuple[str, ...] = Border.ROUNDED,
        symbols: Dict[str, Any] = None,
        ascii_only: bool = False
    ):
        self.name = name
        self.border_style = border_style
        self.symbols = symbols or {}
        self.ascii_only = ascii_only


SKINS = {
    "Default": Skin("Default", border_style=Border.ROUNDED, symbols=Sym._UNICODE),
    "ASCII": Skin("ASCII", border_style=Border.ASCII, symbols=Sym._ASCII, ascii_only=True),
    "High Contrast": Skin("High Contrast", border_style=Border.DOUBLE, symbols=Sym._UNICODE),
    "Neon": Skin("Neon", border_style=Border.HEAVY, symbols=Sym._UNICODE)
}

CURRENT_SKIN = SKINS["Default"]
DEFAULT_BORDER_STYLE = CURRENT_SKIN.border_style


def register_skin(name: str, skin: Skin):
    """Register a custom skin by name."""
    SKINS[name] = skin


def set_skin(name_or_skin: Union[str, Skin]):
    """Apply a skin (affects symbols and default borders)."""
    global CURRENT_SKIN, DEFAULT_BORDER_STYLE
    if isinstance(name_or_skin, Skin):
        skin = name_or_skin
    else:
        skin = SKINS.get(name_or_skin)
    if not skin:
        return
    CURRENT_SKIN = skin
    DEFAULT_BORDER_STYLE = skin.border_style
    supports = Term.supports_unicode()
    if not supports:
        Term.unicode = False
        Sym.use_ascii(True)
        DEFAULT_BORDER_STYLE = Border.ASCII
        return
    if skin.symbols:
        Sym.apply(skin.symbols)
    if skin.ascii_only:
        Term.unicode = False
        Sym.use_ascii(True)
    else:
        Term.unicode = True


def set_unicode(enabled: bool = True):
    """Force Unicode or ASCII symbols (borders remain controlled by skin)."""
    Term.unicode = enabled
    Sym.use_ascii(not enabled)


if not Term.supports_unicode():
    set_skin("ASCII")


# ------------------------------------------------------------------------------
# 6. BOX DRAWING
# ------------------------------------------------------------------------------

def box(
    content: Any,
    title: Optional[str] = None,
    width: Union[int, str] = 0,
    color: Union[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = "",
    style: Tuple[str, ...] = None,
    padding: int = 1,
    align: Align = Align.LEFT,
    min_width: int = 10,
    gradient_dir: str = "vertical"
) -> str:
    """
    Draw a bordered box around content.
    
    Args:
        content: Text content (can include newlines and ANSI codes)
        title: Optional title displayed in top border
        width: Box width (fixed int or "50%")
        color: Border color (ANSI string or (start_rgb, end_rgb) tuple for gradient)
        style: Border style tuple
        padding: Vertical padding inside box
        align: Content text alignment
        min_width: Minimum box width
    
    Returns:
        Formatted box as string
    """
    if style is None:
        style = DEFAULT_BORDER_STYLE
    tl, h, tr, v, br, b, bl = style
    
    # Handle gradient color
    is_gradient = isinstance(color, tuple) and len(color) == 2
    if not color:
        color = CURRENT_THEME.primary
    
    # Determine width
    term_w = Term.width()
    parsed_width = Term.parse_width(width, term_w)
    
    is_auto = isinstance(width, str) and width.strip().lower() in ("auto", "fit", "min")
    if parsed_width <= 0:
        # Auto-calculate width based on content
        content_lines = str(content).split('\n')
        max_content_w = max((Term.vlen(l) for l in content_lines), default=0)
        if title:
            max_content_w = max(max_content_w, Term.vlen(title) + 4)
        parsed_width = max_content_w + 4 # 2 for borders, 2 for padding spaces
        if is_auto:
            min_width = 0
        
    width = max(min_width, min(parsed_width, term_w))
    inner_w = width - 4
    
    lines = []
    
    def apply_color(text: str, line_idx: int, total_lines: int, start_pos: int = 0, total_len: int = None) -> str:
        if is_gradient:
            if gradient_dir == "horizontal":
                total_len = total_len or max(1, len(text))
                return _gradient_segment(text, color[0], color[1], start_pos, total_len)
            # Vertical gradient for the whole box
            r = int(color[0][0] + (color[1][0] - color[0][0]) * line_idx / (total_lines - 1)) if total_lines > 1 else color[0][0]
            g = int(color[0][1] + (color[1][1] - color[0][1]) * line_idx / (total_lines - 1)) if total_lines > 1 else color[0][1]
            b = int(color[0][2] + (color[1][2] - color[0][2]) * line_idx / (total_lines - 1)) if total_lines > 1 else color[0][2]
            return f"{C.rgb(r, g, b)}{text}{C.R}"
        return f"{color}{text}{C.R}"

    # Estimate total lines for gradient
    content_str = str(content)
    wrapped_lines_count = sum(len(Term.wrap(l, inner_w)) if l.strip() else 1 for l in content_str.split('\n'))
    total_expected_lines = 1 + padding * 2 + wrapped_lines_count + 1

    # --- TOP BORDER ---
    if title:
        title_vis = Term.vlen(title)
        if title_vis > inner_w - 2:
            title = Term.truncate(title, inner_w - 2)
            title_vis = Term.vlen(title)
        
        dashes = width - title_vis - 4
        dash_l = dashes // 2
        dash_r = dashes - dash_l
        
        left = apply_color(tl + h * dash_l, 0, total_expected_lines, 0, width)
        right_segment = h * dash_r + tr
        right = apply_color(right_segment, 0, total_expected_lines, width - len(right_segment), width)
        lines.append(f"{left} {C.B}{title}{C.R} {right}")
    else:
        lines.append(apply_color(tl + h * (width - 2) + tr, 0, total_expected_lines, 0, width))
    
    line_idx = 1

    # --- TOP PADDING ---
    for _ in range(padding):
        left = apply_color(v, line_idx, total_expected_lines, 0, width)
        right = apply_color(v, line_idx, total_expected_lines, width - 1, width)
        lines.append(f"{left}{' ' * (width - 2)}{right}")
        line_idx += 1
    
    # --- CONTENT ---
    for raw_line in content_str.split('\n'):
        if not raw_line.strip():
            left = apply_color(v, line_idx, total_expected_lines, 0, width)
            right = apply_color(v, line_idx, total_expected_lines, width - 1, width)
            lines.append(f"{left} {Term.pad('', inner_w)} {right}")
            line_idx += 1
        else:
            for line in Term.wrap(raw_line, inner_w):
                padded = Term.pad(line, inner_w, align)
                left = apply_color(v, line_idx, total_expected_lines, 0, width)
                right = apply_color(v, line_idx, total_expected_lines, width - 1, width)
                lines.append(f"{left} {padded} {right}")
                line_idx += 1
    
    # --- BOTTOM PADDING ---
    for _ in range(padding):
        left = apply_color(v, line_idx, total_expected_lines, 0, width)
        right = apply_color(v, line_idx, total_expected_lines, width - 1, width)
        lines.append(f"{left}{' ' * (width - 2)}{right}")
        line_idx += 1
    
    # --- BOTTOM BORDER ---
    lines.append(apply_color(bl + b * (width - 2) + br, total_expected_lines - 1, total_expected_lines, 0, width))
    
    return "\n".join(lines)


def box_gradient(
    content: Any,
    title: Optional[str] = None,
    width: Union[int, str] = 0,
    gradient: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = None,
    direction: str = "vertical",
    **kwargs
) -> str:
    """Helper to draw a box with a vertical or horizontal border gradient."""
    if not gradient:
        return box(content, title=title, width=width, **kwargs)
    return box(
        content,
        title=title,
        width=width,
        color=gradient,
        gradient_dir=direction,
        **kwargs
    )


def shadow_box(
    content: Any,
    title: Optional[str] = None,
    width: Union[int, str] = 0,
    shadow_color: str = "",
    shadow_char: str = "█",
    **kwargs
) -> str:
    """Helper to draw a box and apply a shadow in one call."""
    b = box(content, title=title, width=width, **kwargs)
    return shadow(b, color=shadow_color, char=shadow_char)


def calendar_widget(year: int = None, month: int = None, color: str = "") -> str:
    """
    Render a beautiful calendar for a given month.
    """
    if year is None:
        year = datetime.datetime.now().year
    if month is None:
        month = datetime.datetime.now().month
    if not color:
        color = CURRENT_THEME.primary
        
    cal = calendar.monthcalendar(year, month)
    month_name = calendar.month_name[month]
    
    headers = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
    rows = []
    
    today = datetime.date.today()
    
    for week in cal:
        row = []
        for i, day in enumerate(week):
            if day == 0:
                row.append("")
            else:
                d_str = str(day)
                if year == today.year and month == today.month and day == today.day:
                    row.append(f"{C.BG_WHT}{C.BLK}{C.B} {d_str} {C.R}")
                elif i >= 5: # Weekend
                    row.append(f"{C.D}{d_str}{C.R}")
                else:
                    row.append(d_str)
        rows.append(row)
        
    # Create the calendar table inside a box
    cal_table = table(
        headers, 
        rows, 
        padding=0, 
        color=color, 
        alignments=[Align.CENTER] * 7,
        style=Border.NONE
    )
    
    return box(
        cal_table,
        title=f" {month_name} {year} ",
        color=color,
        padding=1,
        align=Align.CENTER
    )


def shadow(content: str, color: str = "", char: str = "█", offset_x: int = 1, offset_y: int = 1) -> str:
    """
    Add a robust drop-shadow effect to a boxed string.
    
    Args:
        content: The text/box to add shadow to
        color: Shadow color
        char: Shadow character
        offset_x: Horizontal shadow offset
        offset_y: Vertical shadow offset
    """
    if not content:
        return ""
    
    lines = content.split('\n')
    if not lines:
        return content
    
    clr = color or C.GRY
    result = []
    
    # Visible width of the content
    width = max(Term.vlen(l) for l in lines)
    
    # Add shadow to the right of each line
    for i, line in enumerate(lines):
        padding = " " * (width - Term.vlen(line))
        result.append(f"{line}{padding}{clr}{char * offset_x}{C.R}")
    
    # Add bottom shadow lines
    for _ in range(offset_y):
        result.append(f"{' ' * offset_x}{clr}{char * width}{C.R}")
    
    return "\n".join(result)


def floating_box(
    content: str,
    x: int = 1,
    y: int = 1,
    save_cursor: bool = True
):
    """
    Render a box at absolute terminal coordinates.
    """
    if not Term.is_tty:
        print(content)
        return
    
    if save_cursor:
        sys.stdout.write("\033[s") # Save
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        Term.move_cursor(y + i, x)
        sys.stdout.write(line)
    
    if save_cursor:
        sys.stdout.write("\033[u") # Restore
    sys.stdout.flush()


def overlay(
    base: str,
    overlay_content: str,
    x: int = 2,
    y: int = 2
) -> str:
    """
    Overlay one string onto another at specific character coordinates.
    Useful for static compositions.
    """
    base_lines = base.split('\n')
    overlay_lines = overlay_content.split('\n')
    
    result_lines = base_lines[:]
    
    for i, o_line in enumerate(overlay_lines):
        target_y = y + i - 1
        if 0 <= target_y < len(result_lines):
            result_lines[target_y] = _replace_visible(result_lines[target_y], x - 1, o_line)
        elif target_y >= len(result_lines):
            # Extend base if overlay is taller
            while len(result_lines) <= target_y:
                result_lines.append("")
            result_lines[target_y] = _replace_visible("", x - 1, o_line)
            
    return "\n".join(result_lines)


def markdown_to_ansi(text: str) -> str:
    """Alias for markdown()."""
    return markdown(text)


def _replace_visible(text: str, start: int, replacement: str) -> str:
    """Replace visible characters in text with replacement string, preserving ANSI codes."""
    result = []
    visible_idx = 0
    replacement_len = Term.vlen(replacement)
    end = start + replacement_len
    
    i = 0
    while i < len(text):
        match = Term._ANSI_RE.match(text[i:])
        if match:
            result.append(match.group())
            i += len(match.group())
        else:
            if start <= visible_idx < end:
                if visible_idx == start:
                    result.append(replacement)
            else:
                result.append(text[i])
            
            visible_idx += 1
            i += 1
    
    if visible_idx < start:
        result.append(" " * (start - visible_idx))
        result.append(replacement)
        
    return "".join(result)


def hline(width: Union[int, str] = 0, char: str = None, color: str = "") -> str:
    """Draw a horizontal line with percentage width support."""
    if char is None:
        char = Sym.H_LINE
    if not color:
        color = CURRENT_THEME.border
    
    term_w = Term.width()
    parsed_width = Term.parse_width(width, term_w)
    
    if parsed_width <= 0:
        width = min(80, term_w - 2)
    else:
        width = min(parsed_width, term_w)
        
    return f"{color}{char * width}{C.R}"


def divider(text: str = "", width: Union[int, str] = 0, char: str = None, color: str = "") -> str:
    """Draw a horizontal divider with optional centered text and percentage width support."""
    if char is None:
        char = Sym.H_LINE
    if not color:
        color = CURRENT_THEME.border
    
    term_w = Term.width()
    parsed_width = Term.parse_width(width, term_w)
    
    if parsed_width <= 0:
        width = min(80, term_w - 2)
    else:
        width = min(parsed_width, term_w)
    
    if not text:
        return hline(width, char, color)
    
    text_vis = Term.vlen(text)
    remaining = width - text_vis - 2
    left = max(0, remaining // 2)
    right = max(0, remaining - left)
    
    return f"{color}{char * left} {C.R}{text}{color} {char * right}{C.R}"


# ------------------------------------------------------------------------------
# 7. TABLE
# ------------------------------------------------------------------------------

def table(
    headers: List[str],
    rows: List[List[Any]],
    footers: List[str] = None,
    color: str = "",
    alignments: List[Align] = None,
    max_col_width: Union[int, str] = 40,
    widths: List[Union[int, str]] = None,
    padding: int = 1,
    wrap: bool = True,
    style: Tuple[str, ...] = None,
    row_sep: bool = False,
    expand: bool = False
) -> str:
    """
    Create a formatted table with multi-line cell support and percentage width.
    
    Args:
        headers: Column header labels
        rows: List of data rows
        footers: Optional column footer labels
        color: Border color
        alignments: Per-column alignment
        max_col_width: Maximum column width (fixed int or "25%")
        widths: Optional list of widths (fixed int or "25%")
        padding: Cell padding spaces
        wrap: Wrap text in cells if it exceeds max_col_width
        style: Border style tuple
        row_sep: Add separator lines between rows
        expand: Expand table to full terminal width
    """
    if not headers:
        return ""
    
    if style is None:
        style = DEFAULT_BORDER_STYLE
    if not color:
        color = CURRENT_THEME.border
    
    num_cols = len(headers)
    term_w = Term.width()
    
    # Calculate overhead (borders and padding)
    overhead = (num_cols + 1) + (num_cols * padding * 2)
    available_content_w = max(10, term_w - overhead)
    
    # Default alignments
    if alignments is None:
        alignments = [Align.LEFT] * num_cols
    alignments.extend([Align.LEFT] * (num_cols - len(alignments)))
    
    # Determine column widths
    def _auto_col_width(i: int, max_w: int) -> int:
        max_col = max_w
        h_lines = str(headers[i]).split('\n')
        max_col = max(max_col, max(Term.vlen(l) for l in h_lines))
        for row in rows:
            if i < len(row):
                cell_text = str(row[i])
                if "\n" in cell_text:
                    max_col = max(max_col, max(Term.vlen(l) for l in cell_text.split("\n")))
                else:
                    max_col = max(max_col, Term.vlen(cell_text))
        if footers and i < len(footers):
            f_lines = str(footers[i]).split('\n')
            max_col = max(max_col, max(Term.vlen(l) for l in f_lines))
        return min(max_col, max_w)

    if widths:
        # Use provided widths
        col_widths = []
        for w in widths:
            col_widths.append(Term.parse_width(w, available_content_w))
        # Pad with default if too few widths provided
        while len(col_widths) < num_cols:
            col_widths.append(Term.parse_width(max_col_width, available_content_w))
        parsed_max_w = Term.parse_width(max_col_width, available_content_w)
        for i in range(num_cols):
            if col_widths[i] <= 0:
                col_widths[i] = _auto_col_width(i, parsed_max_w)
    else:
        # Calculate optimal widths
        parsed_max_w = Term.parse_width(max_col_width, available_content_w)
        col_widths = []
        for i, header in enumerate(headers):
            # Header width (can be multi-line)
            h_lines = str(header).split('\n')
            max_w = max(Term.vlen(l) for l in h_lines)
            
            # Row widths
            for row in rows:
                if i < len(row):
                    cell_text = str(row[i])
                    if "\n" in cell_text:
                        max_w = max(max_w, max(Term.vlen(l) for l in cell_text.split("\n")))
                    else:
                        max_w = max(max_w, Term.vlen(cell_text))
            
            # Footer widths
            if footers and i < len(footers):
                f_lines = str(footers[i]).split('\n')
                max_w = max(max_w, max(Term.vlen(l) for l in f_lines))
                
            col_widths.append(min(max_w, parsed_max_w))
            
    # Expand if requested
    if expand:
        total_content_w = sum(col_widths)
        if total_content_w < available_content_w:
            extra = available_content_w - total_content_w
            # Distribute extra width to columns
            for i in range(num_cols):
                col_widths[i] += extra // num_cols
            # Add remaining to last column
            col_widths[-1] += extra % num_cols
    
    TL, H, TR, V, BR, B, BL = style
    # Custom intersections for table
    if style == Border.NONE:
        TM = ML = MM = MR = BM = " "
    elif style == Border.ASCII or not Term.unicode:
        TM = ML = MM = MR = BM = "+"
    else:
        TM = "┬"
        ML = "├"
        MM = "┼"
        MR = "┤"
        BM = "┴"

    pad_str = " " * padding
    lines = []
    
    # Top border: ┌───┬───┐
    top = f"{color}{TL}" + TM.join(H * (w + padding * 2) for w in col_widths) + f"{TR}{C.R}"
    lines.append(top)
    
    # Headers: │ H │ H │
    header_lines = []
    for i, h in enumerate(headers):
        header_lines.append(Term.wrap(str(h), col_widths[i]))
    
    max_h_height = max(len(hl) for hl in header_lines)
    for h_idx in range(max_h_height):
        row_parts = []
        for i in range(num_cols):
            text = header_lines[i][h_idx] if h_idx < len(header_lines[i]) else ""
            cell = Term.pad(text, col_widths[i], Align.CENTER)
            row_parts.append(f"{pad_str}{C.B}{cell}{C.R}{pad_str}")
        lines.append(f"{color}{V}{C.R}" + f"{color}{V}{C.R}".join(row_parts) + f"{color}{V}{C.R}")
    
    # Header separator: ├───┼───┤
    sep = f"{color}{ML}" + MM.join(H * (w + padding * 2) for w in col_widths) + f"{MR}{C.R}"
    lines.append(sep)
    
    # Data rows: │ D │ D │
    for r_idx, row in enumerate(rows):
        cell_lines = []
        max_cell_height = 1
        
        for i in range(num_cols):
            val = str(row[i]) if i < len(row) else ""
            if wrap:
                wrapped = []
                for part in val.split("\n"):
                    wrapped.extend(Term.wrap(part, col_widths[i]))
                cell_lines.append(wrapped)
                max_cell_height = max(max_cell_height, len(wrapped))
            else:
                truncated = Term.truncate(val, col_widths[i]) if Term.vlen(val) > col_widths[i] else val
                cell_lines.append([truncated])
        
        for h in range(max_cell_height):
            row_parts = []
            for i in range(num_cols):
                text = cell_lines[i][h] if h < len(cell_lines[i]) else ""
                cell = Term.pad(text, col_widths[i], alignments[i])
                row_parts.append(f"{pad_str}{cell}{pad_str}")
            lines.append(f"{color}{V}{C.R}" + f"{color}{V}{C.R}".join(row_parts) + f"{color}{V}{C.R}")
            
        # Optional row separator
        if row_sep and r_idx < len(rows) - 1:
            lines.append(sep)
    
    # Footer separator: ├───┼───┤
    if footers:
        lines.append(sep)
        footer_lines = []
        for i, f in enumerate(footers):
            footer_lines.append(Term.wrap(str(f), col_widths[i]))
        
        max_f_height = max(len(fl) for fl in footer_lines)
        for f_idx in range(max_f_height):
            row_parts = []
            for i in range(num_cols):
                text = footer_lines[i][f_idx] if f_idx < len(footer_lines[i]) else ""
                cell = Term.pad(text, col_widths[i], Align.CENTER)
                row_parts.append(f"{pad_str}{C.B}{cell}{C.R}{pad_str}")
            lines.append(f"{color}{V}{C.R}" + f"{color}{V}{C.R}".join(row_parts) + f"{color}{V}{C.R}")

    # Bottom border: └───┴───┘
    bottom = f"{color}{BL}" + BM.join(B * (w + padding * 2) for w in col_widths) + f"{BR}{C.R}"
    lines.append(bottom)
    
    return "\n".join(lines)


# ------------------------------------------------------------------------------
# 8. COLUMNS LAYOUT
# ------------------------------------------------------------------------------

def cols(
    columns: List[Union[List[str], str]],
    spacing: int = 2,
    alignments: List[Align] = None,
    widths: List[Union[int, str]] = None
) -> str:
    """
    Render content in side-by-side columns with support for percentage widths.
    
    Args:
        columns: List of columns (each column can be a list of lines or a multi-line string)
        spacing: Space between columns
        alignments: Per-column alignment
        widths: Optional list of widths (fixed int or "50%")
    
    Returns:
        Formatted columns string
    """
    if not columns:
        return ""
    
    num_cols = len(columns)
    term_w = Term.width()
    
    if alignments is None:
        alignments = [Align.LEFT] * num_cols
    
    # Normalize columns: convert everything to List[str]
    norm_cols = []
    for col in columns:
        if isinstance(col, str):
            norm_cols.append(col.split('\n'))
        else:
            # Handle potential multi-line strings within the list
            flat_col = []
            for item in col:
                s = str(item)
                if '\n' in s:
                    flat_col.extend(s.split('\n'))
                else:
                    flat_col.append(s)
            norm_cols.append(flat_col)
    
    # Calculate widths
    col_widths = []
    available_w = term_w - (spacing * (num_cols - 1))
    
    if widths:
        for i, w in enumerate(widths):
            if i < num_cols:
                parsed_w = Term.parse_width(w, available_w)
                col_widths.append(parsed_w)
        # Fill remaining widths if not enough provided
        while len(col_widths) < num_cols:
            col_widths.append(max((Term.vlen(line) for line in norm_cols[len(col_widths)]), default=0))
    else:
        for col in norm_cols:
            max_w = max((Term.vlen(line) for line in col), default=0)
            col_widths.append(max_w)
    
    # Wrap each column's content to its assigned width
    wrapped_cols = []
    for i, col in enumerate(norm_cols):
        width = col_widths[i]
        wrapped_col = []
        for line in col:
            # Using Term.wrap to handle ANSI codes and word wrapping
            wrapped_col.extend(Term.wrap(line, width))
        wrapped_cols.append(wrapped_col)
    
    # Build output
    lines = []
    spacer = " " * spacing
    
    for row_items in itertools.zip_longest(*wrapped_cols, fillvalue=""):
        parts = []
        for i, text in enumerate(row_items):
            align = alignments[i] if i < len(alignments) else Align.LEFT
            width = col_widths[i]
            parts.append(Term.pad(text, width, align))
        lines.append(spacer.join(parts).rstrip())
    
    return "\n".join(lines)


def responsive_cols(
    columns: List[Union[List[str], str]],
    threshold: int = 80,
    spacing: int = 2,
    alignments: List[Align] = None,
    widths: List[Union[int, str]] = None,
    vertical_spacing: int = 1
) -> str:
    """
    Render columns if terminal width >= threshold, otherwise render as vertical sections.
    """
    if Term.width() < threshold:
        output = []
        for col in columns:
            if isinstance(col, str):
                output.append(col)
            else:
                output.append("\n".join(col))
        return ("\n" * (vertical_spacing + 1)).join(output)
    
    return cols(columns, spacing, alignments, widths)


# ------------------------------------------------------------------------------
# 9. KEY-VALUE DISPLAY
# ------------------------------------------------------------------------------

def kvlist(
    items: List[Tuple[str, Any]],
    separator: str = ":",
    key_color: str = "",
    val_color: str = "",
    key_width: int = 0,
    padding: int = 0,
    width: int = 0
) -> str:
    """
    Format key-value pairs in aligned columns.
    
    Args:
        items: List of (key, value) tuples
        separator: Key-value separator
        key_color: Color for keys
        val_color: Color for values
        key_width: Minimum key column width (0 = auto)
        padding: Left padding (indentation)
        width: Total width for wrapping (0 = terminal width)
    
    Returns:
        Formatted key-value string
    """
    if not items:
        return ""
    
    if not key_color:
        key_color = CURRENT_THEME.dim
    
    term_w = width or Term.width()
    indent = " " * padding
    
    # Calculate key column width
    k_col_width = max(key_width, max(Term.vlen(str(k)) for k, _ in items))
    
    # Calculate available width for values
    # Space for: indent + key + ' ' + separator + ' '
    prefix_len = padding + k_col_width + 1 + Term.vlen(separator) + 1
    v_col_width = max(10, term_w - prefix_len)
    
    lines = []
    for key, val in items:
        k_str = str(key)
        v_str = str(val)
        k_pad = k_col_width - Term.vlen(k_str)
        
        # Wrap value
        v_lines = Term.wrap(v_str, v_col_width)
        
        # First line
        lines.append(
            f"{indent}{key_color}{k_str}{C.R}{' ' * k_pad} {separator} {val_color}{v_lines[0] if v_lines else ''}{C.R}"
        )
        
        # Subsequent lines
        for v_line in v_lines[1:]:
            # Align with the start of the first line's value
            lines.append(
                f"{indent}{' ' * k_col_width}   {val_color}{v_line}{C.R}"
            )
    
    return "\n".join(lines)


# ------------------------------------------------------------------------------
# 10. TREE VIEW
# ------------------------------------------------------------------------------

def tree(
    data: Union[Dict, List, str],
    label: str = "root",
    color: str = "",
    _prefix: str = "",
    _is_last: bool = True
) -> str:
    """
    Render hierarchical data as a tree.
    """
    if not color:
        color = CURRENT_THEME.primary
        
    lines = []
    
    # Branch symbols
    if Term.unicode:
        connector = "└── " if _is_last else "├── "
        next_prefix = _prefix + ("    " if _is_last else "│   ")
    else:
        connector = "\\-- " if _is_last else "+-- "
        next_prefix = _prefix + ("    " if _is_last else "|   ")
    
    # Header for current node
    if _prefix == "":
        lines.append(f"{color}{C.B}{label}{C.R}")
    else:
        lines.append(f"{C.D}{_prefix}{C.R}{color}{connector}{C.R}{label}")
    
    if isinstance(data, dict):
        items = list(data.items())
        for i, (key, val) in enumerate(items):
            is_last = (i == len(items) - 1)
            lines.append(tree(val, label=str(key), color=color, _prefix=next_prefix, _is_last=is_last))
    elif isinstance(data, list):
        for i, val in enumerate(data):
            is_last = (i == len(data) - 1)
            label_val = f"[{i}]" if not isinstance(val, (dict, list)) else f"Item {i}"
            lines.append(tree(val, label=label_val, color=color, _prefix=next_prefix, _is_last=is_last))
    elif not isinstance(data, (dict, list)):
        # Leaf node
        # We've already printed the label, if it's a simple value we might want to show it
        # But tree usually shows keys as labels. If it's a leaf, the label IS the value often.
        if lines:
            lines[-1] += f"{C.D} : {C.R}{data}"

    return "\n".join(lines)


# ------------------------------------------------------------------------------
# 11. DATA VISUALIZATION (SPARKLINE)
# ------------------------------------------------------------------------------

def sparkline(data: List[float], color: str = "", gradient: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = None) -> str:
    """
    Generate a tiny inline sparkline graph with optional color gradient.
    
    Args:
        data: List of values to plot
        color: Single color to use (default if gradient is None)
        gradient: Optional tuple of (start_rgb, end_rgb) for linear color gradient
    """
    if not data:
        return ""
    
    if not color and not gradient:
        color = CURRENT_THEME.primary
        
    # Unicode block characters for different heights
    if Term.unicode:
        blocks = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    else:
        blocks = [" ", ".", ":", "-", "=", "+", "*", "#", "#"]
    
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val
    
    n = len(data)
    result = []
    for i, val in enumerate(data):
        if range_val == 0:
            idx = 4
        else:
            idx = int(((val - min_val) / range_val) * (len(blocks) - 1))
        
        char = blocks[idx]
        
        if gradient:
            start_rgb, end_rgb = gradient
            r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * i / (n - 1)) if n > 1 else start_rgb[0]
            g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * i / (n - 1)) if n > 1 else start_rgb[1]
            b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * i / (n - 1)) if n > 1 else start_rgb[2]
            result.append(f"{C.rgb(r, g, b)}{char}")
        else:
            result.append(f"{color}{char}")
    
    return "".join(result) + C.R


# ------------------------------------------------------------------------------
# 12. TABS & NAVIGATION
# ------------------------------------------------------------------------------

def tabs(
    options: List[str],
    active_idx: int = 0,
    color: str = "",
    active_color: str = "",
    width: int = 0
) -> str:
    """
    Render a horizontal tab bar.
    """
    if not options:
        return ""
    
    if not color:
        color = CURRENT_THEME.dim
    if not active_color:
        active_color = CURRENT_THEME.primary
        
    term_width = width or Term.width()
    
    # Calculate spacing
    tab_strings = []
    for i, opt in enumerate(options):
        if i == active_idx:
            tab_strings.append(f"{active_color}{C.B} {opt.upper()} {C.R}")
        else:
            tab_strings.append(f"{color} {opt} {C.R}")
    
    # Join with separators
    sep_char = "│" if Term.unicode else "|"
    separator = f"{C.D}{sep_char}{C.R}"
    content = separator.join(tab_strings)
    
    # Add bottom border to active tab
    border_line = []
    current_pos = 0
    for i, opt in enumerate(options):
        tab_len = len(opt) + 2 # +2 for spaces
        if i == active_idx:
            border_line.append(f"{active_color}{Sym.H_LINE_HVY * tab_len}{C.R}")
        else:
            border_line.append(f"{C.D}{Sym.H_LINE * tab_len}{C.R}")
        
        if i < len(options) - 1:
            join_char = "┴" if Term.unicode else "+"
            border_line.append(f"{C.D}{join_char}{C.R}")
            
    return content + "\n" + "".join(border_line)


def breadcrumbs(path: List[str], color: str = "", separator: str = None) -> str:
    """
    Render a navigation breadcrumb trail.
    """
    if not path:
        return ""
    
    if separator is None:
        separator = f" {Sym.ARROW_R} "
    if not color:
        color = CURRENT_THEME.primary
        
    parts = []
    for i, item in enumerate(path):
        if i == len(path) - 1:
            parts.append(f"{color}{C.B}{item}{C.R}")
        else:
            parts.append(f"{C.D}{item}{C.R}")
            
    return separator.join(parts)


# ------------------------------------------------------------------------------
# 13. CHARTS & DATA VIZ
# ------------------------------------------------------------------------------

def barchart(
    data: Dict[str, float],
    width: int = 40,
    color: str = "",
    label_color: str = ""
) -> str:
    """
    Render a horizontal bar chart.
    """
    if not data:
        return ""
        
    if not color:
        color = CURRENT_THEME.primary
    if not label_color:
        label_color = CURRENT_THEME.secondary
        
    max_val = max(data.values()) if data.values() else 1
    max_label_len = max(len(k) for k in data.keys())
    
    lines = []
    for label, val in data.items():
        # Scale value to width
        bar_len = int((val / max_val) * width) if max_val > 0 else 0
        bar_char = "█" if Term.unicode else "#"
        bar = f"{color}{bar_char * bar_len}{C.R}"
        
        # Add numeric value at the end
        val_str = f" {C.D}{val}{C.R}"
        
        # Pad label
        padded_label = label.ljust(max_label_len)
        sep = "│" if Term.unicode else "|"
        lines.append(f"{label_color}{padded_label}{C.R} {sep} {bar}{val_str}")
        
    return "\n".join(lines)


# ------------------------------------------------------------------------------
# 14. PROGRESS INDICATORS
# ------------------------------------------------------------------------------

def progress(
    current: int,
    total: int,
    width: int = 30,
    label: str = "",
    color: str = "",
    empty_color: str = "",
    filled_char: str = "█",
    empty_char: str = "░",
    show_percent: bool = True,
    show_count: bool = False,
    indeterminate: bool = False
) -> str:
    """
    Generate a progress bar string.
    
    Args:
        current: Current progress value
        total: Total/maximum value
        width: Bar width in characters
        label: Optional label prefix
        color: Filled portion color
        empty_color: Empty portion color
        filled_char: Character for filled portion
        empty_char: Character for empty portion
        show_percent: Show percentage after bar
        show_count: Show count (current/total) after bar
        indeterminate: If True, show a moving block instead of filled progress
    
    Returns:
        Progress bar string
    """
    if not color:
        color = CURRENT_THEME.success
    if not empty_color:
        empty_color = CURRENT_THEME.dim
    if not Term.unicode:
        if filled_char == "█":
            filled_char = "#"
        if empty_char == "░":
            empty_char = "-"
    
    if indeterminate:
        # Moving block effect based on current as a frame counter
        block_width = max(2, width // 5)
        pos = current % (width + block_width) - block_width
        
        bar_chars = list(empty_char * width)
        for i in range(block_width):
            idx = pos + i
            if 0 <= idx < width:
                bar_chars[idx] = filled_char
        
        bar = f"{color}{''.join(bar_chars)}{C.R}"
        show_percent = False
        show_count = False
    else:
        pct = min(1.0, max(0.0, current / total)) if total > 0 else 0
        filled_len = int(width * pct)
        empty_len = width - filled_len
        bar = f"{color}{filled_char * filled_len}{C.R}{empty_color}{empty_char * empty_len}{C.R}"
    
    parts = []
    if label:
        parts.append(label)
    parts.append(bar)
    if show_percent:
        parts.append(f"{int(pct * 100):3d}%")
    if show_count and not indeterminate:
        parts.append(f"[{current}/{total}]")
    
    return " ".join(parts)


class ProgressBar:
    """
    Stateful progress bar for live updates.
    """
    def __init__(
        self,
        total: int = 0,
        label: str = "",
        width: int = 30,
        color: str = "",
        show_eta: bool = True,
        show_count: bool = True
    ):
        self.total = total
        self.label = label
        self.width = width
        self.color = color or C.GRN
        self.show_eta = show_eta
        self.show_count = show_count
        self.current = 0
        self.start_time = time.time()
        self.last_update = self.start_time
        self._is_managed = False
        self.indeterminate = (total <= 0)

    def update(self, value: int = None, increment: int = 1):
        """Update progress value."""
        if value is not None:
            self.current = value
        else:
            self.current += increment
        self.last_update = time.time()
        if not self._is_managed:
            self._render()

    def _render(self):
        if not Term.is_tty:
            return
        
        bar = progress(
            self.current, self.total,
            width=self.width,
            label=self.label,
            color=self.color,
            show_percent=not self.indeterminate,
            show_count=self.show_count and not self.indeterminate,
            indeterminate=self.indeterminate
        )
        
        if not self.indeterminate and self.show_eta and 0 < self.current < self.total:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                rate = self.current / elapsed
                remaining = (self.total - self.current) / rate if rate > 0 else 0
                bar += f" {C.D}ETA: {ftime(remaining)}{C.R}"
        
        sys.stdout.write(f"\r{bar}")
        sys.stdout.flush()

    def render(self, width: int = None) -> str:
        """Render the progress bar as a string (for use in Live)."""
        elapsed = self.last_update - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.current) / rate if rate > 0 else 0
        
        render_width = width if width is not None else self.width
        
        bar_str = progress(
            self.current, self.total, 
            width=render_width, label=self.label, color=self.color,
            show_percent=not self.indeterminate, 
            show_count=self.show_count and not self.indeterminate,
            indeterminate=self.indeterminate
        )
        
        suffix = ""
        if not self.indeterminate and self.show_eta and 0 < self.current < self.total:
            suffix = f" {C.D}ETA: {ftime(remaining)}{C.R}"
        elif not self.indeterminate and self.show_eta and self.current >= self.total:
            suffix = f" {C.D}Done in {int(elapsed)}s{C.R}"
            
        return f"{bar_str}{suffix}"

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if Term.is_tty and not self._is_managed:
            self.current = self.total
            self._render()
            print()


def progress_bar(total: int = 0, **kwargs) -> ProgressBar:
    """Convenience helper to create a ProgressBar."""
    return ProgressBar(total=total, **kwargs)


class Live:
    """
    Context manager for managing live-updating components (dashboards, multiple bars).
    """
    def __init__(self, refresh_rate: float = 0.1):
        self.refresh_rate = refresh_rate
        self.components = []
        self._stop_event = threading.Event()
        self._thread = None

    def add(self, component: Any):
        self.components.append(component)
        return component

    def __enter__(self):
        Term.hide_cursor()
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        Term.show_cursor()
        print() # Move to next line after finish

    def _run(self):
        last_lines = 0
        while not self._stop_event.is_set():
            output = []
            for comp in self.components:
                # Handle callables for dynamic layouts
                actual_comp = comp() if callable(comp) else comp
                
                if hasattr(actual_comp, 'render'):
                    res = actual_comp.render()
                else:
                    res = str(actual_comp)
                
                # Split multi-line results
                if "\n" in res:
                    output.extend(res.split("\n"))
                else:
                    output.append(res)
            
            # Move cursor back to start of previous render
            if last_lines > 0:
                Term.cursor_up(last_lines)
            
            # Clear lines and print new output
            for line in output:
                Term.clear_line()
                print(line)
            
            last_lines = len(output)
            time.sleep(self.refresh_rate)


class Fullscreen:
    """
    Context manager for full-screen interactive mode.
    Clears screen on enter and restores on exit.
    """
    def __init__(self, hide_cursor: bool = True):
        self.hide_cursor = hide_cursor

    def __enter__(self):
        Term.alt_buffer(True)
        Term.clear()
        Term.move_cursor(1, 1)
        if self.hide_cursor:
            Term.hide_cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hide_cursor:
            Term.show_cursor()
        Term.alt_buffer(False)


class Spinner:
    """
    Animated loading spinner (context manager or manual control).
    
    Usage:
        with Spinner("Loading data..."):
            time.sleep(2)
    
    Manual usage:
        s = Spinner("Processing")
        s.start()
        do_work()
        s.stop(success=True, final_msg="Done!")
    """
    
    def __init__(
        self,
        text: str = "Processing",
        color: str = "",
        frames: Tuple[str, ...] = None,
        speed: float = 0.08
    ):
        self.text = text
        self.color = color or CURRENT_THEME.primary
        self.frames = frames or Sym.DOTS
        self.speed = speed
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._success = True
        self._lock = threading.Lock()
        self._idx = 0
        self.active = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop(success=exc_type is None)

    def render(self) -> str:
        """Render the current frame of the spinner (for use in Live)."""
        f = self.frames[self._idx % len(self.frames)]
        self._idx += 1
        return f"{self.color}{f}{C.R} {self.text}"

    def frame(self) -> str:
        """Get the current frame as a string."""
        with self._lock:
            return self.render()

    def _animate(self):
        while not self._stop.wait(self.speed):
            sys.stdout.write(f"\r{self.frame()}...")
            sys.stdout.flush()
    
    def update(self, text: str):
        """Update spinner text while running."""
        with self._lock:
            self.text = text
    
    def start(self, text: str = None):
        """Start the spinner animation."""
        if text:
            self.update(text)
        if not self.active:
            self.active = True
            self._stop.clear()
            if Term.is_tty:
                Term.hide_cursor()
                self._thread = threading.Thread(target=self._animate, daemon=True)
                self._thread.start()
            else:
                print(f"{self.text}...")

    def stop(self, success: bool = True, final_msg: str = None):
        """Stop the spinner animation and print final status."""
        if self.active:
            self._stop.set()
            self._success = success
            
            if self._thread:
                self._thread.join(timeout=0.2)
                self._thread = None
            
            self.active = False
            
            if Term.is_tty:
                Term.show_cursor()
                Term.clear_line()
                
                msg = final_msg or self.text
                if self._success:
                    globals()['success'](msg)
                else:
                    globals()['error'](msg)
                sys.stdout.write("\n")
                sys.stdout.flush()

    def fail(self):
        """Mark as failed (affects exit message)."""
        self._success = False


@contextmanager
def spinner_task(text: str = "Processing", **kwargs):
    """
    Context manager helper for spinners.
    """
    sp = Spinner(text, **kwargs)
    sp.start()
    try:
        yield sp
        sp.stop(success=True)
    except Exception:
        sp.stop(success=False)
        raise


class Grid:
    """
    Responsive grid layout system.
    """
    def __init__(self, columns: int = 2, padding: int = 2, vertical_spacing: int = 0):
        self.columns = columns
        self.padding = padding
        self.vertical_spacing = vertical_spacing
        self.items = []

    def add(self, content: Any):
        self.items.append(content)

    def render(self, width: int = None) -> str:
        term_w = width or Term.width()
        
        # Adjust columns if width is too small
        actual_cols = self.columns
        if term_w < 40: actual_cols = 1
        elif term_w < 80: actual_cols = min(self.columns, 2)
        
        col_w = (term_w - (actual_cols - 1) * self.padding) // actual_cols
        
        # Group items into rows
        rows = [self.items[i:i + actual_cols] for i in range(0, len(self.items), actual_cols)]
        
        result_rows = []
        for row in rows:
            row_items = []
            for item in row:
                if hasattr(item, 'render'):
                    try:
                        text = item.render(width=col_w)
                    except TypeError:
                        text = item.render()
                else:
                    text = str(item)
                row_items.append(text)
            
            result_rows.append(cols(row_items, spacing=self.padding, widths=[col_w] * len(row_items)))
                
        return ("\n" * (self.vertical_spacing + 1)).join(result_rows)


def notification(msg: str, title: str = "Notification", color: str = C.CYN_B, duration: float = 2.0):
    """
    Display a temporary floating notification.
    """
    # This is a simple implementation that prints it at the bottom of the screen
    h = Term.height()
    content = box(msg, title=title, color=color, padding=0)
    lines = content.split('\n')
    
    # Save cursor, move to bottom, print, restore cursor
    sys.stdout.write("\033[s") # Save
    for i, line in enumerate(lines):
        Term.move_cursor(h - len(lines) + i, 1)
        Term.clear_line()
        sys.stdout.write(line)
    sys.stdout.flush()
    time.sleep(duration)
    # Clear notification
    for i in range(len(lines)):
        Term.move_cursor(h - len(lines) + i, 1)
        Term.clear_line()
    sys.stdout.write("\033[u") # Restore
    sys.stdout.flush()


def tooltip(text: str, target_text: str = "") -> str:
    """
    Return text with a 'tooltip' marker, or display a temporary 
    tooltip at the bottom of the screen if target_text is not provided.
    """
    if target_text:
        return f"{target_text} {C.D}({text}){C.R}"
    
    if not Term.is_tty:
        return ""
    
    # Move to bottom line
    Term.move_cursor(Term.height(), 1)
    Term.clear_line()
    sys.stdout.write(f"{C.BG_BLU}{C.WHT_B} {text} {C.R}")
    sys.stdout.flush()
    return ""


class Dashboard:
    """
    Layout manager for complex dashboards with responsive section support.
    """
    def __init__(self, mode: Layout = Layout.VERTICAL, spacing: int = 2, vertical_spacing: int = 1):
        self.mode = mode
        self.spacing = spacing
        self.vertical_spacing = vertical_spacing
        self.sections: List[Dict[str, Any]] = []

    def add(self, component: Any, title: str = "", weight: int = 1, boxed: bool = True):
        """Add a section to the dashboard with optional title and weight."""
        self.sections.append({
            "comp": component,
            "title": title,
            "weight": weight,
            "boxed": boxed
        })

    def render(self, width: int = None) -> str:
        """Render the dashboard based on its layout mode."""
        if not self.sections:
            return ""
            
        term_w = width or Term.width()
        
        if self.mode == Layout.VERTICAL:
            output = []
            for sec in self.sections:
                content = sec["comp"]
                if hasattr(content, 'render'):
                    try:
                        content = content.render(width=term_w)
                    except TypeError:
                        content = content.render()
                
                if sec["boxed"]:
                    output.append(box(content, title=sec["title"], width=term_w))
                else:
                    output.append(content)
            return ("\n" * (self.vertical_spacing + 1)).join(output)
            
        elif self.mode == Layout.HORIZONTAL:
            total_weight = sum(sec["weight"] for sec in self.sections)
            available_w = term_w - (self.spacing * (len(self.sections) - 1))
            
            rendered_comps = []
            widths = []
            
            # Use floating point weights to avoid rounding gaps
            remaining_w = available_w
            for i, sec in enumerate(self.sections):
                if i == len(self.sections) - 1:
                    sec_w = remaining_w
                else:
                    sec_w = int((sec["weight"] / total_weight) * available_w)
                    remaining_w -= sec_w
                
                widths.append(sec_w)
                
                content = sec["comp"]
                if hasattr(content, 'render'):
                    try:
                        content = content.render(width=sec_w)
                    except TypeError:
                        content = content.render()
                
                if sec["boxed"]:
                    rendered_comps.append(box(content, title=sec["title"], width=sec_w))
                else:
                    rendered_comps.append(content)
            
            return cols(rendered_comps, spacing=self.spacing, widths=widths)
            
        elif self.mode == Layout.GRID:
            # Responsive grid: 3 cols if > 120, 2 if > 80, 1 otherwise
            if term_w > 120: cols_count = 3
            elif term_w > 80: cols_count = 2
            else: cols_count = 1
            
            grid = Grid(columns=cols_count, padding=self.spacing, vertical_spacing=self.vertical_spacing)
            for sec in self.sections:
                content = sec["comp"]
                if sec["boxed"]:
                    # We don't pass width here, Grid.render will handle it
                    grid.add(box(content, title=sec["title"]))
                else:
                    grid.add(content)
            return grid.render(width=term_w)
            
        return ""


# ------------------------------------------------------------------------------
# 11. LOGGING & CLI
# ------------------------------------------------------------------------------

class Logger:
    """
    Structured logging for CLI applications.
    """
    def __init__(self, name: str = "APP", level: str = "INFO"):
        self.name = name
        self.level = level

    def log(self, level: str, msg: str, **kwargs):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        color = {
            "INFO": C.CYN,
            "WARN": C.YLW,
            "ERROR": C.RED,
            "DEBUG": C.D,
            "SUCCESS": C.GRN
        }.get(level, "")
        
        prefix = f"{C.D}[{now}]{C.R} {color}{level:7}{C.R} {C.B}{self.name}{C.R}:"
        print(f"{prefix} {msg}")
        if kwargs:
            items = list(kwargs.items())
            print(kvlist(items, padding=2))

    def info(self, msg: str, **kwargs): self.log("INFO", msg, **kwargs)
    def warn(self, msg: str, **kwargs): self.log("WARN", msg, **kwargs)
    def error(self, msg: str, **kwargs): self.log("ERROR", msg, **kwargs)
    def debug(self, msg: str, **kwargs): self.log("DEBUG", msg, **kwargs)
    def success(self, msg: str, **kwargs): self.log("SUCCESS", msg, **kwargs)


class CLI:
    """
    Simple command/argument parser integration.
    """
    def __init__(self, description: str = ""):
        self.description = description
        self.commands = {}

    def command(self, name: str, help_text: str = ""):
        def decorator(func):
            self.commands[name] = {"func": func, "help": help_text}
            return func
        return decorator

    def run(self):
        if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
            print(f"\n{C.B}{self.description}{C.R}\n")
            print(f"{C.U}Commands:{C.R}")
            for name, info in self.commands.items():
                print(f"  {C.CYN}{name:12}{C.R} {info['help']}")
            return

        cmd_name = sys.argv[1]
        if cmd_name in self.commands:
            try:
                # Basic arg parsing: pass remaining sys.argv as *args
                self.commands[cmd_name]["func"](*sys.argv[2:])
            except Exception as e:
                exception_box(e)
        else:
            print(f"{C.RED}Unknown command: {cmd_name}{C.R}")


def exception_box(e: Exception, show_traceback: bool = True):
    """Display a pretty exception box with optional traceback."""
    error_msg = f"{C.RED}{C.B}{type(e).__name__}: {C.R}{e}"
    
    if show_traceback:
        tb = traceback.format_exc()
        if "NoneType: None" in tb:
            content = error_msg
        else:
            # Clean up traceback lines
            tb_lines = []
            for line in tb.split('\n'):
                if not line.strip(): continue
                if "traceback.format_exc()" in line: continue
                
                # Highlight file paths and line numbers
                line = re.sub(r'File "(.*?)", line (\d+)', f'File {C.CYN}"\\1"{C.R}, line {C.YLW}\\2{C.R}', line)
                tb_lines.append(f"  {line}")
            
            content = f"{error_msg}\n\n{C.D}Traceback (most recent call last):{C.R}\n" + "\n".join(tb_lines)
    else:
        content = error_msg
        
    print("\n" + box(content, title=" ERROR ", color=C.RED, style=Border.HEAVY, padding=1))


# ------------------------------------------------------------------------------
# 12. INTERACTIVE PROMPTS
# ------------------------------------------------------------------------------

class Menu:
    """
    Standalone menu utility built on top of select().
    """
    def __init__(
        self,
        options: List[str],
        title: str = "Select an option:",
        default: int = 0,
        color: str = "",
        multi: bool = False,
        vim_keys: bool = True,
        search: bool = False
    ):
        self.options = options
        self.title = title
        self.default = default
        self.color = color
        self.multi = multi
        self.vim_keys = vim_keys
        self.search = search

    def run(self) -> Union[int, List[int]]:
        return select(
            self.options,
            title=self.title,
            default=self.default,
            color=self.color,
            multi=self.multi,
            vim_keys=self.vim_keys,
            search=self.search
        )

    def choose(self) -> Union[str, List[str], None]:
        res = self.run()
        if self.multi:
            return [self.options[i] for i in res] if res else []
        if res == -1:
            return None
        return self.options[res]

def select(
    options: List[str],
    title: str = "Select an option:",
    default: int = 0,
    color: str = "",
    multi: bool = False,
    vim_keys: bool = True,
    search: bool = False
) -> Union[int, List[int], List[str]]:
    """
    Interactive selection menu using arrow keys.
    
    Args:
        options: List of options to choose from
        title: Menu title
        default: Initial selection index
        color: Highlight color
        multi: Enable multiple selection
        vim_keys: Enable hjkl keys
        search: Enable type-ahead filtering
    
    Returns:
        Selected index (int) or list of indices if multi=True.
        Returns -1 or empty list if cancelled.
    """
    if not options:
        return [] if multi else -1
    
    if not color:
        color = CURRENT_THEME.primary
        
    idx = default
    selected = set()
    query = ""
    
    # Hide cursor while selecting
    Term.hide_cursor()
    
    try:
        while True:
            # Filter options if searching
            filtered = []
            for i, opt in enumerate(options):
                if not query or query.lower() in opt.lower():
                    filtered.append((i, opt))
            
            if not filtered:
                # If everything is filtered out, show a message
                print(f"\r{C.B}{title}{C.R} {C.D}(Filter: {query}){C.R}")
                print(f"  {C.RED}No matches found.{C.R}")
                num_lines = 2
            else:
                # Adjust index to stay within filtered results
                idx = max(0, min(idx, len(filtered) - 1))
                
                print(f"\r{C.B}{title}{C.R} {C.D}(Filter: {query}){C.R}" if search else f"\r{C.B}{title}{C.R}")
                for i, (orig_idx, opt) in enumerate(filtered):
                    is_current = (i == idx)
                    is_selected = (orig_idx in selected)
                    
                    marker = f"{color}{Sym.TRIANGLE}{C.R}" if is_current else " "
                    
                    if multi:
                        box_char = f"{color}[{Sym.CHECK}]{C.R}" if is_selected else "[ ]"
                        line = f" {marker} {box_char} {opt}"
                    else:
                        line = f" {marker} {opt}"
                    
                    # Highlight match
                    if query and query.lower() in opt.lower():
                        start = opt.lower().find(query.lower())
                        end = start + len(query)
                        highlighted = opt[:start] + C.B + C.U + opt[start:end] + C.R + opt[end:]
                        if is_current:
                            line = line.replace(opt, highlighted)
                        else:
                            line = line.replace(opt, highlighted)

                    if is_current:
                        print(f"{C.B}{line}{C.R}")
                    else:
                        print(line)
                num_lines = len(filtered) + 1
            
            # Read key
            key = Term.get_key()
            
            if key in (Key.UP, Key.K if vim_keys and not query else None):
                idx = (idx - 1) % len(filtered) if filtered else 0
            elif key in (Key.DOWN, Key.J if vim_keys and not query else None):
                idx = (idx + 1) % len(filtered) if filtered else 0
            elif key == Key.SPACE and multi:
                if filtered:
                    orig_idx, _ = filtered[idx]
                    if orig_idx in selected:
                        selected.remove(orig_idx)
                    else:
                        selected.add(orig_idx)
            elif key == Key.ENTER:
                if multi:
                    return sorted(list(selected))
                return filtered[idx][0] if filtered else -1
            elif key == Key.ESC:
                return [] if multi else -1
            elif key == Key.BACKSPACE and search:
                query = query[:-1]
            elif search and len(key) == 1 and key.isprintable():
                query += key
                idx = 0 # Reset selection on new search
                
            # Move cursor back up to redraw
            Term.cursor_up(num_lines)
            
    finally:
        # Move cursor to end of menu and show it
        # (Approximate - might leave some traces if filtered list was longer)
        # Better: clear all lines first
        for _ in range(num_lines):
            Term.clear_line()
            print()
        Term.show_cursor()


def confirm(
    message: str,
    default: bool = False,
    yes_char: str = "y",
    no_char: str = "n"
) -> bool:
    """
    Yes/No confirmation prompt.
    
    Args:
        message: Prompt message
        default: Default answer (True=yes, False=no)
        yes_char: Character for yes
        no_char: Character for no
    
    Returns:
        True for yes, False for no
    """
    y = yes_char.upper() if default else yes_char.lower()
    n = no_char.upper() if not default else no_char.lower()
    hint = f"{C.D}[{y}/{n}]{C.R}"
    
    try:
        answer = input(f"{message} {hint} ").strip().lower()
        
        if not answer:
            return default
        
        return answer.startswith(yes_char.lower())
        
    except (KeyboardInterrupt, EOFError):
        print()
        return False


def prompt(
    message: str,
    default: str = "",
    validator: Callable[[str], bool] = None,
    error_msg: str = "Invalid input.",
    hints: List[Tuple[Callable[[str], bool], str]] = None,
    password: bool = False,
    show_password: bool = False,
    history: List[str] = None
) -> str:
    """
    Advanced interactive text input prompt with real-time validation and hints.
    
    Args:
        message: Prompt message
        default: Initial value
        validator: Real-time validation function
        error_msg: Error message for invalid input
        hints: List of (validator_func, hint_text) tuples
        password: Hide characters (mask with *)
        show_password: If True, don't mask password
        history: Optional list for up/down history navigation
    """
    import pyperclip
    
    value = default
    pos = len(value)
    hist_idx = len(history) if history else 0
    num_hint_lines = 0
    
    # Save original cursor visibility
    Term.hide_cursor()
    
    try:
        while True:
            # Mask password
            display_val = value
            if password and not show_password:
                display_val = "*" * len(value)
            
            # Validation
            is_valid = validator(value) if validator else True
            val_color = CURRENT_THEME.success if is_valid else CURRENT_THEME.error
            
            # Render main prompt
            prefix = f"{message}: "
            sys.stdout.write(f"\r")
            Term.clear_line()
            
            colored_val = f"{val_color}{display_val}{C.R}"
            sys.stdout.write(f"{prefix}{colored_val}")
            
            # Render hints below
            if hints:
                print() # Move to next line
                for h_val, h_text in hints:
                    h_met = h_val(value)
                    h_color = CURRENT_THEME.success if h_met else CURRENT_THEME.dim
                    h_sym = Sym.CHECK if h_met else " "
                    print(f"  {h_color}{h_sym} {h_text}{C.R}")
                num_hint_lines = len(hints)
                # Move back up to the prompt line
                Term.cursor_up(num_hint_lines + 1)
            else:
                if not is_valid:
                    sys.stdout.write(f" {CURRENT_THEME.error}({error_msg}){C.R}")
            
            # Move cursor to correct position on the prompt line
            prefix_len = Term.vlen(prefix)
            sys.stdout.write(f"\r\033[{prefix_len + pos}C")
            
            Term.show_cursor()
            sys.stdout.flush()
            
            key = Term.get_key()
            
            # Clear hints before next iteration if they exist
            if hints:
                # Move to end of hints
                sys.stdout.write(f"\033[{num_hint_lines + 1}B\r")
                for _ in range(num_hint_lines + 1):
                    Term.cursor_up(1)
                    Term.clear_line()
                sys.stdout.write("\r")
            
            if key == Key.ENTER:
                if is_valid:
                    # Final clean render
                    Term.clear_line()
                    print(f"{prefix}{C.GRN}{display_val}{C.R}")
                    return value
            elif key == Key.ESC:
                if hints:
                    for _ in range(num_hint_lines): print()
                print()
                return default
            elif key == Key.LEFT:
                pos = max(0, pos - 1)
            elif key == Key.RIGHT:
                pos = min(len(value), pos + 1)
            elif key == Key.BACKSPACE:
                if pos > 0:
                    value = value[:pos-1] + value[pos:]
                    pos -= 1
            elif key == Key.DELETE:
                if pos < len(value):
                    value = value[:pos] + value[pos+1:]
            elif key == Key.HOME:
                pos = 0
            elif key == Key.END:
                pos = len(value)
            elif key == Key.UP and history:
                if hist_idx > 0:
                    hist_idx -= 1
                    value = history[hist_idx]
                    pos = len(value)
            elif key == Key.DOWN and history:
                if hist_idx < len(history) - 1:
                    hist_idx += 1
                    value = history[hist_idx]
                    pos = len(value)
                else:
                    hist_idx = len(history)
                    value = ""
                    pos = 0
            elif len(key) == 1 and key.isprintable():
                value = value[:pos] + key + value[pos:]
                pos += 1
            elif key == Key.TAB and password:
                show_password = not show_password
            elif key == "\x16":
                try:
                    pasted = pyperclip.paste()
                    if pasted:
                        value = value[:pos] + pasted + value[pos:]
                        pos += len(pasted)
                except Exception:
                    pass
            
            Term.hide_cursor()
            
    finally:
        Term.show_cursor()


# ------------------------------------------------------------------------------
# OPTIONAL: TEXTUAL INTEGRATION (DRAGGABLE WINDOWS)
# ------------------------------------------------------------------------------

def _require_textual():
    if not _TEXTUAL_AVAILABLE:
        raise RuntimeError(
            "Textual is not installed. Install with: pip install clui[textual]"
        )


class DraggableWindow(Container):
    """
    Draggable window widget for Textual.
    """
    DEFAULT_CSS = """
    DraggableWindow {
        border: heavy $accent;
        background: $panel;
        padding: 0 1;
        width: 40;
        height: 12;
    }
    DraggableWindow .titlebar {
        dock: top;
        height: 1;
        background: $boost;
        color: $text;
        layout: horizontal;
        align: left middle;
    }
    DraggableWindow .titlebar-label {
        width: 1fr;
    }
    DraggableWindow .titlebar-buttons {
        layout: horizontal;
        align: right middle;
    }
    DraggableWindow .titlebar-buttons Button {
        dock: right;
        width: 3;
        min-width: 3;
        height: 1;
        padding: 0 0;
        background: $surface;
        color: $text;
        border: none;
    }
    """

    def __init__(self, title: str = "Window", *args, **kwargs):
        _require_textual()
        super().__init__(*args, **kwargs)
        self.title = title
        self.dragging = False
        self.offset_x = 0
        self.offset_y = 0
        self._maximized = False
        self._minimized = False
        self._prev_size = None
        self._prev_offset = None

    def compose(self) -> ComposeResult:
        yield Container(
            Static(self.title, classes="titlebar-label"),
            Container(
                Button("–", id="min"),
                Button("□", id="max"),
                Button("x", id="close"),
                classes="titlebar-buttons"
            ),
            classes="titlebar"
        )
        yield Static("Content", classes="body")

    def on_mouse_down(self, event: MouseDown) -> None:
        if event.y == 0:
            self.capture_mouse()
            self.dragging = True
            self.offset_x = event.x
            self.offset_y = event.y

    def on_mouse_move(self, event: MouseMove) -> None:
        if self.dragging:
            self.styles.offset = (
                self.styles.offset.x + (event.x - self.offset_x),
                self.styles.offset.y + (event.y - self.offset_y),
            )

    def on_mouse_up(self, event: MouseUp) -> None:
        self.dragging = False
        self.release_mouse()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close":
            self.remove()
        elif event.button.id == "max":
            if not self._maximized:
                self._prev_size = (self.styles.width, self.styles.height)
                self._prev_offset = (self.styles.offset.x, self.styles.offset.y)
                self.styles.offset = (0, 0)
                self.styles.width = "100%"
                self.styles.height = "100%"
                self._maximized = True
            else:
                if self._prev_size:
                    self.styles.width, self.styles.height = self._prev_size
                if self._prev_offset:
                    self.styles.offset = self._prev_offset
                self._maximized = False
        elif event.button.id == "min":
            if not self._minimized:
                self._prev_size = (self.styles.width, self.styles.height)
                self.styles.height = 1
                self._minimized = True
            else:
                if self._prev_size:
                    self.styles.width, self.styles.height = self._prev_size
                self._minimized = False


class WindowDemo(App):
    CSS = """
    Screen {
        background: $background;
    }
    """

    def compose(self) -> ComposeResult:
        win = DraggableWindow("CLUI Window")
        yield win


def run_textual_demo():
    """
    Launch a minimal Textual demo showing a draggable window.
    """
    _require_textual()
    WindowDemo().run()


# ------------------------------------------------------------------------------
# DEMO
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    def run_demo():
        while True:
            Term.clear()
            banner = f"""{C.CYN_B}ULTIMATE CLI LIBRARY v3.0{C.R}
{C.D}Interactive Showcase & Documentation{C.R}"""
            print(box(banner, width=60, style=Border.DOUBLE, align=Align.CENTER))
            print()

            choice_idx = select(
                [
                    "Layouts (Grid, Boxes, Columns)",
                    "Tier 2 (Nested, Shadows, Overlays)",
                    "Responsive (Percentages, Adaptive)",
                    "Dashboards (Multi-pane, Complex)",
                    "Themes (Global Styling, Palettes)",
                    "Navigation (Tabs, View Switching)",
                    "Advanced Data (Trees, Grids)",
                    "Data Display (Tables, KV Lists)",
                    "Live Components (Spinners, Progress Bars)",
                    "Feedback (Status, Bullets, Dividers)",
                    "Interactive (Prompts, Tooltips)",
                    f"{C.RED}Exit Demo{C.R}"
                ],
                title="Choose a component to showcase:"
            )
            
            if choice_idx == -1: break
            
            # Get the string value for the choice
            choices = [
                "Layouts", "Tier 2", "Responsive", "Dashboards", "Themes", "Nav", "Advanced",
                "Data", "Live", "Feedback", "Interactive", "Exit"
            ]
            choice = choices[choice_idx] if choice_idx < len(choices) else "Exit"

            if "Exit" in choice:
                break

            Term.clear()
            if "Layouts" in choice:
                # ... existing layout demo ...
                print(divider(" Layouts Showcase ", color=C.CYN))
                grid = Grid(2, 2, width=70)
                grid.set_cell(0, 0, box("Top Left Cell", width=32, color=C.BLU))
                grid.set_cell(0, 1, box("Top Right Cell", width=32, color=C.GRN))
                grid.set_cell(1, 0, box("Bottom Left Cell", width=32, color=C.YLW))
                grid.set_cell(1, 1, box("Bottom Right Cell", width=32, color=C.MGT))
                print(grid.render())
                print()
                print(cols([
                    ["Left Column", "More text", "Third line"],
                    ["Middle Column", "Center aligned?", "Maybe"],
                    ["Right Column", "Aligned right", "End"]
                ], spacing=5))

            elif "Responsive" in choice:
                print(divider(" Responsive Layouts ", color=C.CYN_B))
                
                print(f"{C.CYN_B}1. Percentage Widths{C.R}")
                print(box("This box is exactly 50% of the terminal width.", width="50%", color=C.BLU))
                print(box("This box is 80% width and centered.", width="80%", align=Align.CENTER, color=C.GRN))
                print()
                
                print(f"{C.CYN_B}2. Percentage Columns{C.R}")
                print(cols(
                    ["Left (30%)", "Middle (40%)", "Right (30%)"],
                    widths=["30%", "40%", "30%"],
                    spacing=2
                ))
                print()
                
                print(f"{C.CYN_B}3. Adaptive Layout (Responsive Columns){C.R}")
                print(f"{C.D}Try resizing your terminal and run this demo again.{C.R}")
                adaptive = responsive_cols(
                    [
                        box("Column A\nAlways readable", width=25, color=C.YLW),
                        box("Column B\nAlways readable", width=25, color=C.MGT),
                        box("Column C\nAlways readable", width=25, color=C.CYN)
                    ],
                    threshold=85, # Switch to vertical if < 85 chars
                    spacing=2
                )
                print(adaptive)
                print()
                
                print(f"{C.CYN_B}4. Fluid Dividers{C.R}")
                print(divider(" 100% Divider ", width="100%", color=C.MGT))
                print(divider(" 50% Divider ", width="50%", color=C.MGT))

            elif "Tier 2" in choice:
                print(divider(" Tier 2 Features ", color=C.YLW))
                
                # Nested boxes demo
                inner1 = box("Inner Box 1", color=C.GRN, width=25)
                inner2 = box("Inner Box 2", color=C.YLW, width=25)
                nested = box(
                    cols([inner1, inner2], spacing=2),
                    title=" Nested Boxes ",
                    color=C.CYN,
                    padding=1
                )
                print(nested)
                print()
                
                # Shadow demo
                simple_box = box("This box has a shadow!", width=30, color=C.MGT)
                print(shadow(simple_box))
                print()
                
                # Overlay demo
                background = box(" " * 40 + "\n" * 5, title=" Background ", width=45)
                foreground = box("Overlay", color=C.RED_B, style=Border.HEAVY)
                composed = overlay(background, foreground, x=10, y=3)
                print(composed)

            elif "Live" in choice:
                print(divider(" Live Components Showcase ", color=C.MGT))
                
                # Tier 3 Demo: Multiple components inside a box
                s1 = Spinner("Downloading...", color=C.CYN)
                s2 = Spinner("Compiling...", color=C.YLW)
                pb1 = ProgressBar(100, label="Overall", width=25)
                pb2 = ProgressBar(100, label="Current", width=25, color=C.BLU)
                
                # Mark as managed so they don't print to stdout themselves
                pb1._is_managed = True
                pb2._is_managed = True
                
                with Live() as live:
                    # We use a lambda to wrap the components in a box dynamically
                    live.add(lambda: box(
                        cols([
                            [s1.render(), s2.render()],
                            [pb1.render(), pb2.render()]
                        ], spacing=4),
                        title=" System Status ",
                        width=70,
                        color=C.MGT
                    ))
                    
                    for i in range(101):
                        pb1.update(i)
                        pb2.update(i % 20 * 5)
                        time.sleep(0.05)
                
                print()
                success("All tasks finished!")

            elif "Dashboards" in choice:
                print(divider(" Dashboard Layouts (Tier 6) ", color=C.CYN_B))
                
                # 1. Horizontal Dashboard with Weights
                print(f"{C.B}1. Horizontal Weighted Layout (2:1:1){C.R}")
                db_h = Dashboard(mode=Layout.HORIZONTAL)
                db_h.add("Main Stats\nWeight: 2", title=" Primary ", weight=2)
                db_h.add("Secondary\nWeight: 1", title=" Side A ", weight=1)
                db_h.add("Tertiary\nWeight: 1", title=" Side B ", weight=1)
                print(db_h.render())
                print()
                
                # 2. Grid Dashboard
                print(f"{C.B}2. Auto-Grid Layout (Responsive Columns){C.R}")
                db_g = Dashboard(mode=Layout.GRID)
                db_g.add("Metric 1: 85%", title=" CPU ")
                db_g.add("Metric 2: 42%", title=" RAM ")
                db_g.add("Metric 3: OK", title=" DISK ")
                db_g.add("Metric 4: 120ms", title=" LATENCY ")
                print(db_g.render())
                print()
                
                # 3. Live Dashboard Demo
                print(f"{C.B}3. Live Multi-pane Dashboard{C.R}")
                print(f"{C.D}Press Ctrl+C to stop live demo...{C.R}")
                time.sleep(1)
                
                # Create components
                cpu_pb = ProgressBar(100, label="CPU", width=20, color=C.RED)
                ram_pb = ProgressBar(100, label="RAM", width=20, color=C.GRN)
                net_spinner = Spinner("Traffic", color=C.CYN)
                
                # Mark as managed
                cpu_pb._is_managed = True
                ram_pb._is_managed = True
                
                db_live = Dashboard(mode=Layout.HORIZONTAL)
                db_live.add(cpu_pb, title=" Processor ")
                db_live.add(ram_pb, title=" Memory ")
                db_live.add(net_spinner, title=" Network ")
                
                try:
                    with Live() as live:
                        live.add(db_live.render)
                        for i in range(50):
                            cpu_pb.update(20 + (i * 1.5) % 60)
                            ram_pb.update(40 + (i * 0.5) % 30)
                            time.sleep(0.1)
                except KeyboardInterrupt:
                    pass

            elif "Themes" in choice:
                print(divider(" Theme Management (Tier 7) ", color=C.CYN_B))
                
                theme_names = list(THEMES.keys())
                print(f"{C.B}Select a theme to see it in action:{C.R}")
                t_idx = select(theme_names)
                
                if t_idx != -1:
                    theme_name = theme_names[t_idx]
                    set_theme(theme_name)
                    
                    print(divider(f" Current Theme: {theme_name} "))
                    success(f"Successfully switched to {theme_name} theme!")
                    info("All components now use these colors.")
                    
                    print(box(
                        cols([
                            [
                                f"{C.B}Theme Colors:{C.R}",
                                f"Primary: {CURRENT_THEME.primary}███{C.R}",
                                f"Secondary: {CURRENT_THEME.secondary}███{C.R}",
                                f"Success: {CURRENT_THEME.success}███{C.R}"
                            ],
                            [
                                f"{C.B}Components:{C.R}",
                                progress(75, 100, width=20),
                                hline(20),
                                "Nested components follow suit."
                            ]
                        ]),
                        title=f" {theme_name} Preview ",
                        color=CURRENT_THEME.primary
                    ))
                    
                    print(f"\n{C.D}Press Enter to return to main menu...{C.R}")
                    input()

            elif "Nav" in choice:
                print(divider(" Navigation & Tabs (Tier 10) ", color=C.CYN_B))
                
                t_options = ["Overview", "Settings", "Logs", "Help"]
                active_t = 0
                
                while True:
                    Term.clear()
                    print(divider(" Interactive Tab Navigation ", color=C.CYN_B))
                    print(f"{C.D}Use Left/Right to switch tabs, ESC to exit demo.{C.R}\n")
                    
                    print(tabs(t_options, active_idx=active_t))
                    print("\n")
                    
                    # Tab content
                    if active_t == 0:
                        print(box(
                            cols([
                                [f"{C.B}System Status{C.R}", "All systems nominal", f"Uptime: {C.GRN}14d{C.R}"],
                                [progress(85, 100, label="CPU Usage")]
                            ]),
                            title=" Overview "
                        ))
                    elif active_t == 1:
                        print(kvlist([
                            ("Theme", "Neon"),
                            ("Auto-save", f"{C.GRN}Enabled{C.R}"),
                            ("Language", "Python 3.12")
                        ], key_color=C.CYN))
                    elif active_t == 2:
                        print(f"{C.D}[12:00:01] System started{C.R}")
                        print(f"{C.D}[12:05:22] Connection established{C.R}")
                        print(f"{C.YLW}[12:10:45] Warning: High memory usage{C.R}")
                    elif active_t == 3:
                        bullet("Use arrow keys to navigate")
                        bullet("Press ESC to return to menu")
                    
                    key = Term.get_key()
                    if key == Key.LEFT:
                        active_t = (active_t - 1) % len(t_options)
                    elif key == Key.RIGHT:
                        active_t = (active_t + 1) % len(t_options)
                    elif key == Key.ESC or key == Key.ENTER:
                        break

            elif "Advanced" in choice:
                print(divider(" Advanced Data Structures (Tier 8) ", color=C.CYN_B))
                
                # 1. Tree View
                print(f"{C.B}1. Hierarchical Tree View{C.R}")
                project_data = {
                    "src": {
                        "main.py": "12KB",
                        "utils": {
                            "network.py": "5KB",
                            "crypto.py": "8KB"
                        }
                    },
                    "tests": ["test_api.py", "test_core.py"],
                    "docs": "README.md",
                    "requirements.txt": "1KB"
                }
                print(tree(project_data, label=" my_project/ ", color=CURRENT_THEME.secondary))
                print()
                
                # 2. Grid System
                print(f"{C.B}2. Responsive Grid System{C.R}")
                g = Grid(columns=3, padding=4)
                g.add(box("Cell 1", color=C.RED, width=20))
                g.add(box("Cell 2", color=C.GRN, width=20))
                g.add(box("Cell 3", color=C.BLU, width=20))
                g.add(box("Cell 4", color=C.YLW, width=20))
                print(g.render())

            elif "Data" in choice:
                print(divider(" Data Display Helpers ", color=C.CYN_B))
                
                # 1. Sparklines
                print(f"{C.B}1. Sparklines (Inline Trends){C.R}")
                data_points = [1, 3, 2, 5, 4, 8, 3, 6, 7, 2, 4, 5]
                print(f"Server Load: {sparkline(data_points, color=C.GRN)} {C.D}(last 12h){C.R}")
                print(f"Memory:      {sparkline([8, 7, 6, 5, 4, 3, 2, 1], color=C.RED)} {C.D}(cleaning up){C.R}")
                print()
                
                # 2. Key-Value Lists
                print(f"{C.B}2. Key-Value Alignment{C.R}")
                stats = [
                    ("Status", f"{C.GRN}Online{C.R}"),
                    ("Uptime", "14d 2h 35m"),
                    ("Version", "3.0.0-alpha"),
                    ("Environment", "Production")
                ]
                print(kvlist(stats, key_color=C.CYN))
                print()
                
                # 3. Tables
                print(f"{C.B}3. Responsive Tables{C.R}")
                headers = ["ID", "Process", "CPU %", "Memory"]
                rows = [
                    ["1024", "python3 clui.py", "12.5", "45MB"],
                    ["2048", "vscode-server", "4.2", "512MB"],
                    ["4096", "docker-daemon", "0.8", "120MB"]
                ]
                print(table(headers, rows, color=CURRENT_THEME.secondary))
                print()
                
                # 4. Bar Charts
                print(f"{C.B}4. Horizontal Bar Charts{C.R}")
                browser_data = {
                    "Chrome": 64.5,
                    "Safari": 18.2,
                    "Edge": 4.5,
                    "Firefox": 3.2,
                    "Opera": 2.1
                }
                print(barchart(browser_data, color=C.CYN, width=30))

            elif "Feedback" in choice:
                print(divider(" Feedback Helpers ", color=C.YLW))
                success("Operation successful")
                info("Background task started")
                warn("Memory usage high")
                error("Connection lost")
                
                print()
                bullet("Main task")
                bullet("Sub-task", indent=1)
                bullet("Deep task", indent=2)

            elif "Interactive" in choice:
                print(divider(" Interactive Prompts & Validation Hints ", color=C.BLU))
                
                # Tier 4: Validation Hints Demo
                print(f"{C.CYN_B}Tier 4: Real-time Validation Hints{C.R}")
                print(f"{C.D}Try entering a weak vs strong password, or a valid email.{C.R}")
                print()

                # Password validation hints
                pw_hints = [
                    (lambda v: len(v) >= 8, "At least 8 characters"),
                    (lambda v: any(c.isupper() for c in v), "Contains uppercase"),
                    (lambda v: any(c.isdigit() for c in v), "Contains a digit"),
                    (lambda v: any(c in "!@#$%^&*" for c in v), "Contains special character (!@#$%^&*)")
                ]
                
                pwd = prompt(
                    "Set a new password", 
                    password=True, 
                    hints=pw_hints
                )
                
                # Email validation hints
                email_hints = [
                    (lambda v: "@" in v, "Must contain @"),
                    (lambda v: "." in v.split("@")[-1] if "@" in v else False, "Domain must have a dot"),
                    (lambda v: len(v.split(".")[-1]) >= 2 if "." in v else False, "Valid TLD (e.g. .com)")
                ]
                
                email = prompt(
                    "Enter email address",
                    hints=email_hints,
                    validator=lambda v: "@" in v and "." in v
                )

                print()
                name = prompt("Finally, enter your name", default="Guest")
                is_cool = confirm("Are you having fun?")
                
                print()
                if is_cool:
                    success(f"Excellent, {name}! Your account for {email} is ready.")
                else:
                    info(f"We'll try harder next time, {name}.")
                
                print()
                print(f"Check the bottom of the screen for a tooltip! {tooltip('This is a temporary tooltip')}")
                time.sleep(2)

            print()
            input(f"{C.D}Press Enter to return to menu...{C.R}")

    try:
        run_demo()
    except KeyboardInterrupt:
        print("\nDemo terminated.")
