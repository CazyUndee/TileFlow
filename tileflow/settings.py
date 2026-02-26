from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(slots=True)
class TileFlowSettings:
    ktransformers_path: Optional[str] = None


class SettingsStore:
    def __init__(self, home: Optional[Path] = None) -> None:
        self.home = home or (Path.home() / ".tileflow")
        self.path = self.home / "settings.json"
        self.home.mkdir(parents=True, exist_ok=True)

    def load(self) -> TileFlowSettings:
        if not self.path.exists():
            return TileFlowSettings()
        try:
            raw: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            backup = self.path.with_suffix(self.path.suffix + ".corrupt")
            try:
                shutil.copyfile(self.path, backup)
            except Exception:
                pass
            self.save(TileFlowSettings())
            return TileFlowSettings()
        if not isinstance(raw, dict):
            self.save(TileFlowSettings())
            return TileFlowSettings()
        return TileFlowSettings(**raw)

    def save(self, settings: TileFlowSettings) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
        tmp.replace(self.path)
