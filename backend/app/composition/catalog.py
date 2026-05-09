"""
Background catalog: scans backend/app/backgrounds/<category>/ at startup, reads
optional sidecar .json metadata files, and exposes lookup helpers.

Folder layout:
    backgrounds/
      oceano/
        wave1.jpg
        wave1.json        # optional metadata
        ...
      marmol/
      ...

Sidecar JSON schema (all keys optional, defaults are sensible):
    {
      "ground_y": 0.72,                  # relative y of horizon/ground (0..1)
      "light_dir": [0.4, -0.6],          # 2D unit-ish vector; OpenCV y is down
      "reflective": true,                # enable vertical reflection
      "reflective_type": "glossy",       # "glossy" | "matte"
      "label": "Mármol pulido"           # optional display label
    }
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
THUMB_SIZE = 256


@dataclass
class BackgroundEntry:
    id: str               # "<category>/<filename_without_ext>"
    category: str
    name: str             # original filename without extension
    path: str             # absolute path to source image
    label: str
    reflective: bool = False
    reflective_type: str = "matte"
    ground_y: float | None = None
    light_dir: tuple[float, float] | None = None

    def to_public(self) -> dict:
        d = asdict(self)
        d.pop("path")
        d["thumb_url"]  = f"/backgrounds/thumb/{self.category}/{self.name}"
        d["full_url"]   = f"/backgrounds/full/{self.category}/{self.name}"
        return d


class BackgroundCatalog:
    def __init__(self, root: Path):
        self.root: Path = root
        self._entries: dict[str, BackgroundEntry] = {}
        self._by_category: dict[str, list[BackgroundEntry]] = {}

    def scan(self) -> None:
        self._entries.clear()
        self._by_category.clear()
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            print(f"📂 Carpeta de backgrounds creada (vacía): {self.root}")
            return

        for category_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            category = category_dir.name
            entries: list[BackgroundEntry] = []
            for img_path in sorted(category_dir.iterdir()):
                if img_path.suffix.lower() not in VALID_EXTS:
                    continue
                meta_path = img_path.with_suffix(".json")
                meta: dict = {}
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception as exc:
                        print(f"⚠️  Metadatos JSON inválidos en {meta_path}: {exc}")

                name = img_path.stem
                entry = BackgroundEntry(
                    id=f"{category}/{name}",
                    category=category,
                    name=name,
                    path=str(img_path.resolve()),
                    label=meta.get("label", name.replace("_", " ").title()),
                    reflective=bool(meta.get("reflective", False)),
                    reflective_type=str(meta.get("reflective_type", "matte")),
                    ground_y=meta.get("ground_y"),
                    light_dir=tuple(meta["light_dir"]) if isinstance(meta.get("light_dir"), (list, tuple)) and len(meta["light_dir"]) == 2 else None,
                )
                self._entries[entry.id] = entry
                entries.append(entry)

            if entries:
                self._by_category[category] = entries

        total = sum(len(v) for v in self._by_category.values())
        print(f"📚 Catálogo de backgrounds: {total} imágenes en {len(self._by_category)} categorías")

    def list_grouped(self) -> dict[str, list[dict]]:
        return {cat: [e.to_public() for e in entries] for cat, entries in self._by_category.items()}

    def get(self, background_id: str) -> BackgroundEntry | None:
        return self._entries.get(background_id)

    def get_path(self, category: str, name: str) -> Path | None:
        entry = self._entries.get(f"{category}/{name}")
        return Path(entry.path) if entry else None
