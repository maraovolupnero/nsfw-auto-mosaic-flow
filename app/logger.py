from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any


FIELDS = [
    "filename", "status", "detection_count", "masked_count", "detected_classes", "masked_classes",
    "confidence", "pixel_size", "expand_ratio", "feather_px", "error_message",
]


class CsvResultLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        if self.path.exists():
            with self.path.open("r", newline="", encoding="utf-8-sig") as handle:
                current_fields = next(csv.reader(handle), [])
            if current_fields != FIELDS:
                archived = self.path.with_name(f"{self.path.stem}_legacy_{datetime.now():%Y%m%d_%H%M%S}{self.path.suffix}")
                self.path.replace(archived)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8-sig") as handle:
                csv.DictWriter(handle, fieldnames=FIELDS).writeheader()

    def write(self, **values: Any) -> None:
        row = {field: values.get(field, "") for field in FIELDS}
        with self._lock, self.path.open("a", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writerow(row)
