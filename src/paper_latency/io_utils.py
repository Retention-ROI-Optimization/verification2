from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    return resolved


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_dataframe(path: str | Path, df: pd.DataFrame) -> Path:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    df.to_csv(resolved, index=False)
    return resolved


def cached_dataframe(path: str | Path) -> pd.DataFrame | None:
    resolved = Path(path)
    if not resolved.exists():
        return None
    return pd.read_csv(resolved)


def seed_dir(root: str | Path, seed: int) -> Path:
    return ensure_dir(Path(root) / f'seed_{int(seed)}')


def scenario_dir(root: str | Path, seed: int, family: str) -> Path:
    safe_family = family.replace(' ', '_').replace('/', '_')
    return ensure_dir(seed_dir(root, seed) / safe_family)
