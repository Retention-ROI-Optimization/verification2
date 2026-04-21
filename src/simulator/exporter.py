from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def _write_one(df: pd.DataFrame, path: Path, file_format: str) -> Path:
    if file_format == "parquet":
        try:
            df.to_parquet(path.with_suffix(".parquet"), index=False)
            return path.with_suffix(".parquet")
        except Exception:
            df.to_csv(path.with_suffix(".csv"), index=False)
            return path.with_suffix(".csv")

    df.to_csv(path.with_suffix(".csv"), index=False)
    return path.with_suffix(".csv")


def export_tables(
    tables: Dict[str, pd.DataFrame],
    output_dir: str,
    file_format: str = "csv",
) -> Dict[str, str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    exported = {}
    for name, df in tables.items():
        exported_path = _write_one(df=df, path=Path(output_dir) / name, file_format=file_format)
        exported[name] = str(exported_path)

    return exported
