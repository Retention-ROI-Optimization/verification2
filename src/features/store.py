from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd


@dataclass
class FeatureStorePaths:
    feature_csv_path: Path
    metadata_path: Path


class FileFeatureStore:
    """Simple file-based feature store."""

    def __init__(self, root_dir: str | Path = 'data/feature_store') -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def build_paths(self, dataset_name: str = 'customer_features') -> FeatureStorePaths:
        safe = dataset_name.replace(' ', '_').replace('/', '_')
        return FeatureStorePaths(
            feature_csv_path=self.root_dir / f'{safe}.csv',
            metadata_path=self.root_dir / f'{safe}_metadata.json',
        )

    def save(self, features: pd.DataFrame, metadata: Dict[str, Any], dataset_name: str = 'customer_features') -> FeatureStorePaths:
        paths = self.build_paths(dataset_name)
        features.to_csv(paths.feature_csv_path, index=False)
        paths.metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8',
        )
        return paths

    def load(self, dataset_name: str = 'customer_features') -> tuple[pd.DataFrame, Dict[str, Any]]:
        paths = self.build_paths(dataset_name)
        features = pd.read_csv(paths.feature_csv_path)
        metadata = json.loads(paths.metadata_path.read_text(encoding='utf-8'))
        return features, metadata
