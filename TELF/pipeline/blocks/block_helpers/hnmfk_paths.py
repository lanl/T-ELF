# post_processing/Peacock/node_sources.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


from ....factorization import HNMFk  # adjust relative path if needed


@dataclass(frozen=True)
class Node:
    dir: Path
    csv: Path


class NodeSource:
    def iter_nodes(self) -> Iterator[Node]:
        raise NotImplementedError


class HNMFkNodeSource(NodeSource):
    def __init__(self, experiment_path: Path, *, only_existing_csv: bool = True) -> None:
        self.experiment_path = Path(experiment_path).expanduser().resolve()
        self.only_existing_csv = only_existing_csv

    def iter_nodes(self) -> Iterator[Node]:
        model = HNMFk(experiment_name=str(self.experiment_path))
        model.load_model()
        seen = set()
        for node in model.traverse_nodes():
            node_dir = Path(node["node_save_path"]).resolve().parent
            if node_dir in seen:
                continue
            seen.add(node_dir)
            # prefer latest cluster file in that directory
            csvs = sorted(node_dir.glob("cluster_for_k=*.csv"), key=lambda p: p.stat().st_mtime)
            if csvs:
                yield Node(dir=node_dir, csv=csvs[-1])
            elif not self.only_existing_csv:
                # allow “expected” path even if it doesn’t exist yet
                yield Node(dir=node_dir, csv=node_dir / "cluster_for_k=UNKNOWN.csv")


class GlobNodeSource(NodeSource):
    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def iter_nodes(self) -> Iterator[Node]:
        for csv in sorted(self.root.rglob("cluster_for_k=*.csv")):
            yield Node(dir=csv.parent, csv=csv)
