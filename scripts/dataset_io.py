"""LeRobotDataset v3.0 のメタデータと実データを、LeRobot 本体を介さずに読む。

依存は pyarrow だけ。

LeRobotDataset 経由で読まないのは、壊れたデータセットが load の時点で例外になり、
どこがどう壊れているかを列挙できないため。check_dataset.py は壊れている前提で全項目を数えるので、ファイルを直接読む。

対象は codebase_version v3.0（meta/episodes/*.parquet を持つ形）。
v2.1 のデータセットは meta/episodes.jsonl なのでこのままでは読めない。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


@dataclass(frozen=True)
class Episode:
    """meta/episodes の 1 行。dataset_from_index/to_index は data 全体での行番号。"""

    index: int
    length: int
    from_index: int
    to_index: int


@dataclass
class VideoFile:
    """1 本の mp4 と、そこへ詰められたエピソード。

    v3.0 は複数エピソードを 1 本の mp4 に連結する。
    フレーム数の照合は「この mp4 に入る全エピソードの length 合計」と比べる。
    エピソード 1 本の length と mp4 全体のフレーム数を比べると必ず落ちる。
    """

    chunk_index: int
    file_index: int
    episodes: list[Episode]

    @property
    def expected_frames(self) -> int:
        return sum(ep.length for ep in self.episodes)


def load_info(root: Path) -> dict:
    path = root / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} が無い。データセット root を指定しているか確認する")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_all(paths: list[Path]) -> pa.Table:
    if not paths:
        return pa.table({})
    return pa.concat_tables([pq.read_table(p) for p in paths], promote_options="default")


def load_data(root: Path) -> pa.Table:
    """data/**/*.parquet を chunk/file 順に連結して返す。"""
    return _read_all(sorted((root / "data").rglob("*.parquet")))


def load_episodes(root: Path) -> pa.Table:
    """meta/episodes/**/*.parquet を連結して返す。"""
    return _read_all(sorted((root / "meta" / "episodes").rglob("*.parquet")))


def episode_list(episodes: pa.Table) -> list[Episode]:
    return [
        Episode(
            index=int(episodes.column("episode_index")[row].as_py()),
            length=int(episodes.column("length")[row].as_py()),
            from_index=int(episodes.column("dataset_from_index")[row].as_py()),
            to_index=int(episodes.column("dataset_to_index")[row].as_py()),
        )
        for row in range(episodes.num_rows)
    ]


def video_keys(info: dict) -> list[str]:
    return [k for k, f in info.get("features", {}).items() if f.get("dtype") == "video"]


def video_path(root: Path, info: dict, key: str, chunk_index: int, file_index: int) -> Path:
    template = info.get("video_path")
    if not template:
        raise ValueError("info.json に video_path テンプレートが無い")
    return root / template.format(video_key=key, chunk_index=chunk_index, file_index=file_index)


def has_video_columns(episodes: pa.Table, key: str) -> bool:
    return f"videos/{key}/chunk_index" in episodes.column_names


def group_by_video_file(episodes: pa.Table, key: str) -> list[VideoFile]:
    """エピソード行を、出力先の mp4（chunk, file）ごとにまとめる。"""
    groups: dict[tuple[int, int], VideoFile] = {}
    order: list[tuple[int, int]] = []
    for row, ep in enumerate(episode_list(episodes)):
        chunk_index = int(episodes.column(f"videos/{key}/chunk_index")[row].as_py())
        file_index = int(episodes.column(f"videos/{key}/file_index")[row].as_py())
        ident = (chunk_index, file_index)
        if ident not in groups:
            groups[ident] = VideoFile(chunk_index, file_index, [])
            order.append(ident)
        groups[ident].episodes.append(ep)
    return [groups[i] for i in order]


def episode_rows(data: pa.Table, episode_index: int) -> pa.Table:
    return data.filter(pc.equal(data.column("episode_index"), episode_index))
