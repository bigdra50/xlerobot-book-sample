"""データセットのメタデータと実データが食い違っていないかを照合する。

記録が途中で落ちると、meta/info.json、meta/episodes、data/*.parquet、videos/*.mp4 が
互いにずれる。読み込みでエラーになるものもあるが、ずれたまま学習できてしまう組み合わせもあるので、
push と学習の前に実行する。

1 つでも FAIL が出たら、先へ進まない。

    python check_dataset.py <dataset_root>
    python check_dataset.py <dataset_root> --deep   # mp4 を全フレームデコードして照合
    python check_dataset.py <dataset_root> --verbose

exit code: 全部 PASS で 0、1 つでも FAIL で 1。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import av
import numpy as np

import dataset_io

# timestamp は frame_index / fps の理論値と一致するはず。
# 浮動小数の丸め分だけ許す。LeRobotDataset の tolerance_s 既定値と同じ。
TIMESTAMP_TOL_S = 1e-4


@dataclass
class Check:
    """1 つの検査項目。per-episode の検査は failures にエピソード番号を積む。"""

    name: str
    total: int = 0
    failures: list[str] = field(default_factory=list)
    detail: str = ""

    def add(self, ok: bool, label: str = "") -> None:
        self.total += 1
        if not ok:
            self.failures.append(label)

    @property
    def ok(self) -> bool:
        return not self.failures

    def line(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        body = self.detail or f"{self.total - len(self.failures)}/{self.total} ok"
        if self.failures:
            shown = ", ".join(self.failures[:10])
            more = "" if len(self.failures) <= 10 else f", ... +{len(self.failures) - 10}"
            body = f"{body}  NG: {shown}{more}"
        return f"[{status}] {self.name:<34} {body}"


def check_totals(info: dict, data, episodes) -> list[Check]:
    rows = data.num_rows
    eps = episodes.num_rows
    a = Check("info.total_frames vs data rows")
    a.detail = f"info={info.get('total_frames')} data={rows}"
    a.add(info.get("total_frames") == rows)
    b = Check("info.total_episodes vs meta rows")
    b.detail = f"info={info.get('total_episodes')} meta={eps}"
    b.add(info.get("total_episodes") == eps)
    return [a, b]


def check_episodes(info: dict, data, episodes) -> list[Check]:
    fps = float(info["fps"])
    rows_c = Check("episode row count")
    span_c = Check("episode index span")
    ts_c = Check("episode timestamp")
    fi_c = Check("episode frame_index sequence")
    idx_c = Check("episode index sequence")

    for ep in dataset_io.episode_list(episodes):
        table = dataset_io.episode_rows(data, ep.index)
        label = str(ep.index)
        rows_c.add(table.num_rows == ep.length, label)
        span_c.add(ep.to_index - ep.from_index == ep.length, label)
        if table.num_rows == 0:
            ts_c.add(False, label)
            fi_c.add(False, label)
            idx_c.add(False, label)
            continue

        timestamp = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
        frame_index = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
        index = np.asarray(table.column("index").to_pylist(), dtype=np.int64)

        expected_ts = frame_index / fps
        ts_c.add(bool(np.all(np.abs(timestamp - expected_ts) < TIMESTAMP_TOL_S)), label)
        fi_c.add(bool(np.array_equal(frame_index, np.arange(table.num_rows))), label)
        idx_c.add(
            bool(np.array_equal(index, np.arange(ep.from_index, ep.from_index + table.num_rows))),
            label,
        )
    return [rows_c, span_c, ts_c, fi_c, idx_c]


def _count_frames(path: Path) -> int:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return sum(1 for packet in container.demux(stream) for _ in packet.decode())


def _decode_first(path: Path) -> bool:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            for _ in packet.decode():
                return True
    return False


def check_videos(root: Path, info: dict, episodes, *, deep: bool) -> list[Check]:
    keys = dataset_io.video_keys(info)
    if not keys:
        return []
    exists_c = Check("video file exists")
    open_c = Check("video first frame decode")
    count_c = Check("video frame count (--deep)")

    for key in keys:
        if not dataset_io.has_video_columns(episodes, key):
            exists_c.add(False, f"{key}: メタに chunk/file 列が無い")
            continue
        for group in dataset_io.group_by_video_file(episodes, key):
            path = dataset_io.video_path(root, info, key, group.chunk_index, group.file_index)
            label = f"{key} chunk{group.chunk_index:03d}/file{group.file_index:03d}"
            if not path.is_file():
                exists_c.add(False, label)
                continue
            exists_c.add(True, label)
            try:
                open_c.add(_decode_first(path), label)
            except Exception as exc:  # noqa: BLE001 - コンテナ破損をそのまま出す
                open_c.add(False, f"{label} ({exc})")
                continue
            if deep:
                try:
                    frames = _count_frames(path)
                except Exception as exc:  # noqa: BLE001
                    count_c.add(False, f"{label} ({exc})")
                    continue
                count_c.add(
                    frames == group.expected_frames,
                    f"{label} video={frames} parquet={group.expected_frames}",
                )
    checks = [exists_c, open_c]
    if deep:
        checks.append(count_c)
    return checks


def run(root: Path, *, deep: bool, verbose: bool) -> int:
    info = dataset_io.load_info(root)
    data = dataset_io.load_data(root)
    episodes = dataset_io.load_episodes(root)

    print(f"== structure: {root.name} ==")
    print(f"codebase_version={info.get('codebase_version')} fps={info.get('fps')}")

    checks: list[Check] = []
    checks += check_totals(info, data, episodes)
    checks += check_episodes(info, data, episodes)
    checks += check_videos(root, info, episodes, deep=deep)

    for check in checks:
        if verbose or not check.ok:
            print(check.line())

    failed = [c for c in checks if not c.ok]
    print(f"--- {len(checks) - len(failed)}/{len(checks)} checks PASS")
    print(f"structure: {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="meta/info.json を含むデータセット root")
    parser.add_argument("--deep", action="store_true", help="mp4 を全フレームデコードして照合する")
    parser.add_argument("--verbose", action="store_true", help="PASS した項目も表示する")
    args = parser.parse_args()
    return run(args.root.resolve(), deep=args.deep, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
