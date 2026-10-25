"""host に ZMQ で関節の目標を送り、腕と head を 1 コマンドずつ動かす。Mac で走らせる。

    uv run arm-zmq status
    uv run arm-zmq move lift+=5 elbow-=10 [--speed 20]
    uv run arm-zmq move pan=0 wrist_flex=30 right_arm_gripper=10

host を --observe.port=5557 で起動しておく（host/xlerobot_host.py）。
観測は PUB の口（5557）から読み、目標は指令の口（5555）へ送る。host が動いたまま使える。

関節の名前は短い名前（pan、lift、elbow、wrist_flex、wrist_roll、gripper は左腕、head_pan、head_tilt）か、
host の名前（right_arm_elbow_flex など）で書く。
角度は、キャリブレーションの範囲の中央を 0 とするモーターの度。
出力は 1 行 1 つの JSON で、最後の行が結果になる。終了コードは 0 が完了、2 が目標に届かず、3 が追いつかないので停止。

安全の約束:
- 目標はキャリブレーションの範囲から margin 内側へ切り詰める
- 動かす関節を同時に、指定した速さで線形に動かす。速さの上限は 60 deg/s
- 指令した角度から実際の角度が limit 度以上遅れた観測が 3 回続いたら、その場の角度を目標にして止める。
  host の観測は電流を持たないので、押し当てたことを遅れで見分ける
- 送るのは動かす関節の目標だけにする。台車の速度は送らない（host の車輪の口がつながっていれば、台車はその口に従う）
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

STEPS_PER_DEG = 4096.0 / 360.0
SHORT = {
    "pan": "left_arm_shoulder_pan",
    "lift": "left_arm_shoulder_lift",
    "elbow": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
    "head_pan": "head_motor_1",
    "head_tilt": "head_motor_2",
}
FULL = set(SHORT.values()) | {
    n.replace("left_arm_", "right_arm_") for n in SHORT.values() if n.startswith("left")
}
MAX_SPEED_DPS = 60.0
# host のキャリブレーション JSON（Pi の ~/.cache/huggingface/lerobot/calibration/robots/xlerobot/<robot.id>.json）を
# この場所へ写しておく。別の場所は --calibration で指す
DEFAULT_CAL = Path(__file__).resolve().parents[2] / "runs" / "calibration" / "follower_xlerobot.json"


def resolve(name: str) -> str:
    full = SHORT.get(name, name)
    if full not in FULL:
        raise ValueError(f"知らない関節の名前: {name}")
    return full


def _half_range_deg(name: str, cal: dict) -> float:
    c = cal[name]
    return (c["range_max"] - c["range_min"]) / 2.0 / STEPS_PER_DEG


def norm_to_deg(name: str, value: float, cal: dict) -> float:
    """host の値（腕と head は −100〜100、gripper は 0〜100）を、範囲の中央を 0 とする度にする。"""
    half = _half_range_deg(name, cal)
    if name.endswith("gripper"):
        return (value / 50.0 - 1.0) * half
    return value / 100.0 * half


def deg_to_norm(name: str, deg: float, cal: dict) -> float:
    half = _half_range_deg(name, cal)
    if name.endswith("gripper"):
        return (deg / half + 1.0) * 50.0
    return deg / half * 100.0


def clamp_deg(name: str, deg: float, cal: dict, margin_deg: float) -> float:
    limit = _half_range_deg(name, cal) - margin_deg
    return max(-limit, min(limit, deg))


_TARGET = re.compile(r"^([a-z_0-9]+)(=|\+=|-=)(-?[0-9]+(?:\.[0-9]+)?)$")


def parse_targets(args: list[str], current_deg: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for arg in args:
        m = _TARGET.match(arg)
        if m is None:
            raise ValueError(f"目標の書き方が違う: {arg}（例: elbow+=5、pan=0）")
        name, op, value = resolve(m.group(1)), m.group(2), float(m.group(3))
        base = current_deg[name]
        out[name] = value if op == "=" else base + value if op == "+=" else base - value
    return out


def ramp(start: dict[str, float], target: dict[str, float], speed_dps: float, rate_hz: float) -> list[dict]:
    """いちばん大きく動く関節が speed_dps になるように、全部の関節を同時に線形に動かす目標の列。"""
    largest = max(abs(target[k] - start[k]) for k in target)
    n = max(1, math.ceil(largest / speed_dps * rate_hz))
    steps = [{k: start[k] + (target[k] - start[k]) * i / n for k in target} for i in range(1, n)]
    return [*steps, dict(target)]


class Lagging:
    """指令した角度に、実際の角度が追いつかない状態が続いたら、その関節の名前を返す。"""

    def __init__(self, limit_deg: float, count: int) -> None:
        self.limit_deg = limit_deg
        self.count = count
        self.n = 0

    def check(self, commanded: dict[str, float], present: dict[str, float]) -> str | None:
        worst = max(commanded, key=lambda k: abs(commanded[k] - present[k]))
        if abs(commanded[worst] - present[worst]) < self.limit_deg:
            self.n = 0
            return None
        self.n += 1
        return worst if self.n >= self.count else None


# --- ここから下は host とのやり取り ---


class HostLink:
    def __init__(self, host: str, obs_port: int, cmd_port: int) -> None:
        import zmq

        self.zmq = zmq
        self.ctx = zmq.Context()
        self.obs = self.ctx.socket(zmq.SUB)
        self.obs.setsockopt(zmq.CONFLATE, 1)
        self.obs.setsockopt_string(zmq.SUBSCRIBE, "")
        self.obs.connect(f"tcp://{host}:{obs_port}")
        self.cmd = self.ctx.socket(zmq.PUSH)
        self.cmd.setsockopt(zmq.CONFLATE, 1)
        self.cmd.connect(f"tcp://{host}:{cmd_port}")

    def observe(self, timeout_s: float = 2.0) -> dict[str, float]:
        if not self.obs.poll(int(timeout_s * 1000)):
            raise TimeoutError("host の観測が届かない。host を --observe.port 付きで起動しているか確かめる")
        obs = json.loads(self.obs.recv_string())
        return {k.removesuffix(".pos"): float(v) for k, v in obs.items() if k.endswith(".pos")}

    def send(self, goals_norm: dict[str, float]) -> None:
        self.cmd.send_string(json.dumps({f"{k}.pos": v for k, v in goals_norm.items()}))

    def close(self) -> None:
        self.obs.close(linger=0)
        self.cmd.close(linger=500)
        self.ctx.term()


def _emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def _degrees(values: dict[str, float], cal: dict) -> dict[str, float]:
    return {k: round(norm_to_deg(k, v, cal), 1) for k, v in values.items() if k in cal}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="arm-zmq")
    ap.add_argument("--host", default="xlerobot-pi.local")
    ap.add_argument("--obs-port", type=int, default=5557)
    ap.add_argument("--cmd-port", type=int, default=5555)
    ap.add_argument("--calibration", type=Path, default=DEFAULT_CAL)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    mv = sub.add_parser("move")
    mv.add_argument("targets", nargs="+")
    mv.add_argument("--speed", type=float, default=20.0, help="deg/s（上限 60）")
    mv.add_argument("--rate", type=float, default=20.0)
    mv.add_argument("--margin", type=float, default=3.0, help="範囲の端から残す度")
    mv.add_argument("--lag-limit", type=float, default=12.0, help="止める遅れの度")
    mv.add_argument("--settle", type=float, default=1.5, help="目標を送り終えてから待つ秒")
    args = ap.parse_args(argv)
    cal = json.loads(args.calibration.expanduser().read_text())
    link = HostLink(args.host, args.obs_port, args.cmd_port)
    try:
        now = link.observe()
        if args.cmd == "status":
            _emit({"event": "status", "deg": _degrees(now, cal)})
            return 0
        start = {k: norm_to_deg(k, v, cal) for k, v in now.items() if k in cal}
        wanted = parse_targets(args.targets, start)
        target = {k: clamp_deg(k, v, cal, args.margin) for k, v in wanted.items()}
        clamped = [k for k in target if abs(target[k] - wanted[k]) > 1e-6]
        speed = min(args.speed, MAX_SPEED_DPS)
        _emit(
            {
                "event": "start",
                "from_deg": {k: round(start[k], 1) for k in target},
                "to_deg": {k: round(v, 1) for k, v in target.items()},
                "clamped": clamped,
            }
        )
        lag = Lagging(args.lag_limit, 3)
        period = 1.0 / args.rate
        steps = ramp({k: start[k] for k in target}, target, speed, args.rate)
        commanded = dict(steps[0])
        present = dict(start)
        t_end = None
        i = 0
        while True:
            t0 = time.monotonic()
            if i < len(steps):
                commanded = steps[i]
                i += 1
                if i == len(steps):
                    t_end = t0 + args.settle
            link.send({k: deg_to_norm(k, v, cal) for k, v in commanded.items()})
            try:
                present = {k: norm_to_deg(k, v, cal) for k, v in link.observe(0.5).items() if k in cal}
            except TimeoutError:
                pass
            stuck = lag.check(commanded, present)
            if stuck is not None:
                hold = {k: present[k] for k in target}
                link.send({k: deg_to_norm(k, v, cal) for k, v in hold.items()})
                _emit(
                    {
                        "event": "stopped",
                        "reason": "lagging",
                        "joint": stuck,
                        "deg": {k: round(v, 1) for k, v in hold.items()},
                        "commanded_deg": {k: round(v, 1) for k, v in commanded.items()},
                    }
                )
                return 3
            if t_end is not None and t0 >= t_end:
                break
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
        err = {k: round(present[k] - target[k], 1) for k in target}
        reached = all(abs(e) <= 3.0 for e in err.values())
        _emit(
            {
                "event": "done",
                "reached": reached,
                "deg": {k: round(present[k], 1) for k in target},
                "error_deg": err,
            }
        )
        return 0 if reached else 2
    finally:
        link.close()


if __name__ == "__main__":
    sys.exit(main())
