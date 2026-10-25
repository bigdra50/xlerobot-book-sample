"""追従の開始・一時停止・停止を決める状態機械。

コントローラの姿勢は teleop_xr が IK に渡すが、「今追従してよいか」はこの plugin が決める。
teleop_xr 既定のデッドマン（両 grip 押下中だけ追従）は長時間の操作で手が疲れるので、
右 A のトグルで追従を切り替え、grip は一時停止に割り当てる。
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import EngagePhase, VRSnapshot


@dataclass(frozen=True)
class EngageEvent:
    phase: EngagePhase
    # このティックで基準（VR の姿勢と手先のスナップショット）を取り直す
    rebase: bool

    @property
    def engaged(self) -> bool:
        return self.phase is EngagePhase.ENGAGED


class EngageStateMachine:
    def __init__(self) -> None:
        self.reset()

    @property
    def phase(self) -> EngagePhase:
        return self._phase

    def reset(self) -> None:
        self._phase = EngagePhase.IDLE
        # 初回ティックで A が押されていれば立ち上がりとみなす。切断→再接続で押しっぱなしになる状況は考えない
        self._prev_primary = False

    def update(self, snap: VRSnapshot) -> EngageEvent:
        ev = self._update_toggle(snap)
        self._prev_primary = snap.right.primary
        return ev

    def _update_toggle(self, snap: VRSnapshot) -> EngageEvent:
        both_tracked = snap.left.has_pose and snap.right.has_pose
        rising = snap.right.primary and not self._prev_primary
        prev = self._phase

        # 追従中にトラッキングを失ったら止める。復帰しても勝手に再開せず、A を押し直させる
        if prev is not EngagePhase.IDLE and not both_tracked:
            self._phase = EngagePhase.IDLE
            return EngageEvent(EngagePhase.IDLE, rebase=False)

        if rising:
            if prev is EngagePhase.IDLE:
                # 基準のスナップショットに両手の姿勢が要るので、トラッキング前の A は無視する
                if not both_tracked:
                    return EngageEvent(EngagePhase.IDLE, rebase=False)
                self._phase = EngagePhase.ENGAGED
                return EngageEvent(EngagePhase.ENGAGED, rebase=True)
            self._phase = EngagePhase.IDLE
            return EngageEvent(EngagePhase.IDLE, rebase=False)

        if prev is EngagePhase.IDLE:
            return EngageEvent(EngagePhase.IDLE, rebase=False)

        any_grip = snap.left.squeeze or snap.right.squeeze
        if prev is EngagePhase.ENGAGED:
            if any_grip:
                self._phase = EngagePhase.PAUSED
                return EngageEvent(EngagePhase.PAUSED, rebase=False)
            return EngageEvent(EngagePhase.ENGAGED, rebase=False)

        # PAUSED。両方離れるまで待ち、離れたら手を動かした分を無視するために基準を取り直す
        if any_grip:
            return EngageEvent(EngagePhase.PAUSED, rebase=False)
        self._phase = EngagePhase.ENGAGED
        return EngageEvent(EngagePhase.ENGAGED, rebase=True)
