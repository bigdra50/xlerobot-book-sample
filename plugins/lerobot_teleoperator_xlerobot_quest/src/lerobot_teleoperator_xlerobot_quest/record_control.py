"""lerobot-record の区切りと lerobot-rollout の dagger の切り替えを、Quest のボタンから出す。

lerobot v0.6.1 の lerobot-record と dagger は、操作をキーボードからしか受けない（Teleoperator から渡す口が無い）。
そこで両者が呼ぶ create_key_listener を差し替え、キーの処理をボタンからも呼ぶ。
lerobot-teleoperate は create_key_listener を呼ばないので、この差し替えは何もしない。
"""

from __future__ import annotations

import logging
import re
import sys
import time
from collections.abc import Callable

from lerobot.utils import keyboard_input

logger = logging.getLogger(__name__)

# lerobot-record が区間（エピソードかリセット）の始まりに log_say で出す文
_EPISODE_PREFIX = "Recording episode"
_RESET_MESSAGE = "Reset the environment"
# 区間が始まってからこの秒数は、区切りを受けない
MIN_SECTION_S = 1.0

# dagger は phase を切り替えるとき、自分の logger へこの形で出す。
# logger のフィルターは子 logger から伝わった record には掛からないので、root ではなくこの logger に付ける
_DAGGER_MODULE = "lerobot.rollout.strategies.dagger"
_PHASE_RE = re.compile(r"Phase transition: (\w+) -> (\w+)")
# 補正だけを記録するモードの dagger が、ループに入る直前に出す
_DAGGER_START_PREFIX = "DAgger corrections-only recording started"
# 押したか phase が変わった直後のこの秒数は、ボタンを受けない（押し続けや、取りこぼしの押し直しで 2 段進めない）
DAGGER_PRESS_GAP_S = 0.5


class RecordControl:
    def __init__(
        self, on_reset: Callable[[], None], on_correction_start: Callable[[], None] = lambda: None
    ) -> None:
        self._on_reset = on_reset
        self._on_correction_start = on_correction_start
        self._dispatch: Callable[[str], None] | None = None
        self._original: Callable | None = None
        # 今の区間が始まった時刻。区切りを受けたあと、次の区間が始まるまでは None
        self._section_t: float | None = None
        # Quest の画面に出す状態。"episode" か "reset"。lerobot-record の外では None
        self._section: str | None = None
        self._section_start = 0.0
        self._episode = 0
        # リセット中に B（→）を受けたら True。lerobot-record がエピソードを始めてよいのはこのときだけ
        self._start_requested = False
        # dagger の操作。lerobot-rollout --strategy.type=dagger のときだけ dispatch が入る
        self._dagger_dispatch: Callable[[str], None] | None = None
        self._dagger_keys = ("space", "tab")  # pause_resume、correction の順
        self._dagger_phase = "autonomous"
        self._dagger_t = float("-inf")
        self._dagger_phase_t = 0.0
        # B で交代を頼んだあと、止まったら補正を始める。dagger は phase を 1 回に 1 つしか進めない
        self._dagger_chain = False
        self._dagger_module = None
        self._dagger_original: Callable | None = None

    def install(self) -> None:
        self._original = keyboard_input.create_key_listener

        def create_key_listener(dispatch: Callable[[str], None], **_kwargs):
            self._dispatch = dispatch
            # 公式は pynput でほかのアプリのキーも拾う（チャットで打った r で撮り直しになる）。
            # キーは lerobot-record を動かす端末からだけ受ける
            listener = None
            if sys.stdin.isatty():
                listener = keyboard_input.TerminalKeyListener(self.press)
                listener.start()
            logger.info("Quest の B か、端末で → を押すと記録を始める")
            return listener

        keyboard_input.create_key_listener = create_key_listener
        logging.getLogger().addFilter(self._watch_sections)
        # dagger は create_key_listener を import で束縛しているので、その名前も差し替える。
        # lerobot-rollout は teleop を connect する前に dagger を import している。ほかのコマンドでは import しない
        module = sys.modules.get(_DAGGER_MODULE)
        if module is not None:
            self._dagger_module = module
            self._dagger_original = module.create_key_listener
            module.create_key_listener = self._create_dagger_listener
        logging.getLogger(_DAGGER_MODULE).addFilter(self._watch_dagger)

    def uninstall(self) -> None:
        if self._original is not None:
            keyboard_input.create_key_listener = self._original
            self._original = None
        logging.getLogger().removeFilter(self._watch_sections)
        if self._dagger_module is not None:
            self._dagger_module.create_key_listener = self._dagger_original
            self._dagger_module = self._dagger_original = None
        logging.getLogger(_DAGGER_MODULE).removeFilter(self._watch_dagger)
        self._dagger_dispatch = None

    def _create_dagger_listener(self, dispatch: Callable[[str], None], controls_help: str = "", **_kwargs):
        self._dagger_dispatch = dispatch
        keys = dict(re.findall(r"(\w+)='([^']+)'", controls_help))
        self._dagger_keys = (keys.get("pause_resume", "space"), keys.get("correction", "tab"))
        # 公式の pynput はほかのアプリのキーも拾う。dagger では、チャットで打った空白でポリシーが止まる。
        # キーは lerobot-rollout を動かす端末からだけ受ける
        listener = None
        if sys.stdin.isatty():
            listener = keyboard_input.TerminalKeyListener(dispatch)
            listener.start()
        logger.info("Quest の B で交代（補正の開始と終了）、Y でポリシーの一時停止と再開")
        return listener

    def press(self, key: str) -> None:
        """キーとボタンの両方から呼ぶ。key は lerobot のキー名（right、left、esc、n、r、q）。"""
        if self._dagger_dispatch is not None:
            self._press_dagger(key)
            return
        if self._dispatch is None:
            return
        if key.lower() not in ("esc", "q"):
            # lerobot-record は、区間の始まりで区切りの旗を見る。エピソードの保存中に押し直すと、
            # 次のエピソードが 0 フレームで終わり、保存で落ちる。区切りは区間ごとに 1 回だけ受ける
            if self._section_t is None or time.monotonic() - self._section_t < MIN_SECTION_S:
                logger.info("区切りを受けた直後なので %s を無視した", key)
                return
            self._section_t = None
        logger.info("区切りの %s を受けた", key)
        if key.lower() in ("right", "n") and self._section == "reset":
            self._start_requested = True
        self._dispatch(key)

    def rest_allowed(self) -> bool:
        """X で rest へ戻してよいか。dagger では補正のあいだだけ（ほかの phase では get_action が呼ばれない）。"""
        return self._dagger_dispatch is None or self._dagger_phase == "correcting"

    def _press_dagger(self, key: str) -> None:
        """B（right）で交代、Y（left）で一時停止と再開。"""
        assert self._dagger_dispatch is not None
        now = time.monotonic()
        if now - self._dagger_t < DAGGER_PRESS_GAP_S:
            logger.info("切り替えた直後なので %s を無視した", key)
            return
        pause, correction = self._dagger_keys
        phase = self._dagger_phase
        if key == "right":
            self._dagger_t = now
            if phase == "autonomous":
                self._dagger_chain = True
                self._dagger_dispatch(pause)
            else:
                self._dagger_dispatch(correction)
        elif key == "left" and phase != "correcting":
            # 補正のあいだは止めない。補正を終えるのは B だけにして、記録の切れ目を 1 つの操作にそろえる
            self._dagger_t = now
            self._dagger_chain = False
            self._dagger_dispatch(pause)

    def _watch_dagger(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if message.startswith(_DAGGER_START_PREFIX) and self._dagger_dispatch is not None:
            # dagger はポリシーを動かした状態で始まるが、この時点では Quest をまだ被っていない。
            # 止まった状態で始め、Y で動かし始める。ループは最初に遷移を処理するので、ポリシーは 1 回も指令を出さない
            self._dagger_phase_t = time.monotonic()
            self._dagger_dispatch(self._dagger_keys[0])
            return True
        match = _PHASE_RE.fullmatch(message)
        if match is None:
            return True
        old, new = match.groups()
        self._dagger_phase = new
        self._dagger_t = self._dagger_phase_t = time.monotonic()
        if new == "correcting":
            self._on_correction_start()
        if (
            old == "autonomous"
            and new == "paused"
            and self._dagger_chain
            and self._dagger_dispatch is not None
        ):
            # このログは dagger が遷移を処理している最中に出る。ここで頼んだ遷移は次の周で処理される
            self._dagger_dispatch(self._dagger_keys[1])
        self._dagger_chain = False
        return True

    def status(self, now: float) -> tuple[str, str] | None:
        """Quest の画面に出す 2 行。lerobot-record と dagger の外では None。"""
        if self._dagger_dispatch is not None:
            elapsed = int(now - self._dagger_phase_t)
            clock = f"{elapsed // 60}:{elapsed % 60:02d}"
            if self._dagger_phase == "correcting":
                return "CORRECT", f"A: follow  B: finish  X: rest  {clock}"
            if self._dagger_phase == "paused":
                return "PAUSED", "B: correct  Y: resume"
            return "AUTO", "B: take over  Y: pause"
        elapsed = int(now - self._section_start)
        if self._section == "episode":
            return "REC", f"episode {self._episode}  {elapsed // 60}:{elapsed % 60:02d}"
        if self._section == "reset":
            return "RESET", "B: start recording"
        return None

    def _watch_sections(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if message.startswith(_EPISODE_PREFIX) and self._dispatch is not None:
            if not self._start_requested:
                # 起動の直後と、リセットの時間切れのあとは、lerobot-record が B を待たずにエピソードを始める。
                # 撮り直しの扱いにしてリセットへ戻す。エピソードは 1 フレームも録らずに捨てられる
                logger.info("B を待たずに始まったエピソードを、リセットへ戻した")
                self._dispatch("left")
                return True
            self._start_requested = False
            self._section = "episode"
            self._episode = int(message.rsplit(" ", 1)[-1])
            self._section_t = self._section_start = time.monotonic()
        if message == _RESET_MESSAGE:
            self._section = "reset"
            self._section_t = self._section_start = time.monotonic()
            self._on_reset()
        return True
