"""トリガーの押し込み量から gripper の指令へ。

出力は公式 XLerobot と同じ 0..100 の %。
どちらの端が開きかはキャリブレーションで決まるので、値は open_value と closed_value で受け取る。
"""

from __future__ import annotations


def trigger_to_gripper(trigger: float, *, open_value: float, closed_value: float) -> float:
    """押し込み量（0..1）に比例して、離すと closed_value、握り切ると open_value にする。

    物をつかんでいるあいだトリガーを握り続けなくて済むよう、公式のサンプル（握ると閉じる）とは逆の向きにしている。
    """
    t = min(1.0, max(0.0, trigger))
    return closed_value + (open_value - closed_value) * t
