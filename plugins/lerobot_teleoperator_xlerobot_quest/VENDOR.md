# 同梱した第三者のコード

## 閉形式 IK（lekiwi-vr-teleop）

NektoDron/lekiwi-vr-teleop（Apache-2.0、Copyright 2026 Andrey Demidov）の
`lekiwi_vr_teleop/so101_chain.py` を同梱している。
コミットは `4e1350cbe7ef4b61a53ade160784c4bcb81b2dfa`（2026-08-04）。
license 本文は `LICENSE-lekiwi`。

SO-101 の寸法を定数にした閉形式で、実行時に URDF を読まない。
`--teleop.ik_backend=closed_form` のとき、`ik/analytic.py` の `ClosedFormIK` がこの関数を呼ぶ。
肘は上げる枝（上流 `inverse` の `elbow_up=True`）で解く。

| 同梱ファイル | 出典 | 変更点 |
| --- | --- | --- |
| `ik/vendor/so101_chain.py` | `lekiwi_vr_teleop/so101_chain.py` | なし（先頭に出典のコメントを追加しただけ） |

上流の arm 制御（clutch、速度制限、gripper）は同梱していない。
engage、gripper、head、台車は本書の Teleoperator のまま。

### 閉形式を同梱し直すとき

1. 上流の `so101_chain.py` を取り直す
2. 先頭の出典コメント以外は変えない
3. `pytest plugins/lerobot_teleoperator_xlerobot_quest/tests/test_ik_backends.py` が通ることを見る
4. この表とコミットを更新する
