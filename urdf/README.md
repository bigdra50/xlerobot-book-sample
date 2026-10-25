# XLeRobot の URDF（腕は SO-101）

`xlerobot_so101.urdf` は、SO-101 の腕 2 本を XLeRobot の車体と head に取り付けた URDF。
VR テレオペの IK（`lerobot_teleoperator_xlerobot_quest`）と、本の図がこのファイルを読む。

XLeRobot 公式に同梱された URDF（`simulation/Maniskill/assets/xlerobot/xlerobot.urdf`）は、腕が SO-100 のもので、関節名と関節角 0 の姿勢が SO-101 の実機と合わない。
そこで腕だけを SO-ARM100 公式の SO-101 の URDF に差し替えた。

## 中身

| 部分 | 出典 | 変更点 |
| --- | --- | --- |
| 腕 2 本（`assets/` の STL 13 個） | SO-ARM100 の `Simulation/SO101/so101_new_calib.urdf` と `Simulation/SO101/assets/` | 左右に複製し、link と joint の名前に `left_`、`right_` を付けた |
| 腕の取り付け位置、首の位置 | XLeRobot 公式の `xlerobot.urdf` の寸法 | 腕は回さずに置いた（公式の yaw 90 度は SO-100 の腕の座標系に合わせた値） |
| 車体、天板、head（`xlerobot_meshes/` の STL 9 個） | XLeRobot 公式の `simulation/Maniskill/assets/xlerobot/meshes/` | 見た目のためだけに追加した。当たり判定は無い |
| ベースの仮想関節 `base_x`、`base_y`、`base_theta` | 本リポジトリで追加した | 台車の位置と向きを関節として表す |

関節名は次のとおり。
head は LeRobot の action のキーから `.pos` を除いた名前と同じ。
腕は、そこからさらに `left_arm_` / `right_arm_` を `left_` / `right_` に縮めた名前。

- 腕: `left_shoulder_pan` 〜 `left_gripper`、`right_shoulder_pan` 〜 `right_gripper`
- head: `head_motor_1`（pan）、`head_motor_2`（tilt）

関節角をすべて 0 にした姿勢は、LeRobot のキャリブレーションで中央として置く姿勢と同じ。

## 出典の版とライセンス

| リポジトリ | コミット | ライセンス |
| --- | --- | --- |
| [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) | `a758567c3978dfeefe282ede0500085a48fe8f78`（`so101_new_calib.urdf` の最後の変更は `385e8d7`、2025-07-02） | Apache-2.0、本文は `LICENSE-SO-ARM100` |
| [Vector-Wangel/XLeRobot](https://github.com/Vector-Wangel/XLeRobot) | `3d14695e40c9c68229c0aacffca6053c75cd3eb6`（2026-07-22） | Apache-2.0、本文は `LICENSE-XLeRobot` |

STL は無改変で複製した。
URDF は上の表のとおり組み替えたもので、元のファイルそのものではない。

## メッシュのパス

URDF の中のメッシュは、このディレクトリからの相対パス（`assets/…`、`xlerobot_meshes/…`）で書いてある。
読むツールがキャッシュなど別の場所へ URDF を書き出す場合は、先に絶対パスへ書き換える。
