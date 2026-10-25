# xlerobot-book-sample

『XLeRobotで始める模倣学習』のサンプルコード

## Contents

| 章                   | パス                                                                                                         |
| -------------------- | ------------------------------------------------------------------------------------------------------------ |
| XLeRobotとは         | `install.sh`、`scripts/verify_stack.py`、`plugins/lerobot_robot_xlerobot_book/`                              |
| テレオペのためのIK   | `scripts/closed_form_ik.py`、`urdf/`                                                                         |
| VRテレオペ           | `plugins/lerobot_teleoperator_xlerobot_quest/`、`host/`、`scripts/check_bus.py`、`scripts/check_zmq_host.py` |
| データセットの記録   | `scripts/check_cameras.py`、`scripts/check_dataset.py`                                                       |
| ポリシーの学習と評価 | `install.ps1`、`scripts/freeze_state.py`                                                                     |
| 付録                 | [`navigate/`](navigate/README.md)                                                                            |

## Requirements

- Python 3.12
- uv
- git

## Installation

```bash
./install.sh
source ~/xlerobot-book-env/venv/bin/activate
python scripts/verify_stack.py
```

Windows の学習環境は `install.ps1` で作る

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\install.ps1
```

## License

Copyright 2026 Ryudai Kimura

Apache-2.0（[LICENSE](LICENSE)）。
同梱した第三者のファイルは [NOTICE](NOTICE) を参照。
