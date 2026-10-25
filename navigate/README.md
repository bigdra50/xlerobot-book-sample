# navigate

付録の自律走行のコード
iPad の pocketsensor のアプリをセンサーにして、ROS 2 の Nav2 と RTAB-Map で走らせる

| パス                 | 中身                                                               |
| -------------------- | ------------------------------------------------------------------ |
| `gate/`              | Raspberry Pi で動かす車輪のサーバーと、ARKit の姿勢の記録と解析    |
| `ros2/xlerobot_nav/` | ROS 2 のパッケージ（オドメトリの融合、Nav2 と RTAB-Map の launch） |
| `ros2/docker/`       | ROS 2 Jazzy のコンテナと、起動、巡回、表示のスクリプト             |

## Requirements

- Docker
- iPad に入れた pocketsensor のアプリ。[pocket-sensor](https://github.com/bigdra50/pocket-sensor) の README の手順で Xcode からビルドする
- このリポジトリと同じ階層に clone した pocket-sensor のリポジトリ

```bash
git clone https://github.com/bigdra50/pocket-sensor.git ../pocketsensor
```

iPad の取り付けの位置と向きは、自分の機体で測って `ros2/xlerobot_nav/urdf/xlerobot.urdf.xacro` に書く

## Usage

コマンドはリポジトリのルートで実行する。

1. Raspberry Pi で `host/xlerobot_host.py --base.listen=0.0.0.0:8766` を起動する
2. iPad で pocketsensor のアプリを起動する
3. コンテナを作り、ROS 2 のスタックを起動する

   ```bash
   navigate/ros2/docker/up.sh
   docker exec -d xlerobot-nav bash -lc 'source /ws/env.sh && ros2 launch xlerobot_nav map.launch.py source:=ws://<iPad の IP>:8765 base_host:=<Pi の IP> new:=true > /out/launch.log 2>&1'
   ```

   地図を作るときは `new:=true`、作った地図の上で走らせるときは `localize:=true` にする。

4. `navigate/ros2/docker/view.sh live` で Lichtblick を開く

巡回は `navigate/ros2/docker/patrol.sh`、登録した場所への移動は `ros2 run xlerobot_nav goto_place`、机の手前での停止は `ros2 run xlerobot_nav table_dock` で行う。
引数は各スクリプトの先頭のコメントにある。
