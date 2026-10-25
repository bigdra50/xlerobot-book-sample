# xlerobot_nav を動かす ROS 2 のイメージ。ベースに Nav2 と、深度から疑似 LaserScan を作るパッケージを足す。
#   docker build -t xlerobot-nav:jazzy --build-arg BASE=<ROS 2 Jazzy のイメージ> navigate/ros2/docker
ARG BASE=ros:jazzy-ros-base
FROM ${BASE}
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3-pip python3-colcon-common-extensions \
      ros-jazzy-navigation2 ros-jazzy-nav2-bringup \
      ros-jazzy-depth-image-proc ros-jazzy-pointcloud-to-laserscan \
      ros-jazzy-compressed-depth-image-transport ros-jazzy-foxglove-bridge ros-jazzy-xacro \
 && rm -rf /var/lib/apt/lists/*
# 地図は rtabmap_ros で、iPad の RGB-D と融合した odom から作る。上の層をキャッシュに残すため、別の層にする
RUN apt-get update \
 && apt-get install -y --no-install-recommends ros-jazzy-rtabmap-ros \
 && rm -rf /var/lib/apt/lists/*
# host の観測（ZMQ）を ROS 2 の話題へ写す host_bridge が使う
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3-zmq \
 && rm -rf /var/lib/apt/lists/*
