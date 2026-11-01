# Optional GPU LiDAR layer over the existing Gazebo Classic/Humble runtime.
ARG BASE_IMAGE=usim-gazebo:local
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-velodyne-gazebo-plugins \
    && rm -rf /var/lib/apt/lists/*

COPY packages/gazebo/laser_2nd_pass.vert \
    /usr/share/gazebo-11/media/materials/programs/laser_2nd_pass.vert
