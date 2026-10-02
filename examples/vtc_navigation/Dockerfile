# Optional ROS 2 Humble overlay for Raspberry Pi Cat SLAM and Nav2.
ARG USIM_GAZEBO_IMAGE=usim-gazebo:local
FROM ${USIM_GAZEBO_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        python3-colcon-common-extensions \
        python3-rosdep \
        python3-vcstool \
        ros-humble-nav2-bringup \
    && rm -rf /var/lib/apt/lists/*

COPY examples/vtc_navigation/raspicat.repos /tmp/raspicat.repos
RUN mkdir -p /opt/raspicat_ws/src \
    && vcs import /opt/raspicat_ws/src < /tmp/raspicat.repos \
    && rm /tmp/raspicat.repos

RUN rosdep update \
    && apt-get update \
    && rosdep install -r -y --from-paths /opt/raspicat_ws/src --ignore-src --rosdistro humble \
    && rm -rf /var/lib/apt/lists/*

RUN cd /opt/raspicat_ws \
    && source /opt/ros/humble/setup.bash \
    && colcon build --symlink-install --merge-install \
    && chmod -R a+rX /opt/raspicat_ws

COPY examples/vtc_navigation /opt/vtc_navigation
WORKDIR /workspace

CMD ["/bin/bash", "/opt/vtc_navigation/run.sh"]
