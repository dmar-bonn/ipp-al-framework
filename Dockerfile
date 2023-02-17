FROM ubuntu:focal

ARG DEBIAN_FRONTEND=noninteractive

# Install basic packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils nano git curl python3-pip dirmngr gnupg2 && \
    rm -rf /var/lib/apt/lists/*

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# Install ROS noetic
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

ENV ROS_DISTRO noetic
RUN apt-get update && \
    apt-get install -y --no-install-recommends ros-noetic-desktop-full && \
    rm -rf /var/lib/apt/lists/*
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN rosdep init
RUN rosdep update --rosdistro $ROS_DISTRO

# Install Flightmare dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake libzmqpp-dev libopencv-dev && \
    apt-get install -y --no-install-recommends libgoogle-glog-dev protobuf-compiler ros-$ROS_DISTRO-octomap-msgs ros-$ROS_DISTRO-octomap-ros ros-$ROS_DISTRO-joy python3-vcstool python3-catkin-tools && \
    rm -rf /var/lib/apt/lists/*

RUN echo "export FLIGHTMARE_PATH=/ipp-al-framework/flightmare_ws/src/flightmare" >> ~/.bashrc

COPY flightmare_ws/ /flightmare_ws/
RUN cd /flightmare_ws/ && catkin config --init --mkdirs --extend /opt/ros/$ROS_DISTRO --merge-devel --cmake-args -DCMAKE_BUILD_TYPE=Release
RUN cd /flightmare_ws/src/ && vcs-import < flightmare/flightros/dependencies.yaml

RUN cd flightmare_ws/ && catkin build

#  Install Python dependencies
COPY requirements.txt al_requirements.txt
COPY bayesian_erfnet/requirements.txt dl_requirements.txt

RUN pip3 install -r dl_requirements.txt && \
    pip3 install -r al_requirements.txt && \
    rm al_requirements.txt && \
    rm dl_requirements.txt && \
    rm -r ~/.cache/pip
