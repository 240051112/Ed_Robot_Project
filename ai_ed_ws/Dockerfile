# Use the official NVIDIA base image for Jetson with JetPack 6, which includes CUDA
FROM nvcr.io/nvidia/l4t-base:r36.3.0

# Set up the environment to be non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash

# Set up locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8

# Install ROS 2 Humble
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && rm -rf /var/lib/apt/lists/*
RUN add-apt-repository universe
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-vision-msgs \
    ros-humble-camera-info-manager \
    python3-colcon-common-extensions \
    python3-venv \
    git \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up the Python Virtual Environment
RUN python3 -m venv /opt/ros_venv
ENV PATH="/opt/ros_venv/bin:$PATH"

# Install all Python dependencies correctly
RUN . /opt/ros_venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir \
    empy==3.3.4 \
    catkin-pkg \
    lark \
    cmake \
    ultralytics

# Add the special NVIDIA PyTorch wheel
# Download it first on your host, then it will be copied into the container
ADD https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl /tmp/torch.whl
RUN . /opt/ros_venv/bin/activate && \
    pip install --no-cache-dir --force-reinstall /tmp/torch.whl && \
    rm /tmp/torch.whl

# Build and install Torchvision from source
RUN git clone -b v0.18.0 https://github.com/pytorch/vision.git /tmp/vision
RUN . /opt/ros_venv/bin/activate && \
    cd /tmp/vision && \
    python3 setup.py install && \
    cd / && \
    rm -rf /tmp/vision

# Set up the ROS 2 workspace
WORKDIR /root/ros2_ws
COPY ./src ./src

# Source the ROS setup file and build the workspace
RUN . /opt/ros/humble/setup.bash && \
    . /opt/ros_venv/bin/activate && \
    colcon build

# This is the command that will run when the container starts
CMD ["/bin/bash", "-c", "source /root/ros2_ws/install/setup.bash && /bin/bash"]