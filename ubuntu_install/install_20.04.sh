#!/bin/bash

# Stop execution if any command fails
set -e

sudo apt install build-essential libssl-dev dkms
sudo apt install gcc
sudo apt install python3-pip
sudo apt install git

# update cmake
cd ~/Downloads
sudo apt remove --purge cmake
git clone -b v3.30.2 https://github.com/Kitware/CMake.git
cd CMake
./bootstrap && make && sudo make install
echo 'export PATH=/usr/local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig
#mrt@gannet:~$ cmake --version
#cmake version 3.30.20240918-gb69b5a9

# Check Python version
python3 --version
pip3 --version

# Nvidia drivers
# https://ubuntu.com/server/docs/nvidia-drivers-installation
# make sure no other nvidia drivers installed. If any found, remove them
# dpkg -l | grep -i nvidia
# sudo apt-get remove --purge '^nvidia-.*'
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-driver-470
sudo apt-get update
sudo apt-get upgrade
sudo reboot

# Install Cuda-11-4 (uncheck driver box)
cd ~/Downloads
wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.0-470.42.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.0-470.42.01-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
sudo apt-get update
#sudo apt-get install cuda
sudo apt-get install cuda-11.4

echo 'export CUDA_HOME=/usr/local/cuda-11.4' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig

# Check installation
nvidia-smi
nvcc --version

# Check your CUDA Capability Major/Minor version number: (its 7.5 for the GPU in Zbook create G7)
cd /usr/local/cuda-11.4/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery

# Install cuDNN (wget may fail to get all the file due to nvidia account log in, download it manually)
# from https://developer.nvidia.com/rdp/cudnn-archive
# 	Download cuDNN v8.2.4 (September 2nd, 2021), for CUDA 11.4
#		cuDNN Runtime Library for Ubuntu20.04 x86_64 (Deb)
sudo dpkg -i libcudnn8_8.2.4.15-1+cuda11.4_amd64.deb
sudo apt update
sudo apt install libcudnn8
#		cuDNN Developer Library for Ubuntu20.04 x86_64 (Deb)
sudo dpkg -i libcudnn8-dev_8.2.4.15-1+cuda11.4_amd64.deb
sudo apt update
sudo apt install libcudnn8-dev

#https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/11.x/cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz/
#tar -xJvf cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz
#sudo cp cudnn-linux-x86_64-8.9.7.29_cuda11-archive/include/cudnn*.h /usr/local/cuda-11.4/include
#sudo cp cudnn-linux-x86_64-8.9.7.29_cuda11-archive/lib/libcudnn* /usr/local/cuda-11.4/lib64
#sudo chmod a+r /usr/local/cuda-11.4/include/cudnn*.h /usr/local/cuda-11.4/lib64/libcudnn*
# verify cudnn
#ls /usr/local/cuda-11.4/include | grep cudnn
#ls /usr/local/cuda-11.4/lib64 | grep cudnn

# Configuration (cuda 11.4, cuDNN 8.2, tensorrt 8.6, deepstream 6.0)

# tensorrt-8
cd ~/Downloads
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/local_repos/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-11.8_1.0-1_amd64.deb
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-11.8_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-11.8/nv-tensorrt-local-0628887B-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install tensorrt
pip3 install nvidia-pyindex
pip3 install nvidia-tensorrt
echo 'export PATH=/usr/src/tensorrt/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/src/tensorrt/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# deepstream-6.0
sudo apt update
sudo apt install \
libssl1.1 \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
libgstrtspserver-1.0-0 \
libjansson4
cd ~/Downloads
git clone https://github.com/edenhill/librdkafka.git
cd librdkafka
git reset --hard 7101c2310341ab3f4675fc565f64f0967e135a6a
#if python is not linked to python3, then link them
#sudo ln -s /usr/bin/python3 /usr/bin/python
./configure
#in Makefile replace python with python3
make
sudo make install
sudo mkdir -p /opt/nvidia/deepstream/deepstream-6.0/lib
sudo cp /usr/local/lib/librdkafka* /opt/nvidia/deepstream/deepstream-6.0/lib
#download the deepstream SDK
wget https://developer.nvidia.com/deepstream-6.0_6.0.0-1_amd64deb
sudo apt install ./deepstream-6.0_6.0.0-1_amd64.deb

#check deepstream is installed
dpkg -l | grep deepstream
deepstream-app --version-all

# tensorrt-10
#cd ~/Downloads
#wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/local_repo/nv-tensorrt-local-repo-ubuntu2004-10.0.1-cuda-11.8_1.0-1_amd64.deb
#sudo dpkg -i nv-tensorrt-local-repo-ubuntu2004-10.0.1-cuda-11.8_1.0-1_amd64.deb
#sudo cp /var/nv-tensorrt-local-repo-ubuntu2004-10.0.1-cuda-11.8/nv-tensorrt-local-4BE0C9B6-keyring.gpg /usr/share/keyrings/
#sudo apt update
#sudo apt install tensorrt
#sudo apt install python3-libnvinfer-dev
#echo 'export PATH=/usr/src/tensorrt/bin:$PATH' >> ~/.bashrc
#echo 'export LD_LIBRARY_PATH=/usr/src/tensorrt/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
#source ~/.bashrc

# Install basic dependencies
sudo apt-get update
sudo apt install libhdf5-dev
sudo apt install cmake git git-lfs gedit
#sudo apt install ffmpeg libavcodec-dev libavformat-dev libavdevice-dev libavfilter-dev libavresample-dev libswscale-dev libpostproc-dev

# make directories
mkdir ~/src
mkdir ~/dev

cd ~/src
git clone --recursive -b v1.12.0 https://github.com/pytorch/pytorch  # use v1.12.0 with cuda 11.4
git clone -b v0.13.0 https://github.com/pytorch/vision  # use v0.13.0 with cuda 11.4

# Check Python paths to confirm that sudo will not be needed
python3 -m site
#sys.path should contain packages and later should include pytorch
#USER_BASE should be ~/.local
#USER_SITE should be ~/.local/lib/python3.8/site-packages

# Install PyTorch compatible with your CUDA (Jetson doesnt support mkl)
# follow DETR steps, this is only a reinstall
# for Jetson, run the line without mkl and mkl-include
# If you're installing it system-wide and not using a virtual environment, you might need to use sudo:
python3 -m pip install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests
#sudo python3 -m pip install astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests
python3 -m pip install testresources

cd ~/src/pytorch
# other inputs sometimes needed:
# 	CUDA_HOME=/usr/local/cuda-11.4 
# 	CUDACXX=/usr/local/cuda-11.4/bin/nvcc
export BUILD_VERSION=1.12.0
export TORCH_CUDA_ARCH_LIST="7.5"
python3 setup.py bdist_wheel
cd dist
python3 -m pip install torch-1.12.0*.whl

#check
cd ~/src
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); \
print(f'Compute: {torch.cuda.get_arch_list()}'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
print(f'CUDA version: {torch.version.cuda}'); \
print(f'Device name: {torch.cuda.get_device_name()}')"

cd ~/src/vision
# other inputs sometimes needed:
# 	CUDA_HOME=/usr/local/cuda-11.4 
# 	CUDACXX=/usr/local/cuda-11.4/bin/nvcc
# if numpy version is causing an issue, then force the use of pip to install dependencies correctly.
#	sudo python3 setup.py clean 
export BUILD_VERSION=0.13.0
export TORCH_CUDA_ARCH_LIST="7.5"
python3 setup.py bdist_wheel
cd dist
pip install torchvision-0.13.0*.whl

#check
cd ~/src
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); \
import torchvision; print(f'torchvision version: {torchvision.__version__}'); \
print(f'Compute: {torch.cuda.get_arch_list()}'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
print(f'CUDA version: {torch.version.cuda}'); \
print(f'Device name: {torch.cuda.get_device_name()}')"

# pycuda
python3 -m pip install pycuda
python3 -c "import pycuda; print(pycuda.VERSION)"
#pip install --upgrade toml
#sudo pip install pycuda==2022.2.2  # for numpy 1.24.4
# if not finding cuda.h, then explicitly link cuda when installing pycuda
#sudo pip install pycuda==2022.2.2 --global-option="-I$CUDA_HOME/include" --global-option="-L$CUDA_HOME/lib64"
#cd ~/Downloads
#wget https://files.pythonhosted.org/packages/61/69/f53a6624def08348778a7407683f44c2a9adfdb0b68b9a45f8213ff66c9d/pycuda-2024.1.2.tar.gz
#tar xzvf pycuda-2024.1.2.tar.gz
#cd pycuda-2024.1.2
#./configure.py --cuda-root=/usr/local/cuda-11.4
#make -j4
#sudo python3 setup.py install
#sudo pip install .

# bytetrack
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
#pip3 install --upgrade pillow
#pip3 install --force-reinstall imageio scikit-image
pip3 install -r requirements.txt
python3 setup.py develop --user
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox
# test
# download bytetrack_x_mot17.pth.tar in ByteTrack/pretrained
cd ByteTrack
python3 tools/demo_track.py video -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result

# opencv
sudo apt install libopencv-dev #python3-opencv
python3 -m pip install opencv-python
#python3 -m pip install opencv-python-headless  # Optional, for headless environments

cd ~/src
git clone https://gitlab.com/missionsystems/hyperteaming/detr.git
git clone https://gitlab.com/missionsystems/hyperteaming/csv-to-euroc.git
git clone https://gitlab.com/missionsystems/hyperteaming/window-tracker.git

cd ~/src/detr
pip install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install git+https://github.com/cocodataset/panopticapi.git

# Eigen
# sudo apt-get install libeigen3-dev   # 3.3.0
git clone https://gitlab.com/libeigen/eigen.git   # 3.4.0

# Clone COLMAP from the official repository

# git clone -b 2.1.0 https://github.com/ceres-solver/ceres-solver.git
# cd ceres-solver
# git checkout 2.1.0
# git reset HEAD --hard
# mkdir build
# cd build
# cmake .. && make -j16
# sudo make install

cd ~/src
sudo apt-get update

sudo apt-get install -y git cmake build-essential \
    libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev \
    libboost-regex-dev libboost-system-dev libboost-test-dev \
    libsuitesparse-dev libfreeimage-dev libgoogle-glog-dev libgflags-dev libglew-dev \
    qtbase5-dev libqt5opengl5-dev libcgal-dev libsqlite3-dev\
    libatlas-base-dev libsuitesparse-dev libceres-dev libmetis-dev libhdf5-dev libflann-dev

git clone -b 3.8 https://github.com/colmap/colmap.git
cd colmap
git checkout 3.8
git reset HEAD --hard
echo "installing colmap ${PWD}"
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
# gedit /home/mrt/src/colmap/src/base/image.cc
#  35 | #include "base/projection.h"
# +++ |+#include <cassert>
#  36 | 
cmake -DCMAKE_CUDA_ARCHITECTURES="75" ..
make -j$(nproc)  # Compile using all available cores
sudo make install
# sudo apt install -y colmap << THis is not a dev version, no cmakelists or headers.

# Clone and install PoseLib
cd ~/src
git clone --recursive https://github.com/vlarsson/PoseLib.git
cd PoseLib
echo "installing PoseLib ${PWD}"
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
cmake ..
make -j$(nproc)
sudo make install

# COMMA and SNARK
cd ~/src
git clone https://gitlab.com/orthographic/comma.git
git clone https://gitlab.com/orthographic/snark.git
git clone https://gitlab.com/missionsystems/ms-common.git   #for ms-log-multitool (using ccmake, enable sensors and then logging)

#sudo apt install libboost1.67-dev
sudo apt install libboost-all-dev
sudo apt install qt3d5-dev qt5-default libqt5charts5-dev libqt5charts5-dev libexiv2-dev libassimp-dev libtbb-dev
sudo apt install python3-numpy #unfortunately, comma will be installed using sudo make install, requires numpy also installed using sudo
cd ~/src/comma
mkdir build && cd build && cmake ..
make -j$(nproc)
sudo make install

#find_package(TBB REQUIRED)
#target_link_libraries(cv-cat TBB::tbb)
cd ~/src/snark
mkdir build
cd build
cmake \
    -Dsnark_build_imaging_opencv_contrib=OFF \
    -Dsnark_build_navigation=ON \
    -Dsnark_build_sensors_dc1394=OFF \
    -Dsnark_build_sensors_ouster=ON \
    -Dsnark_build_ros=OFF \
    ..
make -j$(nproc)
sudo make install

# Build limap dependencies and install
python3 -m pip install h5py
cd ~/src/window-tracker
git submodule update --init --recursive

cd limap
export LIMAP=$PWD
cd third-party/TP-LSD
git checkout 5558050
cd tp_lsd/modeling
rm -r DCNv2
git clone https://github.com/lucasjinreal/DCNv2_latest.git DCNv2
cd $LIMAP

# Navigate to the directory containing the setup.py file
#cd third-party/hawp
# Use sed to replace the find_packages argument
#if [ ! -f setup.py ]; then
#  echo "setup.py not found!"
#  exit 1
#fi
#sed -i "s/packages=find_packages(\['hawp'\])/packages=find_packages(where='hawp')/g" setup.py
#echo "Modified setup.py successfully."
#cd $LIMAP

python3 -m pip install -r requirements.txt
python3 -m pip install -Ive .
#python3 setup.py build
#python3 setup.py install

# Check if the limap package is installed
python3 -c "import limap" && echo "LIMAP package installed successfully" || echo "Failed to install LIMAP package"

# demo related, coordinate frame projections
python3 -m pip install pyproj

# compile orbslam3
sudo apt install libepoxy-dev
cd ~/src
git clone https://github.com/stevenlovegrove/Pangolin
cd Pangolin/
mkdir build
cd build
cmake ..
make -j$(nproc)
sudo make install

sudo apt install libncurses-dev libncursesw5-dev libffi-dev
cd ~/src/window-tracker/ORB_SLAM3
./build.sh

echo "Installation completed successfully."

