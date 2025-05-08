#!/bin/bash

# Fallback for symbols if not in UTF-8 locale
if [[ $(locale charmap) != "UTF-8" ]]; then
    CHECK_MARK="[OK]"
    CROSS_MARK="[ERR]"
else
    CHECK_MARK="✔️"
    CROSS_MARK="❌"
fi

MISSING_TOOLS=()

# Function to check the availability and print the version of a command
check_and_print_version() {
    command -v "$1" &> /dev/null
    if [ $? -eq 0 ]; then
        echo -e "${CHECK_MARK} $2 is available."
        eval "$3"
    else
        echo -e "${CROSS_MARK} Error: $2 is NOT available."
        MISSING_TOOLS+=("$2")
    fi
}

# cmake
check_and_print_version "cmake" "cmake" "cmake --version | head -n 1"

# nvcc (CUDA Compiler)
check_and_print_version "nvcc" "nvcc (CUDA compiler)" "nvcc --version | grep release"

# CUDA Runtime Library
if ldconfig -p | grep -q "libcudart"; then
    echo -e "${CHECK_MARK} CUDA runtime library is available."
else
    echo -e "${CROSS_MARK} Error: CUDA runtime library is NOT available."
    MISSING_TOOLS+=("CUDA Library")
fi

# cuDNN
if ldconfig -p | grep -q "libcudnn"; then
    echo -e "${CHECK_MARK} cuDNN is available."
    CUDNN_HEADER=$(find /usr/include /usr/local/include /usr/local/ -name "cudnn_version.h" 2>/dev/null | head -n 1)
    if [ -f "$CUDNN_HEADER" ]; then
        MAJOR=$(grep "#define CUDNN_MAJOR" "$CUDNN_HEADER" | awk '{print $3}')
        MINOR=$(grep "#define CUDNN_MINOR" "$CUDNN_HEADER" | awk '{print $3}')
        PATCH=$(grep "#define CUDNN_PATCHLEVEL" "$CUDNN_HEADER" | awk '{print $3}')
        echo "cuDNN version: ${MAJOR}.${MINOR}.${PATCH}"
    else
        echo "cuDNN version file not found. Installed, but version unknown."
    fi
else
    echo -e "${CROSS_MARK} Error: cuDNN is NOT available."
    MISSING_TOOLS+=("cuDNN")
fi

# TensorRT (shared lib and Python API)
if ldconfig -p | grep -q "libnvinfer"; then
    echo -e "${CHECK_MARK} TensorRT library is available."
    if python3 -c "import tensorrt" &> /dev/null; then
        python3 -c "import tensorrt as trt; print('TensorRT version:', trt.__version__)"
    else
        echo "TensorRT Python API not found. Version detection skipped."
    fi
else
    echo -e "${CROSS_MARK} Error: TensorRT is NOT available."
    MISSING_TOOLS+=("TensorRT")
fi

# Check for DeepStream SDK
if command -v deepstream-app &> /dev/null; then
    echo -e "${CHECK_MARK} DeepStream SDK is available."

    # Capture the version output
    VERSION_OUTPUT=$(deepstream-app --version 2>&1 | grep -i "DeepStreamSDK")
    if [[ -n "$VERSION_OUTPUT" ]]; then
        echo "$VERSION_OUTPUT"
    else
        echo "DeepStream version detected, but exact version string not found."
    fi
else
    echo -e "${CROSS_MARK} Error: DeepStream SDK is NOT available."
    MISSING_TOOLS+=("DeepStream SDK")
fi

# ROS
if [ -n "$ROS_DISTRO" ]; then
    echo -e "${CHECK_MARK} ROS ($ROS_DISTRO) is available."
else
    echo -e "${CROSS_MARK} Error: ROS is NOT available."
    MISSING_TOOLS+=("ROS")
fi

# catkin_make
command -v catkin_make &> /dev/null
if [ $? -eq 0 ]; then
    echo -e "${CHECK_MARK} catkin_make is available."
    catkin_make --help | head -n 1
else
    echo -e "${CROSS_MARK} Error: catkin_make is NOT available."
    MISSING_TOOLS+=("catkin")
fi

# comma (csv-play)
check_and_print_version "csv-play" "comma (csv-play)" "csv-play --help | head -n 1"

# snark (cv-cat)
check_and_print_version "cv-cat" "snark (cv-cat)" "cv-cat --help | head -n 1"

# conda
check_and_print_version "conda" "Anaconda3 (conda)" "conda --version"

# Final summary
echo
echo "========== Summary =========="
if [ ${#MISSING_TOOLS[@]} -eq 0 ]; then
    echo -e "${CHECK_MARK} All tools are available."
else
    echo -e "${CROSS_MARK} Missing tools: ${MISSING_TOOLS[*]}"
fi
echo "============================="

