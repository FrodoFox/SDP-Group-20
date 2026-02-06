#!/bin/bash

# --- CONFIGURATION ---
USER="ubuntu"
HOST="weepinbell"
PACKAGE="opencv-python"
PY_VERSION="3.12"
DEST_PATH="~/.local/lib/python3.12/site-packages"

echo "🚀 Starting deployment..."

# Setting up temporary workspace to download and prepare the package
rm -rf ./bundle_work && mkdir ./bundle_work
cd ./bundle_work

# Downloading the packing
python3 -m pip download --only-binary=:all: --platform manylinux2014_aarch64 --python-version $PY_VERSION $PACKAGE

# Unzipping library and removing wheel files
unzip *.whl
rm *.whl

# 4. Create destination on TurtleBot
ssh ${USER}@${HOST} "mkdir -p $DEST_PATH"

# Moving library to turtlebot
echo "📦 Transferring folders to $DEST_PATH..."
scp -rp . ${USER}@${HOST}:${DEST_PATH}/

echo "✅ Transfer complete. Checking Robot..."
ssh ${USER}@${HOST} "ls -d $DEST_PATH/cv2"