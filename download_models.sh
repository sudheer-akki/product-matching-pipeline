#!/bin/bash

if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown || pip3 install gdown || { echo "Install pip first."; exit 1; }
fi

if ! command -v unzip &> /dev/null; then
    echo "Installing unzip..."
    sudo apt-get install unzip -y || { echo "Install unzip manually."; exit 1; }
fi

echo "Downloading ONNX models..."

# Define model URLs
BERT_URL="1kQBZIqW6MaB82NYigNK4L-JYN-P0MJXa"
DINO_URL="1JJegXQXesViCYWfsmG9vnGbJYBCFDIez"
BERT_INDEX="14ssyYM-gljeDX8hUB7SNn-EKJ3zwFaOP"
DINO_INDEX="1Drs2Mar9BIF0n4pJlt0q1M1SJLfhKWBG"
MONGO_JSON="1o2RtSFqeb-kn0dbIMGye6-wuyhd76dv"
LLAVA_VISION_ENCODER="1T7jqgP1eDe3n6GMFXQUrKf7nuTn1YIFE"
LLAVA_VISION="1pbkiRZfmRowECqy6GBU6dcWvZ4D0Pawf"

# Checking and creating destination directories if not exist
mkdir -p models/engines/bert
mkdir -p models/engines/dinov2
mkdir -p db/faiss
mkdir -p db/mongo
mkdir -p models

# Download only if file doesn't exist
# BERT ONNX
if [ ! -f models/engines/bert/bert.onnx ]; then
    echo "Downloading bert.onnx..."
    gdown --no-cookies --id "$BERT_URL" -O models/engines/bert/bert.onnx
else
    echo "bert.onnx already exists, skipping."
fi

# DINO ONNX
if [ ! -f models/engines/dinov2/dinov2.onnx ]; then
    echo "Downloading dinov2.onnx..."
    gdown --no-cookies --id "$DINO_URL" -O models/engines/dinov2/dinov2.onnx
else
    echo "dinov2.onnx already exists, skipping."
fi

# BERT INDEX
if [ ! -f db/faiss/bert.index ]; then
    echo "Downloading bert.index..."
    gdown --no-cookies --id "$BERT_INDEX" -O db/faiss/bert.index
else
    echo "bert.index already exists, skipping."
fi

# DINO INDEX
if [ ! -f db/faiss/dino.index ]; then
    echo "Downloading dino.index..."
    gdown --no-cookies --id "$DINO_INDEX" -O db/faiss/dino.index
else
    echo "dino.index already exists, skipping."
fi

# Mongo JSON
if [ ! -f db/mongo/data.json ]; then
    echo "Downloading data.json..."
    gdown --no-cookies --id "$MONGO_JSON" -O db/mongo/data.json
else
    echo "data.json already exists, skipping."
fi

# Llava Vision Encoder ZIP
if [ ! -d models/llava_vision_encoder ]; then
    echo "Downloading llava_vision_encoder.zip..."
    gdown --no-cookies --id "$LLAVA_VISION_ENCODER" -O models/llava_vision_encoder.zip
    unzip -o models/llava_vision_encoder.zip -d models
    rm models/llava_vision_encoder.zip
else
    echo "llava_vision_encoder directory already exists, skipping download and unzip."
fi

# Llava Vision ZIP
if [ ! -d models/llava_vision ]; then
    echo "Downloading llava_vision.zip..."
    gdown --no-cookies --id "$LLAVA_VISION" -O models/llava_vision.zip
    unzip -o models/llava_vision.zip -d models
    rm models/llava_vision.zip
else
    echo "llava_vision.zip already exists, skipping."
    echo "llava_vision directory already exists, skipping download and unzip."
fi

echo "Models downloaded successfully."