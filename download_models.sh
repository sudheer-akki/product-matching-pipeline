#!/bin/bash

echo "Downloading ONNX models..."

# Define model URLs
BERT_URL="1kQBZIqW6MaB82NYigNK4L-JYN-P0MJXa"
DINO_URL="1JJegXQXesViCYWfsmG9vnGbJYBCFDIez"
BERT_INDEX="14ssyYM-gljeDX8hUB7SNn-EKJ3zwFaOP"
DINO_INDEX="1Drs2Mar9BIF0n4pJlt0q1M1SJLfhKWBG"
MONGO_JSON="1o2RtSFqeb-kn0dbIMGye6-wuyhd76dv"


# Checking and creating destination directories if not exist
[ ! -d "models/engines/bert" ] && mkdir -p models/engines/bert
[ ! -d "models/engines/dinov2" ] && mkdir -p models/engines/dinov2

# Download files
gdown "https://drive.google.com/uc?id=${BERT_URL}" -O models/engines/bert/bert.onnx
gdown "https://drive.google.com/uc?id=${DINO_URL}" -O models/engines/dinov2/dinov2.onnx
gdown "https://drive.google.com/uc?id=${BERT_INDEX}" -O db/faiss/bert.index
gdown "https://drive.google.com/uc?id=${DINO_INDEX}" -O db/faiss/dino.index
gdown "https://drive.google.com/uc?id=${MONGO_JSON}" -O db/mongo/data.json

echo "Models downloaded successfully."
