# Product Matching Pipeline 

Supports both **image** and **text** Inputs.

**Token Limit Notice:**
+ Text input is limited to a maximum of 128 tokens.
+ All input texts are tokenized using **bert-base-uncased**.
+ Inputs longer than 128 tokens will be automatically truncated.

**Note:** For best results, keep prompts short, focused, and within 1–2 sentences.

---

## Models Used

- **🖼️ DINOv2** – Generates Image Embeddings  
- **🧠 BERT** – Generates text embeddings  
- **📸 LLaVA-OneVision** – Captioning Input Image

---

## Tech Stack

-  **FastAPI** – Backend API  
-  **Gradio** – Web Frontend GUI  
-  **Triton Inference Server** – TensorRT Optimized model deployment  
-  **FAISS** – Vector Database
-  **MongoDB** – Stores metadata (Including Images)  
-  **Docker Compose** – Orchestrates all Containers
---

## Features

- Supports **Image** or **Text** based Semantic Search
- TensorRT-powered inference via **Triton**
- FAISS + MongoDB integration for embeddings & metadata Storage
- Easy-to-use **Gradio UI**
---
## 📁 Project Structure

```
├── LICENSE
├── README.md
├── assets
│   ├── workflow.drawio
│   └── workflow.drawio.png
├── backend
│   ├── __init__.py
│   ├── batching
│   │   ├── __init__.py
│   │   └── base.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── app_state.py
│   │   ├── config_loader.py
│   │   └── custom_log.py
│   ├── database
│   │   ├── __init__.py
│   │   ├── faiss_db.py
│   │   └── mongo_db.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── bert_model.py
│   │   ├── dinov2_model.py
│   │   └── llava_model.py
│   ├── routes
│   │   ├── __init__.py
│   │   ├── api_router.py
│   │   ├── image_search.py
│   │   ├── logger.py
│   │   └── text_search.py
│   ├── services
│   │   ├── __init__.py
│   │   ├── img_vector_search.py
│   │   ├── llava_bert.py
│   │   ├── llava_runner.py
│   │   ├── mongodb.py
│   │   ├── text_vector_search.py
│   │   └── triton_client.py
│   └── utils
│       ├── __init__.py
│       ├── logging.py
│       └── rerank.py
├── main.py
│ 
├── configs
│   └── app_config.yaml
├── dataset
│   ├── sample
│   │   └── images
│   ├── deepfashion_loader.py
│   └── prepare_metadata.py
├── db
│   ├── faiss
│   │     ├── bert_index.faiss
│   │     └── dino_index.faiss
│   └── mongo
│        └── metadata.json
├── docker         
│      ├── Dockerfile.backend
│      ├── Dockerfile.mongo
│      └── Dockerfile.triton
│
├── frontend
│   └─── app.py
│
├── engines
│   ├── bert
│   │   ├── 1
│   │   │   └── model.plan
│   │   ├── bert.onnx
│   │   └── config.pbtxt
│   ├── dinov2
│   │   ├── 1
│   │   │   └── model.plan
│   │   ├── config.pbtxt
│   │   └── dinov2.onnx
│   └── llava_to_bert
│       ├── 1
│       │   └── model.py
│       └── config.pbtxt
├── llava_vision
│   ├── config.json
│   └── rank0.engine
└── llava_vision_encoder
    ├── config.json
    ├── image_newlines.safetensors
    └── model.engine

├── scripts     
│    ├── bert_onnx_convert.py
│    ├── dinov2_onnx_convert.py
│    ├── download_build_llava.sh
│    ├── test.jpg
│    └── test_triton_client.py
├── trt_engine       
│     ├── __init__.py
│     ├── bert_engine.py
│     ├── dinov2_engine.py
│     ├── llava_engine.py
│     ├── llava_utils.py
│     └── trt_base.py
│
├── docker-compose.yaml
├── requirements_app.txt
└── requirements_server.txt
```
---

## Workflow

<img src="assets/workflow.drawio.png" width="600" height="300">
---

## 📦 Dataset – DeepFashion

We use a **subset of the DeepFashion dataset** around 2k fashion product images:

- ✅ High-resolution product images  
- ✅ Product metadata (title, description, category)  
- ✅ Fine-grained attributes  
- 📚 [DeepFashion Dataset Info](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)

---
## Triton Platform Specs:
+ GPU: NVIDIA GeForce RTX 4090 (24GB VRAM)
+ CUDA: Version 12.8
+ Driver: 570.124.06
+ CUDA Compiler: 12.6
+ CPU: AMD EPYC 7282, 16-Core

## ⚡ Quick Start 

### 1. Clone the Repo
```bash
git clone https://github.com/sudheer-akki/product-matching-pipeline
cd /product-matching-pipeline
```

### 2. Download ONNX Models & Sample Database

```
bash download_models.sh
```
**Note:** Make sure the following files are downloaded and in place:

+ bert_onnx -> models/engines/bert/bert.onnx
+ dino_onnx -> models/engines/dinov2/dino.onnx
+ bert index -> db/faiss/bert.index
+ dino index -> db/faiss/dinov2.index
+ MongoDB Metadata -> db/mongo/metadata.json

### 3. Update configuration data

+ open **configs/app_config.yaml** file

**Note:** Update values if needed

### 4. Start the Application (Docker)

```
docker compose up -d
```
This command will automatically:

+ Build Docker engines for each model if not already present.

+ Start the Triton Inference Server with all necessary backends and dependencies.

### Access the Frontend GUI: http://localhost:7860

## 5. Upload Image
+ Open the GUI in your browser.
+ Upload an image from the **dataset/sample** folder.
+ Check the output results for product matches!
