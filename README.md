# Product Matching Pipeline 

Supports both **image** and **text** Inputs.

---

## Models Used

- **🖼️ DINOv2** – Generates Image Embeddings  
- **🧠 BERT** – Generates text embeddings  
- **📸 LLaVA** – Captioning Input Image

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
│      ├── Dockerfile.triton
│      └── Dockerfile.trtllm
│
├── frontend
│   └─── app.py
│
├── models
│    └─── engines
│       ├── bert
│       │   ├── 1
│       │   │   └── model.plan
│       │   ├── bert.onnx
│       │   └── config.pbtxt
│       ├── dinov2
│       │   ├── 1
│       │   │   └── model.plan
│       │   ├── config.pbtxt
│       │   └── dinov2.onnx
│       ├── llava
│       │   ├── 1
│       │   │   └── model.engine
│       │   ├── config.pbtxt
│       └── llava_to_bert
│            ├── 1
│            │   └── model.py
│            └── config.pbtxt
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

## Work Flow

<img src="assets/workflow.drawio.png" width="600" height="300">
---

## 📦 Dataset – DeepFashion

We use a **subset of the DeepFashion dataset** around 2k fashion product images:

- ✅ High-resolution product images  
- ✅ Product metadata (title, description, category)  
- ✅ Fine-grained attributes  
- 📚 [DeepFashion Dataset Info](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)

---

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
**Note:** Makesure to have files loaded into below folder

+ bert_onnx -> models/engines/bert/bert.onnx
+ dino_onnx -> models/engines/dinov2/dino.onnx
+ bert index -> db/faiss/bert.index
+ dino index -> db/faiss/dinov2.index
+ MongoDB Metadata -> db/mongo/metadata.json

## 3. Create TensorRT Engines or Copy them into models/engines folder

* docker compose up triton -d
* docker exec -it container_id bash
* Run below command

```sh
bash create_engines.sh
```

### 4. Start the Application using Docker

```
docker compose up -d
```

**Visit:** Frontend GUI on "http://localhost:7860"

## 5. Upload Image from dataset/sample folder

See the results with metadata
