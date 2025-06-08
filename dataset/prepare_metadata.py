import sys, os
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from deepfashion_loader import DeepFashionLoader
# Adding root path for backend imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.models import DINOv2EmbeddingGenerator, BERTEmbeddingGenerator
from backend.database import FaissIndexHandler


# Load config
base_dir = os.path.dirname(__file__)
config_path = os.path.abspath(os.path.join(base_dir, "../configs/app_config.yaml"))
config = OmegaConf.load(config_path)

# Prepare export path for metadata
EXPORT_FILE = config.mongo.seed_data_path
os.makedirs(os.path.dirname(EXPORT_FILE), exist_ok=True)

# Convert image to base64
def prepare_image_for_mongodb(image: Image.Image, size=(224, 224)) -> str:
    assert isinstance(image, Image.Image), f"Expected PIL.Image, got {type(image)}"
    resized = image.resize(size, Image.BICUBIC)
    buffer = BytesIO()
    resized.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Load dataset
dataset = DeepFashionLoader(
    image_root=config.dataset.image_root,
    labels_root=config.dataset.labels_root,
    captions_file=config.dataset.captions_file
)

# Initializing model embedders
bert_model = BERTEmbeddingGenerator(
    model_name=config.models.bert.name,
    max_length = 128)
    
dino_model = DINOv2EmbeddingGenerator(
    model_name=config.models.dinov2.name,
    shape = (224,224))

# Initializing FAISS indexers
bert_indexer = FaissIndexHandler(
    dimension=config.faiss.bert.dimension,
    saved_index_path=config.faiss.bert.path
)

dino_indexer = FaissIndexHandler(
    dimension=config.faiss.dinov2.dimension,
    saved_index_path=config.faiss.dinov2.path
)

# Accumulate MongoDB documents
all_docs = []


# Process samples
for i in tqdm(np.random.randint(100, len(dataset), size=2000)):
    sample = dataset[i]
    image_file = str(sample["image_file"])
    image = sample["image"]
    image_base64 = prepare_image_for_mongodb(image)
    caption = str(sample["caption"])
    metadata = sample["metadata"]
    numeric_id = int(sample["numeric_id"])
    id_array = np.array([numeric_id], dtype=np.int64)  ## array([value])
    # Prepare document
    doc = {
        "image_file": image_file,
        "numeric_id": numeric_id,
        "image": image_base64,
        "metadata": metadata
    }
    all_docs.append(doc)

    # Generate embeddings and add to FAISS index
    bert_embedding = bert_model.encode(text=caption) # [1,768] Numpy
    dino_embedding = dino_model.encode(image=image) # [B, 768] Numpy

    bert_indexer.add_to_index(id_array, bert_embedding)
    dino_indexer.add_to_index(id_array, dino_embedding)

# Save FAISS indices
bert_indexer.save_index()
dino_indexer.save_index()

# Save metadata to local JSON
with open(EXPORT_FILE, "w") as f:
    json.dump(all_docs, f, indent=2)

print(f"✅ Prepared and saved {len(all_docs)} product entries to {EXPORT_FILE}")