import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List
from PIL import Image
import torch
from torchvision import transforms

class DeepFashionLoader:
    def __init__(self, 
        image_root: str = "deepfashion/images",
        labels_root: str = "deepfashion/labels",
        captions_file: str = "deepfashion/captions.json"):
        self.image_root = image_root
        self.labels_root = labels_root
        self.captions_file = captions_file
        self.captions = self._load_captions()
        self.attr_definitions = self._load_shape_attributes()
        self.fabric_definitions = ["denim", "cotton", "leather", "furry", "knitted", "chiffon", "other", "NA"]
        self.color_definitions = ["floral", "graphic", "striped", "pure color", "lattice", "other", "color block", "NA"]
        self.samples = self._load_samples()

    def _load_captions(self):
        if os.path.isfile(self.captions_file):
            with open(self.captions_file, 'r') as f:
                return json.load(f)
        return {}

    def _load_shape_attributes(self):
        return {
            "sleeve length": ["sleeveless", "short-sleeve", "medium-sleeve", "long-sleeve", "not long-sleeve", "NA"],
            "lower clothing length": ["three-point", "medium short", "three-quarter", "long", "NA"],
            "socks": ["no", "socks", "leggings", "NA"],
            "hat": ["no", "yes", "NA"],
            "glasses": ["no", "eyeglasses", "sunglasses", "glasses in hand/clothes", "NA"],
            "neckwear": ["no", "yes", "NA"],
            "wrist wearing": ["no", "yes", "NA"],
            "ring": ["no", "yes", "NA"],
            "waist accessories": ["no", "belt", "clothing", "hidden", "NA"],
            "neckline": ["V-shape", "square", "round", "standing", "lapel", "suspenders", "NA"],
            "outer clothing a cardigan?": ["yes", "no", "NA"],  # cardigan
            "upper clothing covering navel": ["no", "yes", "NA"]   # covers navel
        }
    
    def _load_samples(self):
        """
        Load all label data from the shape and texture folders and compile sample metadata.
        """
        samples = {}
        for subfolder in ['shape', 'texture']:
            folder_path = os.path.join(self.labels_root, subfolder)
            for filename in sorted(os.listdir(os.path.join(os.getcwd(),folder_path))):
                if filename.endswith(".txt"):
                    file_path = os.path.join(folder_path, filename)
                    self._parse_label_file(file_path, samples)
        return list(samples.values())
    
    def _parse_label_file(self, file_path, samples):
        """
        Parses a label file and updates the samples dictionary.
        """
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if not parts:
                    continue

                image_name = parts[0]
                values = list(map(int, parts[1:]))
                if image_name not in samples:
                    samples[image_name] = {
                        "image_file": image_name,
                        "image_path": os.path.join(self.image_root, image_name),
                        "attributes": {}
                    }

                if "fabric" in file_path.lower():
                    self._process_fabric_annotation(samples[image_name], values)
                elif "color" in file_path.lower():
                    self._process_color_annotation(samples[image_name], values)
                else:
                    self._process_shape_attributes(samples[image_name], values)

    def _process_fabric_annotation(self, sample, values):
        """
        Assigns fabric label to sample.
        """
        value = values[0]
        label = self.fabric_definitions[value] if 0 <= value < len(self.fabric_definitions) else "Unknown"
        sample["attributes"]["fabric"] = label

    def _process_color_annotation(self, sample, values):
        """
        Assigns color label to sample.
        """
        value = values[0]
        label = self.color_definitions[value] if 0 <= value < len(self.color_definitions) else "Unknown"
        sample["attributes"]["color"] = label

    def _process_shape_attributes(self, sample, values):
        """
        Assigns general shape-related attributes to sample.
        """
        for i, (attr_name, attr_values) in enumerate(self.attr_definitions.items()):
            val = values[i] if i < len(values) else -1
            label = attr_values[val] if 0 <= val < len(attr_values) else "NA"
            sample["attributes"][attr_name] = label # attr_values[values[i]] #label
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        #image = self._transform_data(image)
        #image = image.resize((224,224), Image.BICUBIC)
        #image = np.array(image).astype(np.float32) / 255.0
        filename_parts = sample["image_file"].split('-')
        gender = filename_parts[0] if len(filename_parts) > 0 else "unknown"
        category = filename_parts[1] if len(filename_parts) > 1 else "unknown"
        product_id = next((p for p in filename_parts if p.startswith("id_")), "unknown")
        caption = self.captions.get(sample["image_file"], "") 
        raw_attrs = sample["attributes"]
        metadata = {
            "gender": gender,
            "category": category,
            "product_id": product_id,
            "fabric": raw_attrs.get("fabric", "NA"),
            "sleeve length": raw_attrs.get("sleeve length", "NA")
        }
        return {
            "image_file": sample["image_file"],
            "numeric_id": idx,  
            "image": image,
            "caption": caption,
            "metadata": metadata
        }
    
    def _transform_data(self, image):
        transform_ = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # Converts to float32 [0,1]
            transforms.Normalize(
                mean=[0.4815, 0.4578, 0.4082],
                std=[0.2686, 0.2613, 0.2758]
            )
        ])
        return transform_(image)



    
if __name__=="__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    dataset = DeepFashionLoader()

    print(dataset.__getitem__(idx=10000))



        