import tritonclient.http as httpclient
import numpy as np
from transformers import BertTokenizer

# Initialize Triton client
client = httpclient.InferenceServerClient("localhost:8000")

# Prepare input text
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello Triton!", return_tensors="np", padding="max_length", truncation=True, max_length=128)

# Prepare inputs for Triton
input_ids = httpclient.InferInput("input_ids", inputs["input_ids"].shape, "INT64")
attention_mask = httpclient.InferInput("attention_mask", inputs["attention_mask"].shape, "INT64")

input_ids.set_data_from_numpy(inputs["input_ids"])
attention_mask.set_data_from_numpy(inputs["attention_mask"])

# Define desired output
outputs = [httpclient.InferRequestedOutput("last_hidden_state")]

# Send inference request
response = client.infer(
    model_name="bert",
    inputs=[input_ids, attention_mask],
    outputs=outputs
)

# Get and print output
result = response.as_numpy("last_hidden_state")
print("Output shape:", result.shape)
