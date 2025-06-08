import triton_python_backend_utils as pb_utils
import numpy as np
from transformers import BertTokenizer

class TritonPythonModel:
    def initialize(self, args):
        self.llava_model = "llava"
        self.bert_model = "bert_trt"
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def execute(self, requests):
        request = requests[0]
        responses = []
        # 1. Extract batched inputs
        images = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()         # [B]
        prompts = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()       # [B]
        max_tokens = pb_utils.get_input_tensor_by_name(request, "max_tokens").as_numpy()
        temperature = pb_utils.get_input_tensor_by_name(request, "temperature").as_numpy()
        top_k = pb_utils.get_input_tensor_by_name(request, "top_k").as_numpy()
        freq_penalty = pb_utils.get_input_tensor_by_name(request, "frequency_penalty").as_numpy()
        seed = pb_utils.get_input_tensor_by_name(request, "seed").as_numpy()
        batch_size = images.shape[0]

        # 2. Send batched request to LLaVA
        llava_inputs = [
            pb_utils.Tensor("image", images),
            pb_utils.Tensor("prompt", prompts),
            pb_utils.Tensor("max_tokens", max_tokens),
            pb_utils.Tensor("temperature", temperature),
            pb_utils.Tensor("top_k", top_k),
            pb_utils.Tensor("frequency_penalty", freq_penalty),
            pb_utils.Tensor("seed", seed)
        ]

        # llava_result = pb_utils.InferenceRequest(
        # model_name=self.llava_model,
        # inputs=llava_inputs,
        # requested_output_names=["text"]
        # ).exec()

        # captions = llava_result.as_numpy("text")  # [B] byte strings

        dummy_caption = "The person wears a short-sleeve T-shirt with solid color patterns"
        captions = np.array([dummy_caption.encode("utf-8")] * batch_size)

        # 3. Tokenize all captions in batch
        decoded_captions = [cap.decode("utf-8") for cap in captions]
        tokens = self.tokenizer(
            decoded_captions,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=128
        )

        # 4. Send batched BERT request
        bert_inputs = [
            pb_utils.Tensor("input_ids", tokens["input_ids"].astype(np.int32)),
            pb_utils.Tensor("attention_mask", tokens["attention_mask"].astype(np.int32))
        ]

        bert_result = pb_utils.InferenceRequest(
            model_name=self.bert_model,
            inputs=bert_inputs,
            requested_output_names=["pooled_output"]
        ).exec()

        embeddings = bert_result.as_numpy("pooled_output")  # [B, 768]

        # 5. Build responses
        for i in range(batch_size):
            out_caption = pb_utils.Tensor("text", np.array([captions[i]], dtype=np.object_))
            out_embed = pb_utils.Tensor("pooled_output", embeddings[i:i+1])
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_embed, out_caption]))

        return responses
    
        # for request in requests:
        #     try:
        #         # image and prompt preprocessing
        #         image = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()
        #         prompt = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0].decode("utf-8")
        #         max_tokens = pb_utils.get_input_tensor_by_name(request, "max_tokens").as_numpy()
        #         top_k = pb_utils.get_input_tensor_by_name(request, "top_k").as_numpy()
        #         temperature = pb_utils.get_input_tensor_by_name(request, "temperature").as_numpy()
        #         frequency_penalty = pb_utils.get_input_tensor_by_name(request, "frequency_penalty").as_numpy()
        #         seed = pb_utils.get_input_tensor_by_name(request, "seed").as_numpy()

        #         llava_inputs = [
        #             pb_utils.Tensor("image", image),
        #             pb_utils.Tensor("prompt", np.array([prompt.encode("utf-8")], dtype=np.object_)),
        #             pb_utils.Tensor("max_tokens", max_tokens),
        #             pb_utils.Tensor("temperature", temperature),
        #             pb_utils.Tensor("top_k", top_k),
        #             pb_utils.Tensor("frequency_penalty", frequency_penalty),
        #             pb_utils.Tensor("seed", seed),
        #         ]

        #         try:
        #             llava_result = pb_utils.InferenceRequest(
        #             model_name=self.llava_model,
        #             inputs=llava_inputs,
        #             requested_output_names=["text"]
        #             ).exec() 
        #         except Exception as e:
        #             error_msg = f"LLaVA-BERT pipeline failed: {str(e)}"
        #             responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(error_msg)))
        #             return responses

        #         caption = llava_result.as_numpy("text")[0].decode("utf-8")

        #         # Tokenize caption
        #         tokens = self.tokenizer(
        #             caption,
        #             return_tensors="np",
        #             padding="max_length",
        #             truncation=True,
        #             max_length=128
        #         )

        #         # 4. Run BERT
        #         bert_result = pb_utils.InferenceRequest(
        #             model_name=self.bert_model,
        #             inputs=[
        #                 pb_utils.Tensor("input_ids", tokens["input_ids"].astype(np.int32)),
        #                 pb_utils.Tensor("attention_mask", tokens["attention_mask"].astype(np.int32)),
        #             ],
        #             requested_output_names=["pooled_output"]
        #         ).exec()

        #         embedding = bert_result.as_numpy("pooled_output")[0]

        #         output_caption = pb_utils.Tensor("text", np.array([caption.encode("utf-8")], dtype=np.object_))
        #         output_embedding = pb_utils.Tensor("pooled_output", np.expand_dims(embedding, axis=0))
        #         responses.append(pb_utils.InferenceResponse(output_tensors=[output_embedding, output_caption]))
        #     except Exception as e:
        #         error_msg = f"Triton Server LLaVA-BERT pipeline failed: {str(e)}"
        #         responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(error_msg)))
        # return responses
