from models.base_model import BaseModel
import base64
import io
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class MyQwen3VL(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_id = getattr(self.config, 'model_id', None) or getattr(self.config, 'model', None)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map='auto' if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        self.create_ask_message = lambda question: {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ans},
            ],
        }

    def create_text_message(self, texts, question):
        content = []
        for text in texts:
            content.append({"type": "text", "text": text})
        content.append({"type": "text", "text": question})
        message = {
            "role": "user",
            "content": content
        }
        return message

    def create_image_message(self, images, question):
        content = []
        for image_path in images:
            data_uri = f"data:image/jpeg;base64,{encode_image(image_path)}"
            content.append({"type": "image_url", "image_url": {"url": data_uri}})
        content.append({"type": "text", "text": question})
        message = {
            "role": "user",
            "content": content
        }
        return message

    def _messages_to_prompt(self, messages):
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            for c in msg.get("content", []):
                t = c.get("type")
                if t == "text":
                    parts.append(c.get("text", ""))
                elif t == "image_url":
                    url = c.get("image_url", {}).get("url", "")
                    parts.append(f"[Image:{url}]")
                elif t == "image":
                    parts.append(f"[Image:{c.get('image','')}]")
                else:
                    parts.append(str(c))
        prompt = "\n".join(parts)
        return prompt

    def _fallback_vision_info(self, messages):
        images = []
        for msg in messages:
            for c in msg.get("content", []):
                if c.get("type") != "image_url":
                    continue
                url = c.get("image_url", {}).get("url", "")
                if not isinstance(url, str) or not url.startswith("data:image"):
                    continue
                try:
                    b64_data = url.split(",", 1)[1]
                    img_bytes = base64.b64decode(b64_data)
                    images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                except Exception:
                    continue
        return images, []

    def predict(self, question, texts=None, images=None, history=None):
        messages = self.process_message(question, texts, images, history)
        # prepare inputs via processor for VL chat
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = None, None
        try:
            # optional helper if available
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
        except Exception:
            # fallback: recover images directly from message data URIs
            image_inputs, video_inputs = self._fallback_vision_info(messages)

        if not image_inputs:
            image_inputs = None
        if not video_inputs:
            video_inputs = None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=getattr(self.config, 'max_new_tokens', 256))
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        messages.append(self.create_ans_message(output))
        
        token_usage = None
        if self.enable_token_counting:
            input_tokens = len(inputs.input_ids[0])
            output_tokens = len(generated_ids_trimmed[0])
            token_usage = {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            }
        
        return output, messages, token_usage

    def is_valid_history(self, history):
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(item["content"], list):
                return False
            for content in item["content"]:
                if not isinstance(content, dict):
                    return False
                if "type" not in content:
                    return False
                if content["type"] not in content:
                    return False
        return True
