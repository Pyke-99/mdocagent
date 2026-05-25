from models.base_model import BaseModel
import torch
import transformers

class Llama3(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.create_ask_message = lambda question: {
            "role": "user",
            "content": question,
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": ans,
        }
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.config.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "auto"},
        )
    
    def create_text_message(self, texts, question): 
        prompt = ""
        for text in texts:
            prompt = prompt + text + '\n'
        message = {
            "role": "user",
            "content": f"{prompt}\n{question}",
        }
        return message
    
    @torch.no_grad()
    def predict(self, question, texts = None, images = None, history = None):
        self.clean_up()
        messages = self.process_message(question, texts, images, history)
        outputs = self.pipeline(
            messages,
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
        )
        
        token_usage = None
        if self.enable_token_counting:
            # Calculate input tokens
            input_text = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False)
            input_tokens = len(self.pipeline.tokenizer.encode(input_text))
            # Output tokens: only the generated part
            full_output_text = outputs[0]["generated_text"][-1]['content']
            output_tokens = len(self.pipeline.tokenizer.encode(full_output_text))
            token_usage = {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            }
        
        self.clean_up()
        return outputs[0]["generated_text"][-1]['content'], outputs[0]["generated_text"], token_usage
        
    def is_valid_history(self, history):
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(item["content"], str):
                return False
        return True
