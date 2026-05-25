from agents.base_agent import Agent
from mydatasets.base_dataset import BaseDataset
from tqdm import tqdm
import importlib
import json
import torch
from typing import List
import os

class MultiAgentSystem:
    def __init__(self, config):
        self.config = config
        self.agents:List[Agent] = []
        self.models:dict = {}
        for agent_config in self.config.agents:
            if agent_config.model.class_name not in self.models:
                module = importlib.import_module(agent_config.model.module_name)
                model_class = getattr(module, agent_config.model.class_name)
                print("Create model: ", agent_config.model.class_name)
                self.models[agent_config.model.class_name] = model_class(agent_config.model)
            self.add_agent(agent_config, self.models[agent_config.model.class_name])
            
        if config.sum_agent.model.class_name not in self.models:
            module = importlib.import_module(config.sum_agent.model.module_name)
            model_class = getattr(module, config.sum_agent.model.class_name)
            self.models[config.sum_agent.model.class_name] = model_class(config.sum_agent.model)
        self.sum_agent = Agent(config.sum_agent, self.models[config.sum_agent.model.class_name])
        
    def add_agent(self, agent_config, model):
        module = importlib.import_module(agent_config.agent.module_name)
        agent_class = getattr(module, agent_config.agent.class_name)
        agent:Agent = agent_class(agent_config, model)
        self.agents.append(agent)
        
    def predict(self, question, texts, images):
        '''Implement the method in the subclass'''
        pass

    def _has_valid_answer(self, sample):
        if self.config.ans_key not in sample:
            return False
        value = sample.get(self.config.ans_key)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        return True

    def _predict_with_runtime_fallbacks(self, question, texts, images):
        attempts = [
            (texts, images, "full"),
            ((texts or [])[:4], (images or [])[:2], "reduced"),
            ((texts or [])[:2], [], "text_only"),
            ([], (images or [])[:1], "image_only"),
        ]

        last_error = None
        for cur_texts, cur_images, _ in attempts:
            try:
                return self.predict(question, cur_texts, cur_images)
            except RuntimeError as e:
                last_error = e
                print(e)
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                continue

        print(f"All runtime fallback attempts failed: {last_error}")
        return ("Insufficient information to answer the question.", None, None)
    
    def _normalize_token_usage_entry(self, usage):
        if not isinstance(usage, dict):
            return None

        if any(k in usage for k in ["prompt_tokens", "completion_tokens", "total_tokens"]):
            return {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            }

        if any(k in usage for k in ["input", "output", "total"]):
            input_tokens = int(usage.get("input", 0) or 0)
            output_tokens = int(usage.get("output", 0) or 0)
            total_tokens = int(usage.get("total", input_tokens + output_tokens) or 0)
            return {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": total_tokens,
            }

        return None

    def _merge_token_usage(self, *usages):
        if not usages:
            return None
        merged = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        found = False
        for usage in usages:
            normalized = self._normalize_token_usage_entry(usage)
            if not normalized:
                continue
            for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                merged[key] += normalized[key]
            found = True
        if not found:
            return None
        if merged["total_tokens"] == 0:
            merged["total_tokens"] = merged["prompt_tokens"] + merged["completion_tokens"]
        return merged

    def sum(self, sum_question):
        ans, all_messages, token_usage = self.sum_agent.predict(sum_question)
        def extract_final_answer(agent_response):
            try:
                response_dict = json.loads(agent_response)
                answer = response_dict.get("Answer", None)
                return answer
            except:
                return agent_response
        final_ans = extract_final_answer(ans)
        return final_ans, all_messages, token_usage

    def predict_dataset(self, dataset:BaseDataset, resume_path = None):
        samples = dataset.load_data(use_retreival=True)
        if resume_path:
            assert os.path.exists(resume_path)
            with open(resume_path, 'r') as f:
                samples = json.load(f)
        if self.config.truncate_len:
            samples = samples[:self.config.truncate_len]
            
        sample_no = 0
        for sample in tqdm(samples):
            if resume_path and self._has_valid_answer(sample):
                continue
            question, texts, images = dataset.load_sample_retrieval_data(sample)
            predict_output = self._predict_with_runtime_fallbacks(question, texts, images)

            final_ans = None
            final_messages = None
            reorder_result = None
            token_usage = None
            if isinstance(predict_output, tuple):
                if len(predict_output) >= 2:
                    final_ans, final_messages = predict_output[0], predict_output[1]
                if len(predict_output) >= 3:
                    third = predict_output[2]
                    if isinstance(third, dict) and any(k in third for k in ["prompt_tokens", "completion_tokens", "total_tokens"]):
                        token_usage = third
                    else:
                        reorder_result = third
                if len(predict_output) >= 4:
                    token_usage = predict_output[3]
            else:
                final_ans = predict_output

            sample[self.config.ans_key] = final_ans
            if self.config.save_message:
                sample[self.config.ans_key+"_message"] = final_messages
            if token_usage is not None:
                sample["token_usage"] = token_usage
            if getattr(self.config, "save_reorder", False) and reorder_result is not None:
                sample[self.config.ans_key+"_reorder"] = reorder_result
            torch.cuda.empty_cache()
            self.clean_messages()
            
            sample_no += 1
            if sample_no % self.config.save_freq == 0:
                path = dataset.dump_reults(samples)
                print(f"Save {sample_no} results to {path}.")
        path = dataset.dump_reults(samples)
        print(f"Save final results to {path}.")
        
        # Output separate token usage file
        token_usage_samples = []
        for sample in samples:
            token_usage_samples.append({
                "doc_id": sample.get("doc_id"),
                "question": sample.get("question"),
                "token_usage": sample.get("token_usage"),
            })
        base, _ = os.path.splitext(path)
        token_usage_path = base + "_token_usage.json"
        with open(token_usage_path, "w") as f:
            json.dump(token_usage_samples, f, indent=4)
        print(f"Save token usage to {token_usage_path}.")
    
    def clean_messages(self):
        for agent in self.agents:
            agent.clean_messages()
        self.sum_agent.clean_messages()

