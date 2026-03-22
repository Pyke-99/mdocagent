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
    
    def sum(self, sum_question):
        ans, all_messages = self.sum_agent.predict(sum_question)
        def extract_final_answer(agent_response):
            try:
                response_dict = json.loads(agent_response)
                answer = response_dict.get("Answer", None)
                return answer
            except:
                return agent_response
        final_ans = extract_final_answer(ans)
        return final_ans, all_messages

    def predict_dataset(self, dataset:BaseDataset, resume_path = None):
        samples = dataset.load_data(use_retreival=True)
        if resume_path:
            assert os.path.exists(resume_path)
            with open(resume_path, 'r') as f:
                samples = json.load(f)
        if self.config.truncate_len:
            samples = samples[:self.config.truncate_len]
            
        debug_max_samples = 100  # 设置为整数以只跑前 N 条；注释本行则跑全部样本

        if 'debug_max_samples' in locals() and isinstance(debug_max_samples, int):
            samples = samples[:debug_max_samples]
            
        sample_no = 0
        for sample in tqdm(samples):
            if resume_path and self.config.ans_key in sample:
                continue
            question, texts, images = dataset.load_sample_retrieval_data(sample)
            try:
                final_ans, final_messages = self.predict(question, texts, images)
            except RuntimeError as e:
                print(e)
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                final_ans, final_messages = None, None
            sample[self.config.ans_key] = final_ans
            if self.config.save_message:
                sample[self.config.ans_key+"_message"] = final_messages
            torch.cuda.empty_cache()
            self.clean_messages()
            
            sample_no += 1
            if sample_no % self.config.save_freq == 0:
                path = dataset.dump_reults(samples)
                print(f"Save {sample_no} results to {path}.")
        path = dataset.dump_reults(samples)
        print(f"Save final results to {path}.")

        # 如果有反思记录，则在对应 result_dir 下额外保存一个 JSON 文件
        try:
            if hasattr(self, "reflection_records") and self.reflection_records:
                os.makedirs(dataset.config.result_dir, exist_ok=True)
                reflection_path = os.path.join(
                    dataset.config.result_dir, dataset.time + "_reflection.json"
                )
                with open(reflection_path, "w") as f:
                    json.dump(self.reflection_records, f, indent=4, ensure_ascii=False)
                print(f"Save reflection results to {reflection_path}.")
        except Exception as e:
            print(f"Failed to save reflection results: {e}")
    
    def clean_messages(self):
        for agent in self.agents:
            agent.clean_messages()
        self.sum_agent.clean_messages()

