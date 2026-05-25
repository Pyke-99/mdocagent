from models.base_model import BaseModel
from mydatasets.base_dataset import BaseDataset
import os
from typing import Dict, Union
import json
import pandas as pd
from tqdm import tqdm
import re
import importlib

class Agent:
    def __init__(self, config, model=None):
        self.config = config
        self.messages = None
        if model is not None:
            self.model:BaseModel = model
        else:
            module = importlib.import_module(self.config.model.module_name)
            model_class = getattr(module, self.config.model.class_name)
            print("Create model: ", self.config.model.class_name)
            self.model = model_class(self.config.model)
    
    def clean_messages(self):
        self.messages = None
        
    def _predict(self, question, texts=None, images=None, add_to_message = False):
        if not self.config.agent.use_text:
            texts = None
        if not self.config.agent.use_image:
            images = None
        generated_ans, messages, token_usage = self.model.predict(question, texts, images, self.messages)
        if add_to_message:
            self.messages = messages
        return generated_ans, messages, token_usage
    
    def predict(self, question, texts=None, images=None, with_sys_prompt=True):
        if with_sys_prompt:
            question = self.config.agent.system_prompt + question
        return self._predict(question, texts, images, add_to_message = True)
    
    def self_reflect(self, prompt=None, add_to_message = True):
        if prompt is None:
            self_reflect_prompt = self.config.agent.self_reflect_prompt
        else:
            self_reflect_prompt = prompt
        
        generated_ans, messages, _ = self._predict(question = self_reflect_prompt)
        if add_to_message:
            self.messages = messages
        
        return generated_ans
    
    def eval(self, question, answer, gt):
        prompt = self.config.agent.eval_system_prompt.format(question=question, answer=answer, gt=gt)
        try:
            generated_ans, _, _ = self.model.predict(prompt)
            result = extract_evaluation_metrics(generated_ans)
            return result
        except Exception as e:
            print(f"Error evaluating answer: {str(e)}")
            return {"binary_correctness": 0}
    
    def eval_dataset(self, dataset: BaseDataset):
        samples, ans_path = dataset.load_latest_results()
        if self.config.truncate_len:
            samples = samples[:self.config.truncate_len]
        samples_with_answer = []
        for sample in tqdm(samples):
            try:
                question = sample[dataset.config.question_key]
                answer = sample[self.config.ans_key]
                gt = sample[dataset.config.gt_key]
                result = self.eval(question, answer, gt)
                sample['binary_correctness'] = result.get('binary_correctness', None)
                samples_with_answer.append(sample)
            except Exception as e:
                print(f"Error evaluating sample: {str(e)}")
                
        ans_file_path_name = ans_path[:-5]+"_results.json"
        with open(ans_file_path_name, "w") as file:
            json.dump(samples_with_answer, file, indent=4)
            
        samples_with_answer = pd.DataFrame(samples_with_answer)
        path = os.path.join(dataset.config.result_dir,"results.txt")
        with open(path, "a") as file:
            file.write("\nEvaluation Results Summary:\n")
            file.write(f"Result file: {ans_path}\n")
            overall_mean = samples_with_answer['binary_correctness'].mean()
            file.write(f"Average Binary Correctness: {overall_mean:.3f}\n")

            # Per-route evaluation if trace mode exists
            trace_key = self.config.ans_key + "_trace"
            if trace_key in samples_with_answer.columns:
                def mode_is(mode):
                    return samples_with_answer[trace_key].apply(lambda t: isinstance(t, dict) and t.get('mode') == mode)

                single_mask = mode_is('single_agent')
                multi_mask = mode_is('multi_agent')
            else:
                single_mask = pd.Series([False] * len(samples_with_answer), index=samples_with_answer.index)
                multi_mask = pd.Series([False] * len(samples_with_answer), index=samples_with_answer.index)

            single_n = int(single_mask.sum())
            multi_n = int(multi_mask.sum())

            if single_n > 0:
                single_mean = samples_with_answer.loc[single_mask, 'binary_correctness'].mean()
                file.write(f"Single-agent Binary Correctness: {single_mean:.3f} (N={single_n})\n")
            else:
                file.write(f"Single-agent Binary Correctness: N=0\n")

            if multi_n > 0:
                multi_mean = samples_with_answer.loc[multi_mask, 'binary_correctness'].mean()
                file.write(f"Multi-agent Binary Correctness: {multi_mean:.3f} (N={multi_n})\n")
            else:
                file.write(f"Multi-agent Binary Correctness: N=0\n")

            # Also write a small json summary of route stats
            try:
                stats = {
                    'overall_mean': float(overall_mean),
                    'single_mean': float(single_mean) if single_n > 0 else None,
                    'single_n': single_n,
                    'multi_mean': float(multi_mean) if multi_n > 0 else None,
                    'multi_n': multi_n,
                }
                summary_path = ans_path[:-5] + "_eval_by_route.json"
                with open(summary_path, 'w') as sf:
                    json.dump(stats, sf, indent=4)
            except Exception:
                pass

        print(f"Save results to {path}.")

def extract_evaluation_metrics(eval_str: str) -> Dict[str, Union[float, int]]:
    if not isinstance(eval_str, str):
        return {'binary_correctness': 0}

    # Keep evaluation strict: only trust explicit JSON outputs.
    try:
        start_index = eval_str.find('{')
        end_index = eval_str.rfind('}') + 1
        if start_index != -1 and end_index > start_index:
            json_str = eval_str[start_index:end_index]
            metrics = json.loads(json_str)
            return {'binary_correctness': int(metrics.get('binary_correctness', 0))}
    except Exception:
        pass

    return {'binary_correctness': 0}