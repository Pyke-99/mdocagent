#!/usr/bin/env python3
"""
Convert tactic JSON files in tmp/TacticQA/ to searchable txt files.
Each JSON becomes a txt file named {Tactic_ID}_0.txt
"""
import json
import os
from pathlib import Path


def json_to_txt(json_path: str) -> str:
    """Convert a single tactic JSON to searchable text."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    parts = []
    
    # Add basic info
    tactic_name = data.get('Tactic_Name', '')
    if tactic_name:
        parts.append(f"战术名称: {tactic_name}")
    
    tactic_id = data.get('Tactic_ID', '')
    if tactic_id:
        parts.append(f"战术ID: {tactic_id}")
    
    mission_phase = data.get('Mission_Phase', '')
    if mission_phase:
        parts.append(f"任务阶段: {mission_phase}")
    
    tactic_type = data.get('Tactic_Type', '')
    if tactic_type:
        parts.append(f"战术类型: {tactic_type}")
    
    objective = data.get('Objective', '')
    if objective:
        parts.append(f"任务目标: {objective}")
    
    description = data.get('Description', '')
    if description:
        parts.append(f"战术描述: {description}")
    
    env = data.get('Applicable_Environment', '')
    if env:
        parts.append(f"适用环境: {env}")
    
    exec_time = data.get('Execution_Time')
    if exec_time is not None:
        parts.append(f"执行时间: {exec_time}秒")
    
    credibility = data.get('Credibility')
    if credibility is not None:
        parts.append(f"可信度: {credibility}")
    
    # Add tags
    tags = data.get('Semantic_Tags', [])
    if tags:
        tags_str = '，'.join(str(t) for t in tags)
        parts.append(f"语义标签: {tags_str}")
    
    # Add action sequence
    action_seq = data.get('Action_Sequence', [])
    if action_seq:
        parts.append("\n行动序列:")
        for step in action_seq:
            step_no = step.get('Step', '')
            intent = step.get('Intent', '')
            if step_no and intent:
                parts.append(f"  第{step_no}步: {intent}")
            visual_aids = step.get('Visual_Aids', [])
            if visual_aids:
                for aid in visual_aids:
                    parts.append(f"    视觉提示: {aid}")
    
    # Add overall visual aids
    overall = data.get('Visual_Aid_Overall', [])
    if overall:
        parts.append("\n整体视觉提示:")
        for aid in overall:
            parts.append(f"  {aid}")
    
    return '\n'.join(parts)


def convert_all(input_dir: str):
    """Convert all JSON files in directory to txt."""
    p = Path(input_dir)
    json_files = sorted(p.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    for json_file in json_files:
        try:
            data = json.load(open(json_file, 'r', encoding='utf-8'))
            tactic_id = data.get('Tactic_ID', json_file.stem)
            txt_name = f"{tactic_id}_0.txt"
            txt_path = p / txt_name
            
            txt_content = json_to_txt(str(json_file))
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(txt_content)
            
            print(f"✓ {json_file.name} → {txt_name}")
        except Exception as e:
            print(f"✗ {json_file.name}: {e}")


if __name__ == '__main__':
    convert_all('/root/MDocAgent/tmp/TacticQA')
    print("\n转换完成！")
