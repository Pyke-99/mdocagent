import os
import sys

print("Before override:", os.environ.get("CUDA_VISIBLE_DEVICES"))

from hydra import compose, initialize

with initialize(version_base="1.2", config_path="config"):
    cfg = compose(config_name="ptab")
    print("Config mdoc_agent.cuda_visible_devices:", cfg.mdoc_agent.cuda_visible_devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.mdoc_agent.cuda_visible_devices
    print("After override:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    
    import torch
    print("Torch Device Count:", torch.cuda.device_count())
    print("Torch CUDA Available:", torch.cuda.is_available())
