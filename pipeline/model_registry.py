import importlib
from typing import Dict


class ModelRegistry:
    # Process-wide shared cache to avoid loading the same model multiple times.
    _shared_models: Dict[str, object] = {}

    def __init__(self):
        self._models = ModelRegistry._shared_models

    def build_model_key(self, model_cfg):
        model_id = getattr(model_cfg, "model_id", getattr(model_cfg, "model", ""))
        module_name = getattr(model_cfg, "module_name", "")
        class_name = getattr(model_cfg, "class_name", "")
        return f"{module_name}:{class_name}:{model_id}"

    def get_or_create(self, model_key: str, model_cfg):
        if model_key in self._models:
            return self._models[model_key]

        module = importlib.import_module(model_cfg.module_name)
        model_class = getattr(module, model_cfg.class_name)
        self._models[model_key] = model_class(model_cfg)
        return self._models[model_key]
