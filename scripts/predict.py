import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mydatasets.base_dataset import BaseDataset
from agents.mdoc_agent import MDocAgent
from pipeline.main_pipeline import MainPipeline
from pipeline.route_hybrid_pipeline import RouteHybridPipeline
from pipeline.single_agent_pipeline import SingleAgentPipeline
from pipeline.single_verify_pipeline import SingleVerifyPipeline
from pipeline.raav_pipeline import RouteAnalysisAnswerVerifyPipeline
import hydra

@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    # 根据架构模式动态分配显卡
    arch_mode = str(getattr(cfg.mdoc_agent, "architecture_mode", "structured")).lower()
    if arch_mode in ["single_agent"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 单智能体只用1张卡
        print(f"[Info] Running {arch_mode} mode. Defaulting to GPU 0.")
    elif arch_mode in ["single_verify", "single_with_verifier", "bounded_revision_single"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 带验证的单智能体使用2张卡
        print(f"[Info] Running {arch_mode} mode. Defaulting to GPUs 0,1.")
    else:
        # 其他架构保留 yaml 中的默认卡分配
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.mdoc_agent.cuda_visible_devices)
        
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    dataset = BaseDataset(cfg.dataset)
    resume_path = getattr(cfg.mdoc_agent, "resume_path", None)

    architecture_mode = str(
        getattr(cfg.mdoc_agent, "architecture_mode", getattr(cfg.mdoc_agent, "link_mode", "structured"))
    ).lower()
    if architecture_mode == "legacy_nl":
        architecture_mode = "classic"

    if architecture_mode == "classic":
        classic_model_overrides = getattr(cfg.mdoc_agent, "classic_models", {})
        for agent_config in cfg.mdoc_agent.agents:
            agent_name = agent_config.agent
            model_name = classic_model_overrides.get(agent_name, agent_config.model)
            agent_cfg = hydra.compose(config_name="agent/" + agent_name, overrides=[]).agent
            model_cfg = hydra.compose(config_name="model/" + model_name, overrides=[]).model
            agent_config.agent = agent_cfg
            agent_config.model = model_cfg

        sum_model_name = classic_model_overrides.get("sum_agent", cfg.mdoc_agent.sum_agent.model)
        cfg.mdoc_agent.sum_agent.agent = hydra.compose(
            config_name="agent/" + cfg.mdoc_agent.sum_agent.agent,
            overrides=[],
        ).agent
        cfg.mdoc_agent.sum_agent.model = hydra.compose(
            config_name="model/" + sum_model_name,
            overrides=[],
        ).model

        mdoc_agent = MDocAgent(cfg.mdoc_agent)
        mdoc_agent.predict_dataset(dataset, resume_path=resume_path)
        return

    if architecture_mode == "single_agent":
        single_agent_model = getattr(cfg.mdoc_agent, "single_agent_model", "qwen2vl")
        model_cfg = hydra.compose(config_name="model/" + single_agent_model, overrides=[]).model
        pipeline = SingleAgentPipeline(cfg.mdoc_agent, model_cfg)
        pipeline.predict_dataset(dataset, resume_path=resume_path)
        return

    if architecture_mode in {"single_verify", "single_with_verifier", "bounded_revision_single"}:
        single_agent_model = getattr(cfg.mdoc_agent, "single_agent_model", "qwen2vl")
        single_model_cfg = hydra.compose(config_name="model/" + single_agent_model, overrides=[]).model

        verify_cfg = getattr(cfg.mdoc_agent, "single_verify", None)
        verifier_model_name = (
            getattr(verify_cfg, "verifier_model", "qwen2vl")
            if verify_cfg is not None
            else "qwen2vl"
        )
        verifier_model_cfg = hydra.compose(config_name="model/" + verifier_model_name, overrides=[]).model

        pipeline = SingleVerifyPipeline(cfg.mdoc_agent, single_model_cfg, verifier_model_cfg)
        pipeline.predict_dataset(dataset, resume_path=resume_path)
        return

    if architecture_mode == "route_hybrid":
        route_cfg = getattr(cfg.mdoc_agent, "route_hybrid", None)
        gate_model_name = getattr(route_cfg, "gate_model", "qwen2vl") if route_cfg is not None else "qwen2vl"
        route_model_name = getattr(route_cfg, "route_model", gate_model_name) if route_cfg is not None else gate_model_name
        final_model_name = getattr(route_cfg, "final_model", getattr(cfg.mdoc_agent, "agent4_model", "qwen3vl")) if route_cfg is not None else getattr(cfg.mdoc_agent, "agent4_model", "qwen3vl")

        gate_model_cfg = hydra.compose(config_name="model/" + gate_model_name, overrides=[]).model
        route_model_cfg = hydra.compose(config_name="model/" + route_model_name, overrides=[]).model
        final_model_cfg = hydra.compose(config_name="model/" + final_model_name, overrides=[]).model

        pipeline = RouteHybridPipeline(cfg.mdoc_agent, gate_model_cfg, route_model_cfg, final_model_cfg)
        pipeline.predict_dataset(dataset, resume_path=resume_path)
        return

    if architecture_mode == "raav":
        raav_cfg = getattr(cfg.mdoc_agent, "raav", None)
        model_name = getattr(raav_cfg, "model", "qwen3vl") if raav_cfg is not None else "qwen3vl"
        model_cfg = hydra.compose(config_name="model/" + model_name, overrides=[]).model

        pipeline = RouteAnalysisAnswerVerifyPipeline(cfg.mdoc_agent, model_cfg)
        pipeline.predict_dataset(dataset, resume_path=resume_path)
        return

    if architecture_mode != "structured":
        raise ValueError(
            f"Unsupported architecture_mode: {architecture_mode}. Use 'structured', 'classic', 'single_agent', 'single_verify', 'route_hybrid', or 'raav'."
        )

    stage_model_cfgs = {}
    for stage_name, model_name in cfg.mdoc_agent.stage_models.items():
        stage_model_cfgs[stage_name] = hydra.compose(config_name="model/" + model_name, overrides=[]).model

    pipeline = MainPipeline(cfg.mdoc_agent, stage_model_cfgs)
    pipeline.predict_dataset(dataset, resume_path=resume_path)
    
if __name__ == "__main__":
    main()