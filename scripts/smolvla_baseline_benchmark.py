"""冻结公开SmolVLA的原始LIBERO基线；不依赖Corrector或五步LoRA练习。

顺序运行task×真实init_state_id×seed，一回合一视频/输入审计，断连按已完成回合续跑。
这是本项目公开checkpoint基线，不宣称复现Corrector论文的MetaWorld/SmolVLA数值。
"""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from studies.policy_generalization.corrector.batch_evaluation import make_plan, run_plan
except ModuleNotFoundError as error:
    if not error.name.startswith("studies"):
        raise
    from batch_evaluation import make_plan, run_plan  # Notebook将此模块同步到脚本同目录

POLICY_ID = "lerobot/smolvla_libero"
POLICY_SHA = "a24f1f380461b665158d52acd140152c79e8b36d"


def select_initial_state(env, index):
    """必须在reset前设置，禁止用%取模把越界请求悄悄变成已有状态。"""
    subenv = env.envs[0].unwrapped
    if type(index) is not int or not 0 <= index < len(subenv._init_states):
        raise ValueError("请求的真实初始状态编号越界")
    subenv.init_state_id = index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=["libero_object", "libero_spatial", "libero_goal", "libero_10"], default="libero_object")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--initial-state-ids", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000])
    parser.add_argument("--action-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.action_steps <= 50:
        raise ValueError("action-steps须为1..50")
    if args.fps <= 0 or (args.max_steps is not None and args.max_steps <= 0):
        raise ValueError("FPS与最大控制步数必须为正")
    import smolvla_training_workbench as wb
    import smolvla_libero_tools as sim
    args.workdir = args.workdir.absolute()
    sim_args = argparse.Namespace(workdir=args.workdir, sim_fps=args.fps, max_steps=args.max_steps or 280)
    sim.configure(sim_args, wb)  # 固定源码/配置/导入路径检查；此时还没有创建环境或加载VLA
    from libero.libero import benchmark
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, get_task_init_states
    suite = benchmark.get_benchmark_dict()[args.suite]()
    if len(args.task_ids) != len(set(args.task_ids)) or len(args.initial_state_ids) != len(set(args.initial_state_ids)):
        raise ValueError("task/init_state编号不能重复")
    horizon = args.max_steps or TASK_SUITE_MAX_STEPS[args.suite]
    import torch
    from importlib.metadata import version
    runtime = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               **{name: version(name) for name in ["lerobot", "transformers", "mujoco", "robosuite"]}}
    conditions = {}
    for task_id in args.task_ids:
        if not 0 <= task_id < len(suite.tasks):
            raise ValueError(f"task_id越界：{task_id}")
        states = get_task_init_states(suite, task_id, is_libero_plus=False)
        for init_id in args.initial_state_ids:
            if not 0 <= init_id < len(states):
                raise ValueError(f"task{task_id}没有init_state_id={init_id}")
            name = f"{args.suite}/task_{task_id}/init_{init_id}"
            conditions[name] = {"task_id": task_id, "initial_state_id": init_id,
                                "language": suite.tasks[task_id].language}
    method = f"H{args.action_steps}"
    plan = make_plan(list(conditions), args.seeds,
                     {method: {"n_action_steps": args.action_steps, "uses_corrector": False}},
                     {"scope": "public SmolVLA / original LIBERO baseline, not a paper table reproduction",
                      "policy_id": POLICY_ID, "policy_revision": POLICY_SHA, "lerobot_revision": wb.LEROBOT_SHA,
                      "libero_revision": sim.LIBERO_SHA, "conditions": conditions,
                      "runtime": runtime,
                      "fps": args.fps, "horizon": horizon, "control_mode": "relative"})
    holder = {}

    def load_policy():
        import torch
        from huggingface_hub import snapshot_download
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        if not torch.cuda.is_available():
            raise RuntimeError("完整策略基线要求CUDA；CPU只做入口/场景与汇总检查")
        snapshot = snapshot_download(POLICY_ID, revision=POLICY_SHA, token=False)
        cfg = SmolVLAConfig.from_pretrained(snapshot)
        cfg.device, cfg.n_action_steps, cfg.pretrained_path = "cuda", args.action_steps, Path(snapshot)
        cfg.load_vlm_weights = False  # 完整policy权重随后加载，不重复下载原始VLM权重
        policy = SmolVLAPolicy.from_pretrained(snapshot, config=cfg).to("cuda").eval()
        policy.requires_grad_(False)
        policy.reset()
        if policy._queues["action"].maxlen != args.action_steps:
            raise RuntimeError("实际动作队列容量与配置不一致")
        pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=snapshot,
                                            preprocessor_overrides={"device_processor": {"device": "cuda"}})
        holder.update(policy=policy, pre=pre, post=post)

    def run_case(case, directory):
        import torch
        from unittest.mock import patch
        from lerobot.envs.configs import LiberoEnv
        from lerobot.scripts.lerobot_eval import eval_policy
        from lerobot.utils.random_utils import set_seed
        if not holder:
            load_policy()  # 全部回合已完成时，不重新加载模型
        policy, pre, post = holder["policy"], holder["pre"], holder["post"]
        chosen = conditions[case["condition"]]
        camera_keys = list(policy.config.image_features)
        if len(camera_keys) != 2:
            raise RuntimeError("本协议要求两路相机")
        camera_names = [k.removeprefix("observation.images.") for k in camera_keys]
        cfg = LiberoEnv(task=args.suite, task_ids=[chosen["task_id"]], fps=args.fps,
                        episode_length=horizon, observation_height=256, observation_width=256,
                        control_mode="relative", is_libero_plus=False,
                        camera_name_mapping={"agentview_image": camera_names[0], "robot0_eye_in_hand_image": camera_names[1]})
        env = sim.make_vector(cfg)
        env_pre, env_post = cfg.get_env_processors()
        steps, chunk_times = 0, []
        try:
            select_initial_state(env, chosen["initial_state_id"])
            set_seed(case["seed"])
            def observe(batch):
                nonlocal steps
                if steps == 0:
                    state = torch.as_tensor(env.envs[0].unwrapped._env.get_sim_state()).clone()
                    torch.save(state, directory / "initial_sim_state.pt")
                    wb.save_inspection_images(batch, directory / "input_images")
                value = pre(batch)
                if steps == 0:
                    torch.save({k: v.detach().cpu() for k, v in value.items() if isinstance(v, torch.Tensor)},
                               directory / "first_policy_input.pt")
                steps += 1
                return value
            original = policy._get_action_chunk
            def timed(*a, **kw):
                torch.cuda.synchronize(); start = time.perf_counter()
                out = original(*a, **kw)
                torch.cuda.synchronize(); chunk_times.append(time.perf_counter() - start)
                return out
            original_noise = policy.model.sample_noise
            def record_noise(*a, **kw):
                noise = original_noise(*a, **kw)
                if not (directory / "first_noise.pt").exists():
                    torch.save(noise.detach().cpu(), directory / "first_noise.pt")
                return noise
            with patch.object(policy, "_get_action_chunk", timed), patch.object(policy.model, "sample_noise", record_noise):
                info = eval_policy(env=env, policy=policy, env_preprocessor=env_pre, env_postprocessor=env_post,
                                   preprocessor=observe, postprocessor=post, n_episodes=1,
                                   max_episodes_rendered=1, videos_dir=directory / "videos", start_seed=case["seed"])
            videos = [sim.verify_video(p, args.fps) for p in info.get("video_paths", [])]
            if len(videos) != 1 or videos[0]["frames"] != steps:
                raise RuntimeError("视频帧数与控制步数不一致")
            wb.write_json(directory / "eval_info.json", info)
            wb.write_json(directory / "actual_protocol.json", {**chosen, "case": case,
                          "camera_keys": camera_keys, "chunk_size": policy.config.chunk_size,
                          "n_action_steps": policy.config.n_action_steps, "num_steps": policy.config.num_steps,
                          "video": videos[0], "chunk_wall_s": chunk_times})
            return {"success": bool(info["per_episode"][0]["success"]), "env_steps": steps,
                    "chunk_calls": len(chunk_times), "policy_s": sum(chunk_times), "monitor_s": 0.,
                    "event_replans": 0, "video_paths": info["video_paths"], "initial_state_id": chosen["initial_state_id"]}
        finally:
            env.close()
    print(f"冻结计划：{len(plan['cases'])}回合；任务{args.task_ids}，真实初始状态{args.initial_state_ids}，seed{args.seeds}", flush=True)
    rows = run_plan(args.output, plan, {method: run_case})
    print(rows)


if __name__ == "__main__":
    main()
