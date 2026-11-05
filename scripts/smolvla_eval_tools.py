"""评测接入代码；GPU只导出预测，用户指标在CPU单独计算。"""
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import shutil


def load_metric(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"未找到用户指标文件：{path}")
    spec = importlib.util.spec_from_file_location("user_evaluation_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = module.masked_action_mae
    if not callable(function):
        raise TypeError("指标文件须定义masked_action_mae。")
    return function


def check_metric(function):
    """小张量只核对接口与padding，不需要模型、CUDA或网络。"""
    import torch
    cases = [
        (torch.tensor([[[1., 3.], [5., 7.]]]), torch.zeros(1, 2, dtype=torch.bool), 4.),
        (torch.tensor([[[1., 3.], [999., 999.]]]), torch.tensor([[False, True]]), 2.),
        (torch.tensor([[[1., 1.], [999., 999.]], [[3., 3.], [5., 5.]]]),
         torch.tensor([[False, True], [False, False]]), 3.),
    ]
    for pred, mask, expected in cases:
        result = function(pred, torch.zeros_like(pred), mask)
        if not isinstance(result, float) or not math.isfinite(result) or not math.isclose(result, expected, abs_tol=1e-6):
            raise AssertionError(f"指标结果应为float {expected}，实际为{result!r}。")
    bad_inputs = [
        (torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), torch.ones(1, 2, dtype=torch.bool)),
        (torch.zeros(1, 2, 2), torch.zeros(1, 3, 2), torch.zeros(1, 2, dtype=torch.bool)),
        (torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), torch.zeros(1, 2)),
    ]
    for args in bad_inputs:
        try:
            function(*args)
        except (TypeError, ValueError):
            continue
        raise AssertionError("全padding、形状不匹配或非bool mask必须明确拒绝。")


def check_metric_entry(args, log, wb):
    check_metric(load_metric(args.metric_file))
    log.say("masked_action_mae的CPU接口与padding检查通过；尚未评测模型。")


def score_entry(args, log, wb):
    import torch
    function = load_metric(args.metric_file)
    check_metric(function)
    if not args.predictions:
        raise ValueError("score-eval需要--predictions指向paired_predictions.pt。")
    source = Path(args.predictions)
    meta = json.loads((source.parent / "evaluation_manifest.json").read_text())
    if not meta.get("complete"):
        raise RuntimeError("预测导出未完成，保留现场，不能报告完整评测。")
    if wb.digest(source) != meta.get("prediction_sha256"):
        raise RuntimeError("预测文件的SHA256与导出记录不一致。")
    packed = torch.load(source, map_location="cpu", weights_only=True)
    target, mask = packed["target"], packed["is_pad"]
    if mask.dtype != torch.bool or target.ndim != 3 or mask.shape != target.shape[:2]:
        raise ValueError("预测文件的标签/mask格式不正确。")
    if len(meta["samples"]) != len(target) or not torch.isfinite(target).all():
        raise ValueError("样本记录/标签不完整或含非有限数。")
    result = {"scope": "normalized action MAE; not closed-loop task success",
              "prediction_sha256": wb.digest(source), "metric_sha256": wb.digest(args.metric_file),
              "samples": meta["samples"], "valid_timesteps": int((~mask).sum()), "per_sample": []}
    for key in ["base", "trained"]:
        pred = packed[key]
        if pred.shape != target.shape or not torch.isfinite(pred).all():
            raise ValueError("预测形状错误或存在非有限数。")
        result[key + "_mae"] = function(pred, target, mask)
    result["trained_minus_base_mae"] = result["trained_mae"] - result["base_mae"]
    result["prediction_change_mae"] = function(packed["trained"], packed["base"], mask)
    for i, sample in enumerate(meta["samples"]):
        result["per_sample"].append({**sample, **{key + "_mae": function(packed[key][i:i+1], target[i:i+1], mask[i:i+1]) for key in ["base", "trained"]}})
    destination = source.parent / ("score_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    wb.write_json(destination, result)
    shutil.copy2(args.metric_file, destination.with_suffix(".py"))
    log.say(result)
    log.say(f"CPU评测已保存：{destination}；负的trained_minus_base_mae表示本次误差降低。")


def predict_pair(policy, batch, noise):
    """同一份输入/初始噪声，比较冻结base与训练后adapter；不使用动作队列。"""
    import torch
    def copied():
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    with torch.inference_mode():
        policy.reset()
        with policy.disable_adapter():
            base = policy.predict_action_chunk(copied(), noise=noise.clone()).detach().cpu()
        policy.reset()
        trained = policy.predict_action_chunk(copied(), noise=noise.clone()).detach().cpu()
    return base, trained


def export_entry(args, log, wb):
    import torch
    from unittest.mock import patch
    from torch.utils.data import default_collate
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from transformers import AutoModelForImageTextToText
    from peft import PeftConfig
    import inspect

    if not Path(inspect.getfile(make_policy)).resolve().is_relative_to((args.workdir / "lerobot").resolve()):
        raise RuntimeError("评测导入了另一份LeRobot，请使用准备生成的解释器。")

    manifest = json.loads((args.workdir / "environment.json").read_text())
    if not manifest.get("assets_ready"):
        raise RuntimeError("先恢复准备好的模型/数据环境；本入口不下载资产或启动训练。")
    candidates = [args.workdir / "train_run/checkpoints/000005/pretrained_model"]
    if args.archive_dir:
        candidates.append(args.archive_dir / "checkpoints/000005/pretrained_model")
    checkpoint = Path(args.checkpoint) if args.checkpoint else next((p for p in candidates if p.is_dir()), None)
    if checkpoint is None or not (checkpoint / "adapter_config.json").is_file():
        raise FileNotFoundError("没有找到训练后的adapter checkpoint；用--checkpoint指定pretrained_model目录。")
    evidence_path = checkpoint.parent / "training_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    if not evidence.get("trainable") or any("lora_" not in key for key in evidence["trainable"]):
        raise RuntimeError("此对照要求只训练LoRA；检测到其他可训练参数，不能把禁用adapter当作训练前base。")
    adapter = PeftConfig.from_pretrained(checkpoint)
    if adapter.base_model_name_or_path != wb.BASE_ID or adapter.revision != wb.BASE_SHA or adapter.modules_to_save or adapter.bias != "none":
        raise RuntimeError("checkpoint的base版本或训练方式不符合本次冻结base对照。")
    if not args.eval_run or Path(args.eval_run).name != args.eval_run or args.eval_run in (".", ".."):
        raise ValueError("--eval-run必须是一个目录名。")
    output = (args.archive_dir or args.workdir) / "evaluations" / args.eval_run
    if output.exists():
        raise FileExistsError(f"保留已有评测：{output}；请更换--eval-run。")
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    if not cfg.use_peft:
        raise RuntimeError("checkpoint配置没有启用PEFT，不能执行LoRA启用/禁用对照。")
    cfg.pretrained_path, cfg.device = checkpoint, "cuda"
    episode = 825
    if episode in wb.TRAIN_EPISODES:
        raise RuntimeError("验证episode不能与训练episode重叠。")
    wb.validate_dataset_files(Path(manifest["dataset_root"]))
    ds_meta = LeRobotDatasetMetadata(wb.DATA_ID, root=manifest["dataset_root"], revision=wb.DATA_SHA)
    dataset = LeRobotDataset(wb.DATA_ID, root=manifest["dataset_root"], revision=wb.DATA_SHA, episodes=[episode],
                            delta_timestamps=resolve_delta_timestamps(cfg, ds_meta), video_backend="pyav")
    points = sorted({0, len(dataset) // 2, len(dataset) - 1})
    if len(dataset) != 153:
        raise RuntimeError("验证轨迹长度与固定数据版本不一致。")
    preprocessor, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=checkpoint,
                                              preprocessor_overrides={"device_processor": {"device": "cpu"}})
    cpu_batches = [preprocessor(default_collate([dataset[frame]])) for frame in points]
    for batch in cpu_batches:
        target, mask = batch["action"], batch["action_is_pad"]
        if (target.shape != (1, cfg.chunk_size, 7) or mask.shape != target.shape[:2]
                or mask.dtype != torch.bool or not (~mask).any() or not torch.isfinite(target).all()):
            raise RuntimeError("CPU预处理得到的标签或padding不符合协议；尚未加载GPU模型。")
    # 以下才开始使用GPU。所有文件/数据协议检查已完成，不需要用户指标已实现。
    if not torch.cuda.is_available():
        raise RuntimeError("导出预测需要GPU；check-metric/score-eval仅需CPU。")
    original_loader = AutoModelForImageTextToText.from_pretrained
    def load_vlm(*a, **kw):
        if "torch_dtype" in kw:
            kw.setdefault("dtype", kw.pop("torch_dtype"))
        kw.setdefault("token", False)
        return original_loader(*a, **kw)
    output.mkdir(parents=True)
    meta = {"complete": False, "scope": "paired offline prediction on heldout episode; no environment rollout",
            "checkpoint": str(checkpoint), "adapter_sha256": wb.digest(checkpoint / "adapter_model.safetensors"),
            "base_revision": wb.BASE_SHA, "dataset_revision": wb.DATA_SHA, "episode": episode,
            "planned_frames": points, "noise_seed": args.eval_noise_seed, "samples": [],
            "inference_config": {key: getattr(cfg, key) for key in
                                 ["chunk_size", "num_steps", "max_action_dim", "max_state_dim", "use_cache"]},
            "metric_required_for_export": False, "source_manifest": manifest}
    wb.write_json(output / "evaluation_manifest.json", meta)
    with patch.object(AutoModelForImageTextToText, "from_pretrained", staticmethod(load_vlm)):
        policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    policy.eval()
    policy.requires_grad_(False)
    buckets = {key: [] for key in ["base", "trained", "target", "is_pad"]}
    for frame, cpu_batch in zip(points, cpu_batches, strict=True):
        batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in cpu_batch.items()}
        target = batch["action"].detach().cpu()
        mask = batch["action_is_pad"].detach().cpu()
        noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim,
                            generator=torch.Generator().manual_seed(args.eval_noise_seed)).to("cuda")
        base, trained = predict_pair(policy, batch, noise)
        if base.shape != target.shape or trained.shape != target.shape or not all(torch.isfinite(v).all() for v in [base, trained, target]):
            raise RuntimeError("预测/标签形状不一致或出现非有限数，保留未完成输出。")
        if mask.dtype != torch.bool or mask.shape != target.shape[:2] or not (~mask).any():
            raise RuntimeError("动作padding mask不正确。")
        if frame == points[0]:
            repeated_base, repeated_trained = predict_pair(policy, batch, noise)
            torch.testing.assert_close(repeated_base, base, rtol=1e-3, atol=1e-4)
            torch.testing.assert_close(repeated_trained, trained, rtol=1e-3, atol=1e-4)
            meta["repeatability_checked"] = True
        for key, value in [("base", base), ("trained", trained), ("target", target), ("is_pad", mask)]:
            buckets[key].append(value)
        meta["samples"].append({"episode": episode, "frame": frame, "valid_timesteps": int((~mask).sum())})
        wb.save_inspection_images(batch, output / f"frame_{frame:03d}")
        temporary = output / "paired_predictions.pt.tmp"
        torch.save({key: torch.cat(value, dim=0) for key, value in buckets.items()}, temporary)
        temporary.replace(output / "paired_predictions.pt")
        wb.write_json(output / "evaluation_manifest.json", meta)
        log.say(meta["samples"][-1])
    meta["complete"] = True
    meta["prediction_sha256"] = wb.digest(output / "paired_predictions.pt")
    wb.write_json(output / "evaluation_manifest.json", meta)
    log.say(f"预测已保存：{output}。现在可释放GPU，masked_action_mae与score-eval可在本地CPU完成。")
