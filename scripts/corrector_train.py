"""在固定上游MLP/Dataset/损失之上，接入逐轨迹划分、训练记录和epoch边界续训。

不导入用户练习实现；不加载VLA或仿真。inspect只校验缓存；fit显式开始训练。
"""
from __future__ import annotations

import argparse
import ast
import csv
from contextlib import contextmanager
from dataclasses import asdict
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corrector_data_workbench as archive


def student_components(model_cls, cfg, device, lr, weight_decay, epochs):
    """任务A：创建模型、AdamW和CosineAnnealingLR，返回(model, optimizer, scheduler)。

    模型由上游model_cls(cfg)创建并to(device)；优化器只接收其parameters()。
    调度器T_max=epochs，eta_min=0.01*lr。不要在这里加载VLA。
    """
    raise NotImplementedError("先完成任务A：模型、优化器和调度器")


def student_loaders(groups, batch_size, generator):
    """任务B：返回(train_loader, val_loader)，调用DataLoader。

    groups的train/validation已经按轨迹分好；禁止重新random_split。
    训练shuffle=True并传generator；验证shuffle=False；两者num_workers=0。
    """
    raise NotImplementedError("先完成任务B：训练与验证DataLoader")


def student_forward(model, batch, loss_fn, loss_type, beta):
    """任务C：双输入前向并调用上游loss_fn，返回(pred, total, mse, cosine_loss)。

    batch已在目标device，键为z_t、a_t、delta_z。
    loss_fn接收(pred,target,loss_type,beta)；不要item/detach，训练还要反向。
    验证也调用本方法；train/eval模式与no_grad由外围管理。
    """
    raise NotImplementedError("先完成任务C：前向与监督目标")


def student_update(model, optimizer, scaled_loss, should_step):
    """任务D：反向与有条件更新，返回本次更新前grad_norm；不更新时返回None。

    每次都对scaled_loss.backward()。外围已按累积组实际样本数缩放loss。
    只有should_step为True时：clip_grad_norm_(...,1.,error_if_nonfinite=True)，
    optimizer.step()，optimizer.zero_grad(set_to_none=True)，返回grad_norm。
    不更新时保留已累积的梯度。初始清梯度由外围完成。
    """
    raise NotImplementedError("先完成任务D：反向、裁剪、更新和清梯度")


def reference_components(model_cls, cfg, device, lr, weight_decay, epochs):
    """已验证的对照实现；仅--implementation reference显式启用。"""
    model = model_cls(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=.01 * lr)
    return model, optimizer, scheduler


def reference_loaders(groups, batch_size, generator):
    return (DataLoader(groups["train"], batch_size=batch_size, shuffle=True, generator=generator, num_workers=0),
            DataLoader(groups["validation"], batch_size=batch_size, shuffle=False, num_workers=0))


def reference_forward(model, batch, loss_fn, loss_type, beta):
    pred = model(batch["z_t"], batch["a_t"])
    total, mse, cos = loss_fn(pred, batch["delta_z"], loss_type, beta)
    return pred, total, mse, cos


def reference_update(model, optimizer, scaled_loss, should_step):
    scaled_loss.backward()
    if should_step:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return grad_norm
    return None


def training_functions(implementation):
    if implementation not in {"student", "reference"}:
        raise ValueError("implementation必须为student或reference")
    if implementation == "student":
        tasks = (student_components, student_loaders, student_forward, student_update)
        for task in tasks:
            tree = ast.parse(inspect.getsource(task))
            if any(isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call)
                   and isinstance(n.exc.func, ast.Name) and n.exc.func.id == "NotImplementedError"
                   for n in ast.walk(tree)):
                raise NotImplementedError(f"{task.__name__}尚未完成；inspect仍可运行。不会自动换用对照实现。")
        return tasks
    return reference_components, reference_loaders, reference_forward, reference_update


def reference_api(source_root):
    """加载固定、干净上游。只编译原训练文件中的两个纯损失函数，避免W&B副作用。"""
    archive.ensure_source(source_root)
    package = source_root.resolve() / "src/siglip_dynamics"
    # 临时追加，加载后立即恢复；不能让参考仓库内旧LeRobot覆盖已安装的0.6.1。
    sys.path.append(str(package.parent))
    try:
        config = importlib.import_module("siglip_dynamics.config")
        dataset = importlib.import_module("siglip_dynamics.dataset")
        model = importlib.import_module("siglip_dynamics.MLP")
    finally:
        sys.path.remove(str(package.parent))
    for module in (config, dataset, model):
        if Path(module.__file__).resolve().parent != package:
            raise RuntimeError("进程已导入另一份siglip_dynamics，请使用新Python进程")
    # 原函数正文原样执行，不复制另一份数学实现，也不调用原train()的random_split。
    source = package / "train.py"
    nodes = [n for n in ast.parse(source.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in {"_cosine_loss", "_combined_loss"}]
    if len(nodes) != 2:
        raise RuntimeError("固定参考损失函数接口发生变化")
    namespace = {"torch": torch, "F": F, "LossType": config.LossType}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    hashes = {p.name: archive.digest(p) for p in
              [source, package / "MLP.py", package / "dataset.py", package / "config.py"]}
    return config, dataset.SiglipDynamicsDataset, model.SiglipResidualMLP, namespace["_combined_loss"], hashes


def datasets(root, local, dataset_cls, k_step):
    """校验原归档，再复制到本会话盘并调用上游Dataset；绝不跨轨迹随机拆样本。"""
    groups, episodes, records, dimensions = {"train": [], "validation": []}, set(), [], set()
    for source in sorted(root.glob("episode_*")):
        if not archive.verified(source):
            raise RuntimeError(f"缓存未完成：{source}")
        meta = json.loads((source / "metadata.json").read_text())
        if (meta.get("corrector_revision") != archive.CORRECTOR_SHA
                or meta.get("policy_revision") != archive.POLICY_SHA
                or meta.get("action_representation") != "checkpoint_processor_normalized"):
            raise ValueError("特征或动作协议不同，拒绝混合训练")
        split = meta.get("split")
        if split not in groups:
            raise ValueError(f"本入口只接受预先冻结的train/validation缓存：{split}")
        directory = source
        if local is not None:
            local.mkdir(parents=True, exist_ok=True)
            directory = local / source.name
            if directory.resolve() == source.resolve():
                raise ValueError("临时特征目录不能与云盘输入目录相同")
            if directory.exists():
                if (not archive.verified(directory)
                        or json.loads((directory / "COMPLETE.json").read_text())["files"] !=
                        json.loads((source / "COMPLETE.json").read_text())["files"]):
                    raise RuntimeError("临时缓存已存在但来源不同，请换local-cache目录")
            else:
                # 已发布目录包含COMPLETE，不能再次交给publish生成嵌套清单。
                partial = directory.with_name(directory.name + ".partial")
                partial.mkdir(parents=True, exist_ok=True)
                marker = json.loads((source / "COMPLETE.json").read_text())
                for name in marker["files"]:
                    shutil.copy2(source / name, partial / name)
                archive.write_json(partial / "COMPLETE.json", marker)
                if not archive.verified(partial):
                    raise RuntimeError("临时盘特征复制校验失败")
                partial.rename(directory)
        # 上游按行距配对；先确保实际frame连续、每目录只有一条轨迹。
        frames = np.load(directory / "frame_index.npy", mmap_mode="r")
        ep = np.load(directory / "episode_index.npy", mmap_mode="r")
        if (len(ep) != meta["num_frames"] or frames.shape != ep.shape or len(ep) <= k_step
                or len(np.unique(ep)) != 1 or not np.all(np.diff(frames) == 1)):
            raise ValueError("轨迹长度/连续性/episode编号不符合上游配对要求")
        episode = int(ep[0])
        if episode in episodes:
            raise ValueError("出现重复episode；禁止训练/验证泄漏")
        episodes.add(episode)
        shape = np.load(directory / "z_q.npy", mmap_mode="r").shape
        actions = np.load(directory / "actions.npy", mmap_mode="r")
        scales = np.load(directory / "z_scale.npy", mmap_mode="r")
        if (shape != (len(ep), meta["l_tokens"], meta["d_dim"])
                or actions.shape != (len(ep), meta["action_dim"])
                or scales.shape != shape[:2] or not np.isfinite(actions).all()
                or not np.isfinite(scales).all() or (scales <= 0).any()):
            raise ValueError("缓存维度/scale/动作有误")
        dimensions.add((shape[1], shape[2], actions.shape[1], float(meta["fps"])))
        ds = dataset_cls(str(directory), k_step=k_step, h_window=1)
        groups[split].append(ds)
        records.append({"episode": episode, "split": split, "frames": len(ep), "pairs": len(ds),
                        "files": json.loads((source / "COMPLETE.json").read_text())["files"]})
    if len(dimensions) != 1 or any(not group for group in groups.values()):
        raise ValueError("需要同一特征/时间协议下的训练和独立验证轨迹")
    return {k: ConcatDataset(v) for k, v in groups.items()}, records, dimensions.pop()


def atomic_save(path, payload):
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_history(output, history):
    archive.write_json(output / "history.json", history)
    temporary = output / "metrics.csv.partial"
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "step", "train_loss", "train_mse",
                                                 "val_loss", "val_mse", "val_cosine_loss", "zero_prediction_mse"])
        writer.writeheader()
        for item in history:
            writer.writerow({**{k: item[k] for k in ("epoch", "step", "train_loss", "train_mse")},
                             "val_loss": item["validation"]["loss"], "val_mse": item["validation"]["mse"],
                             "val_cosine_loss": item["validation"]["cosine_loss"],
                             "zero_prediction_mse": item["validation"]["zero_prediction_mse"]})
    temporary.replace(output / "metrics.csv")


def move(batch, device):
    # 不移动上游为兼容历史模型返回的z_hist；MLP只需这三个输入。
    return {key: batch[key].to(device) for key in ("z_t", "a_t", "delta_z")}


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, loss_type, beta, forward_fn=reference_forward):
    model.eval()
    sums = np.zeros(5, dtype=np.float64)
    count, zero_targets = 0, 0
    for batch in loader:
        batch = move(batch, device)
        pred, total, mse, cos = forward_fn(model, batch, loss_fn, loss_type, beta)
        target = batch["delta_z"]
        values = [total.item(), mse.item(), cos.item(), target.square().mean().item(),
                  pred.square().mean().sqrt().item()]
        if not np.isfinite(values).all():
            raise FloatingPointError("验证指标出现NaN/Inf")
        n = len(pred)
        sums += np.array(values) * n
        count += n
        zero_targets += int((target.reshape(n, -1).norm(dim=1) == 0).sum().item())
    values = dict(zip(("loss", "mse", "cosine_loss", "zero_prediction_mse", "mean_batch_pred_rms"),
                      (sums / count).tolist()))
    return {**values, "samples": count, "zero_targets": zero_targets}


@contextmanager
def log_to(path):
    class Tee:
        def __init__(self, stream, file):
            self.stream, self.file = stream, file
        def write(self, value):
            self.stream.write(value); self.file.write(value); self.file.flush()
            return len(value)
        def flush(self):
            self.stream.flush(); self.file.flush()
    with path.open("a") as file:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, file), Tee(stderr, file)
        try:
            yield
        except BaseException:
            import traceback
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr


def run(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求CUDA但不可用；不会静默回退CPU")
    if args.batch_size < 1 or args.accumulate < 1 or args.epochs < 1 or args.k_step < 1:
        raise ValueError("batch_size/accumulate/epochs/k_step必须为正整数")
    if args.lr <= 0 or args.weight_decay < 0 or args.patience < 1 or args.beta < 0:
        raise ValueError("优化器/早停/损失参数不合法")
    if args.cpu_threads < 1:
        raise ValueError("cpu_threads必须为正整数")
    torch.set_num_threads(args.cpu_threads)
    print(json.dumps({"python": sys.executable, "torch": torch.__version__, "numpy": np.__version__,
                      "device": args.device, "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None},
          ensure_ascii=False), flush=True)
    config, dataset_cls, model_cls, loss_fn, source_hashes = reference_api(args.source_root)
    groups, records, (tokens, dim, action_dim, fps) = datasets(
        args.features, args.local_cache if args.mode == "fit" else None, dataset_cls, args.k_step)
    widths = (256, 256) if args.profile == "pilot" else (2048, 2048, 2048, 2048)
    cfg = config.SiglipMLPConfig(token_dim=dim, action_dim=action_dim, action_embed_dim=256,
                                scale=config.ModelScale.CUSTOM, custom_widths=widths)
    model_cfg = json.loads(json.dumps(asdict(cfg)))
    protocol = {"source_sha": archive.CORRECTOR_SHA, "source_files": source_hashes, "records": records,
                "model": model_cfg, "h_window": 1, "k_step": args.k_step, "fps": fps,
                "prediction_seconds": args.k_step / fps, "loss_type": args.loss_type, "beta": args.beta,
                "lr": args.lr, "weight_decay": args.weight_decay, "epochs": args.epochs,
                "patience": args.patience, "seed": args.seed, "batch_size": args.batch_size,
                "accumulate": args.accumulate, "precision": "float32", "implementation": args.implementation,
                "training_entry_sha256": archive.digest(Path(__file__))}
    with torch.device("meta"):
        nparams = sum(p.numel() for p in model_cls(cfg).parameters())
    print(json.dumps({"frames": sum(r["frames"] for r in records),
                      "pairs": {k: len(v) for k, v in groups.items()}, "visual_shape": [tokens, dim],
                      "parameters": nparams, "checkpoint_payload_mib_approx": 16 * nparams / 2**20,
                      "protocol": protocol}, ensure_ascii=False, indent=2), flush=True)
    if args.mode == "inspect":
        return
    components_fn, loaders_fn, forward_fn, update_fn = training_functions(args.implementation)
    latest = args.output / "latest.pt"
    if (args.output / "protocol.json").exists():
        if json.loads((args.output / "protocol.json").read_text()) != protocol:
            raise RuntimeError("已有实验协议不同，不能更改后冒充续训；使用新output目录")
        if not args.resume or not latest.is_file():
            raise RuntimeError("已有实验，需显式--resume且有完整latest.pt；否则换output")
    elif args.resume:
        raise RuntimeError("找不到原训练协议，不能续训")
    else:
        archive.write_json(args.output / "protocol.json", protocol)
        archive.write_json(args.output / "config.json", {
            "model": model_cfg, "model_type": "mlp", "h_window": 1, "k_step": args.k_step,
            "data_fps": fps, "prediction_seconds": args.k_step / fps})
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    model, optimizer, scheduler = components_fn(model_cls, cfg, device, args.lr, args.weight_decay, args.epochs)
    loss_type = config.LossType(args.loss_type)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader, val_loader = loaders_fn(groups, args.batch_size, generator)
    start, step, best, stale, history, best_weights = 0, 0, float("inf"), 0, [], None
    if args.resume:
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["protocol"] != protocol:
            raise RuntimeError("checkpoint与协议文件不一致")
        model.load_state_dict(state["state_dict"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, step, best, stale, history = (state[k] for k in ("epoch", "step", "best", "stale", "history"))
        best_weights = state["best_state_dict"]
        torch.set_rng_state(state["rng"]); generator.set_state(state["loader_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"从完整epoch {start}恢复；中断的半个epoch会重做", flush=True)
    else:
        initial = evaluate(model, val_loader, device, loss_fn, loss_type, args.beta, forward_fn)
        archive.write_json(args.output / "initial_validation.json", initial)
    for epoch in range(start, args.epochs):
        if stale >= args.patience:
            break
        began = time.perf_counter()
        model.train(); optimizer.zero_grad(set_to_none=True)
        totals, count = np.zeros(3), 0
        for i, batch in enumerate(train_loader):
            batch = move(batch, device)
            pred, total, mse, cos = forward_fn(model, batch, loss_fn, loss_type, args.beta)
            if not torch.isfinite(total):
                raise FloatingPointError("训练loss出现NaN/Inf")
            n = len(pred)
            group_start = (i // args.accumulate) * args.accumulate * args.batch_size
            group_samples = min(args.batch_size * args.accumulate, len(groups["train"]) - group_start)
            should_step = (i + 1) % args.accumulate == 0 or i + 1 == len(train_loader)
            batch_grad_norm = update_fn(model, optimizer, total * n / group_samples, should_step)
            totals += np.array([total.item(), mse.item(), cos.item()]) * n
            count += n
            if should_step:
                grad_norm = batch_grad_norm
                step += 1
        val = evaluate(model, val_loader, device, loss_fn, loss_type, args.beta, forward_fn)
        scheduler.step()
        improved = val["loss"] < best
        best, stale = (val["loss"], 0) if improved else (best, stale + 1)
        if improved:
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        record = {"epoch": epoch + 1, "step": step, "train_loss": float(totals[0] / count),
                  "train_mse": float(totals[1] / count), "train_cosine_loss": float(totals[2] / count),
                  "validation": val, "grad_norm_last": float(grad_norm),
                  "seconds": time.perf_counter() - began, "next_lr": scheduler.get_last_lr()[0]}
        history.append(record)
        state = {"state_dict": model.state_dict(), "model_cfg": model_cfg,
                 "train_cfg": {"model": model_cfg, "model_type": "mlp", "h_window": 1, "k_step": args.k_step},
                 "protocol": protocol, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "epoch": epoch + 1, "step": step, "best": best, "stale": stale, "history": history,
                 "best_state_dict": best_weights,
                 "rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        atomic_save(latest, state)
        save_history(args.output, history)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.stop_after_epoch and epoch + 1 >= args.stop_after_epoch:
            break
    # 用新模型严格加载最佳权重并复查指标，不以“文件存在”冒充重载成功。
    save_history(args.output, history)  # 修复latest已完成、历史导出却中断的情况。
    # 最佳权重随latest一同提交；中断不会造成latest和单独best文件版本不一致。
    checkpoint = {"state_dict": best_weights, "model_cfg": model_cfg,
                  "train_cfg": {"model": model_cfg, "model_type": "mlp", "h_window": 1, "k_step": args.k_step,
                                "data_fps": fps, "prediction_seconds": args.k_step / fps},
                  "feature_protocol": protocol}
    atomic_save(args.output / "best_model.pt", checkpoint)
    checkpoint = torch.load(args.output / "best_model.pt", map_location=device, weights_only=True)
    reloaded = model_cls(cfg).to(device)
    reloaded.load_state_dict(checkpoint["state_dict"], strict=True)
    result = evaluate(reloaded, val_loader, device, loss_fn, loss_type, args.beta, forward_fn)
    if not math.isclose(result["loss"], best, rel_tol=1e-5, abs_tol=1e-6):
        raise RuntimeError("最佳权重重载指标与训练记录不一致")
    archive.write_json(args.output / "result.json", {"best_validation": result, "epochs_completed": len(history),
                       "optimizer_steps": step, "parameters": nparams,
                       "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
                       "checkpoint_reload_verified": True, "scope": "offline predictor; no robot success-rate claim"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["inspect", "fit"])
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-cache", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--profile", choices=["pilot", "paper-width"], default="pilot")
    parser.add_argument("--implementation", choices=["student", "reference"], default="student",
                        help="默认执行四个用户任务；reference是显式选择的已验证对照，不会自动回退")
    parser.add_argument("--k-step", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-type", choices=["mse", "cosine", "both"], default="cosine")
    parser.add_argument("--beta", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, default=0,
                        help="保留原总epochs和调度器，仅提前停在完整epoch，供断连恢复检查")
    args = parser.parse_args()
    if str(args.output).startswith("/content/drive/") and not os.path.ismount("/content/drive"):
        raise RuntimeError("Drive未挂载，拒绝误存临时目录")
    args.output.mkdir(parents=True, exist_ok=True)
    with log_to(args.output / "train.log"):
        run(args)


if __name__ == "__main__":
    main()
