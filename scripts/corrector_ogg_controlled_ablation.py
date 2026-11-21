"""Run the bounded, paired MLP-OGG strength, cadence and loss ablation."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch


ARMS = {
    "cosine_base": dict(mse_weight=0., eta=1., every=3),
    "cosine_eta2": dict(mse_weight=0., eta=2., every=3),
    "cosine_every1": dict(mse_weight=0., eta=1., every=1),
    "mixed_mse003": dict(mse_weight=.03, eta=1., every=3),
}
MLP_SHA = "05d55ba9452142be66840b3bef6fad10226480b6115be4bbbcf0842116732fad"
POLICY_SHA = "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8"
ENGINE_SHA = "86860f248b46fc3917fd3ee07aa4bd0be5bbb0c0e0701ea88a2ea9d482064d70"
DOMAINS = ("ordinary", "plus")
SIGNATURES = ("state_sha256", "input_sha256", "noise_sha256")
TRIAL_FIELDS = ("success", "env_steps", "chunk_calls", "event_replans", "guidance_calls")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def loss_terms(prediction, target, visual_mismatch):
    return (prediction - target).square().mean(), visual_mismatch(prediction, target).mean()


def mixed_velocity(predictor, visual, x_tau, velocity, timestep, target, eta, weight, visual_mismatch):
    import torch

    if predictor.training or any(p.requires_grad for p in predictor.parameters()):
        raise ValueError("frozen eval predictor required")
    with torch.enable_grad():
        v = velocity.detach().clone().requires_grad_(True)
        action = (x_tau.detach() - timestep[:, None, None] * v)[:, 0, :predictor.cfg.action_dim]
        prediction = predictor(visual.detach(), action)
        mse, cosine = loss_terms(prediction, target.detach(), visual_mismatch)
        loss = weight * mse + cosine
        gradient, = torch.autograd.grad(loss, v)
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise FloatingPointError("nonfinite OGG loss or gradient")
        return (v - eta * gradient).detach(), loss.item()


def load_models(engine, config, domain):
    runtime = Path(config["runtime"])
    core = Path(config["core"])
    engine.obj.verify_core(core)
    sys.path[:0] = [str(core / "scripts"), str(core), str(Path(config["expanded"]) / "expanded_helpers")]
    os.environ.update(HF_HOME=str(runtime / "hf"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      LEROBOT_SRC=str(runtime / "lerobot/src"), MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    import torch
    import smolvla_libero_tools as sim
    import smolvla_training_workbench as wb

    torch.set_num_threads(4)
    if domain == "ordinary":
        sim.configure(SimpleNamespace(workdir=runtime, sim_fps=20, max_steps=280), wb)
    else:
        plus = Path(config["plus_source"])
        os.environ["LIBERO_CONFIG_PATH"] = config["plus_config"]
        sys.path.insert(0, str(plus))
        from libero import libero
        if not Path(libero.__file__).resolve().is_relative_to(plus.resolve()):
            raise ValueError("wrong LIBERO-Plus import")
        if subprocess.check_output(["git", "-C", plus, "rev-parse", "HEAD"], text=True).strip() != engine.PLUS_SHA:
            raise ValueError("wrong LIBERO-Plus revision")

    from corrector_data_workbench import cached_policy_snapshot
    from corrector_hybrid import load_frozen
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint = Path(config["checkpoint"])
    predictor, protocol, weights = load_frozen(checkpoint, Path(config["factory"]),
                                               runtime / "vla-corrector-reference", "cuda", MLP_SHA)
    required = dict(model_type="mlp", h_window=1, anchor_history=3, fps=10, k_step=10, prediction_seconds=1)
    if any(protocol.get(k) != v for k, v in required.items()) or predictor.cfg.action_dim != 7:
        raise ValueError("unexpected frozen MLP protocol")
    snapshot = Path(cached_policy_snapshot(False))
    if digest(snapshot / "model.safetensors") != POLICY_SHA:
        raise ValueError("unexpected SmolVLA checkpoint")
    cfg = SmolVLAConfig.from_pretrained(snapshot, local_files_only=True)
    if (cfg.chunk_size, cfg.num_steps, cfg.rtc_config) != (50, 10, None):
        raise ValueError("unexpected VLA sampler protocol")
    cfg.device, cfg.pretrained_path, cfg.load_vlm_weights, cfg.n_action_steps = "cuda", snapshot, False, 50
    policy = SmolVLAPolicy.from_pretrained(snapshot, config=cfg, local_files_only=True,
                                            strict=True).to("cuda").eval().requires_grad_(False)
    if next(policy.model.vlm_with_expert.vlm.parameters()).dtype != torch.float32:
        raise ValueError("historical FP32 visual encoder required")
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=snapshot,
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    # The existing recorder traverses this slot to check parameters. All cases use only the MLP.
    return SimpleNamespace(policy=policy, pre=pre, post=post, joint=torch.nn.Identity(),
                           legacy={"mlp": predictor}, protocols={"mlp": protocol}, weights={"mlp": weights},
                           hashes={"j": None})


def validate_trial(case, directory, result, historical, reference_path, active, visual_mismatch, predictor):
    import torch

    actions = read(directory / "actions.json")
    commands = actions["submitted_env_actions"]
    queue = read(directory / "queue_trace.json")
    signature = read(directory / "actual_protocol.json")["first"]
    first = active["first"]
    trace = [json.loads(row) for row in (directory / "monitor_steps.jsonl").read_text().splitlines()]
    guides = [row for row in trace if "guidance_loss" in row and "gradient_l2" in row]
    settings = ARMS[case["method"]]
    expected_slots = list(range(0, 10, settings["every"]))
    by_step = {}
    for row in guides:
        by_step.setdefault(row["step"], []).append(row["denoise"])
    if (len(guides) != result["guidance_calls"] or len(guides) != len(active["metrics"])
            or any(slots != expected_slots for slots in by_step.values()) or first is None
            or next(iter(by_step), None) != first["step"]):
        raise ValueError("OGG trigger/cadence trace mismatch")
    if any(not row["original_update_formula_exact_equal"] for row in guides):
        raise ValueError("OGG velocity update was not verified")
    if any(any(values[1:]) for row in guides for values in row["per_action_clean_update_l2"]):
        raise ValueError("OGG directly modified actions after the first slot")

    prefix = first["step"]
    current_reference = dict(signatures={k: signature[k] for k in SIGNATURES}, first_step=prefix,
        prefix_sha256=canonical_sha(commands[:prefix]), commands_sha256=canonical_sha(commands),
        queue_sha256=canonical_sha(queue), candidate_sha256=first["candidate_sha256"],
        first_submitted_action=commands[prefix], outcome={k: result[k] for k in TRIAL_FIELDS})
    historical_match = {k: signature[k] == historical["signatures"][k] for k in SIGNATURES}
    historical_match.update({k: current_reference[k] == historical[k]
        for k in ("first_step", "prefix_sha256", "candidate_sha256", "commands_sha256", "queue_sha256", "outcome")})
    same_host_baseline = case["method"] == "cosine_base"
    if same_host_baseline:
        if reference_path.exists() and read(reference_path) != current_reference:
            raise ValueError("existing same-host baseline did not reproduce")
        comparison = {"reference": "same_host_cosine_base"}
        reference = current_reference
    else:
        reference = read(reference_path)
        comparison = dict(initial_signatures_exact=current_reference["signatures"] == reference["signatures"],
                          pre_guidance_controls_exact=prefix == reference["first_step"] and
                          current_reference["prefix_sha256"] == reference["prefix_sha256"],
                          first_candidate_exact=current_reference["candidate_sha256"] == reference["candidate_sha256"])
        if not all(comparison.values()):
            raise ValueError(f"same-host pairing mismatch: {comparison}")

    # One diagnostic batch after rollout, outside the environment and chunk timings.
    started = time.perf_counter()
    with torch.inference_mode():
        current = torch.as_tensor(actions["normalized_policy_actions"][prefix], device=first["action"].device)
        action = torch.cat((first["action"], current.unsqueeze(0)))
        predictions = predictor(first["visual"].expand(2, -1, -1), action)
        before, after = (loss_terms(x.unsqueeze(0), first["target"], visual_mismatch) for x in predictions)
        terms = dict(before_mse=float(before[0]), before_cosine=float(before[1]),
                     final_action_mse=float(after[0]), final_action_cosine=float(after[1]),
                     final_vs_initial_candidate_l2=float((current-first["action"][0]).norm()),
                     submitted_vs_baseline_l2=sum((x-y)**2 for x,y in
                                                   zip(commands[prefix], reference["first_submitted_action"], strict=True))**.5)
    realized = dict(first_step=prefix, submitted_action=commands[prefix],
                    normalized_submitted_action=current.cpu().tolist(), diagnostic_s=time.perf_counter()-started, **terms)
    write(directory / "ogg_ablation.json", dict(comparison=comparison, historical_match=historical_match, first_event=realized,
          guidance=active["metrics"], baseline_submitted_action=reference["first_submitted_action"]))
    if same_host_baseline:
        write(reference_path, current_reference)
    return dict(result, ogg_ablation=comparison, historical_match=historical_match, first_event=realized,
                guidance_compute_s=sum(row["compute_s"] for row in active["metrics"]))


def evaluate(root, domain, deadline):
    config = read(root / "config.json")
    sys.path.insert(0, config["expanded"])
    import corrector_expanded_eval as engine
    if digest(Path(engine.__file__)) != ENGINE_SHA:
        raise ValueError("unexpected expanded evaluation engine")
    holder = load_models(engine, config, domain)
    import torch
    import corrector_online_runtime as runtime
    from studies.policy_generalization.corrector import batch_evaluation as batch

    golden = read(root / "reference.json")[domain]
    conditions = golden["conditions"]
    actual = engine.ordinary_conditions() if domain == "ordinary" else read(Path(config["expanded"]) / "plus_conditions.json")["conditions"]
    if len(conditions) != 4 or any(actual[k] != v for k, v in conditions.items()):
        raise ValueError("historical diagnostic condition definition changed")
    pairs = [(condition, seed) for condition in conditions for seed in (1000, 1001)]
    wanted = [(condition, seed, method) for method in ARMS for condition, seed in pairs]
    original_runtime, reference_guide = runtime.OnlineCorrector, runtime.guided_velocity
    make_plan, run_plan = batch.make_plan, batch.run_plan
    predictor = holder.legacy["mlp"]
    original_config = engine.registry()["mlp_ogg"]
    registry = {m: {**original_config, **settings, "cosine_weight": 1.} for m, settings in ARMS.items()}
    active = {}
    baseline_dir = root / (domain + "_same_host_baselines")
    baseline_dir.mkdir(exist_ok=True)

    class ConfiguredRuntime(original_runtime):
        def __init__(self, *args, **kwargs):
            settings = ARMS[active["case"]["method"]]
            kwargs.update(eta=settings["eta"], guidance_every=settings["every"])
            super().__init__(*args, **kwargs)
            if (self.method, self.k, self.warmup_steps, self.stride) != ("ogg", 20, 4, 1):
                raise ValueError("historical monitoring protocol changed")
            active["runtime"] = self

    def guided(model, visual, x_tau, velocity, timestep, target, eta=1.):
        settings = ARMS[active["case"]["method"]]
        observed = []
        def capture(_model, inputs, output):
            observed.append((inputs[1].detach(), output.detach()))

        hook = model.register_forward_hook(capture)
        started = time.perf_counter()
        try:
            if settings["mse_weight"]:
                out, loss = mixed_velocity(model, visual, x_tau, velocity, timestep, target, eta,
                                           settings["mse_weight"], runtime.visual_mismatch)
            else:
                out, loss = reference_guide(model, visual, x_tau, velocity, timestep, target, eta)
        finally:
            hook.remove()
        compute_s = time.perf_counter() - started
        (action, prediction), = observed
        controller = active["runtime"]
        with torch.no_grad():
            mse, cosine = loss_terms(prediction, target, runtime.visual_mismatch)
            active["metrics"].append(dict(step=controller.step, denoise=controller.denoise_index,
                                          mse=float(mse), cosine_loss=float(cosine),
                                          weighted_loss=loss, compute_s=compute_s))
            if active["first"] is None:
                candidate = (x_tau - timestep[:, None, None] * velocity)[:, :, :7]
                active["first"] = dict(step=controller.step, action=action, visual=visual.detach(),
                                       target=target.detach(), candidate_sha256=hashlib.sha256(
                                           candidate.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest())
        return out, loss

    def protocol_plan(cs, seeds, methods, protocol):
        for field in ("joint_visual_encoder_dtype", "joint_checkpoint_sha256",
                      "joint_layer_calibration_sha256", "full_target_tensors_per_joint_trial"):
            protocol.pop(field, None)
        protocol.update(study_id=f"ogg_mlp_controlled_20261121_{domain}",
                        experiment_label=f"MLP OGG strength, cadence and loss · {domain}",
                        original_cosine=ARMS["cosine_base"], comparison_arms=ARMS,
                        pairing="strict same-host original cosine reference; archived cross-host comparisons are descriptive",
                        adapter_sha256=digest(Path(__file__)),
                        golden_reference_sha256=digest(root / "reference.json"))
        plan = make_plan(cs, seeds, methods, protocol)
        rank = {v: i for i, v in enumerate(wanted)}
        plan["cases"].sort(key=lambda c: rank[c["condition"], c["seed"], c["method"]])
        return plan

    def audited_plan(output, plan, callbacks, **kwargs):
        def wrap(callback):
            def trial(case, directory):
                active.update(case=case, metrics=[], first=None)
                try:
                    result = callback(case, directory)
                    reference = golden["trials"][case["condition"]][str(case["seed"])]
                    key = canonical_sha([case["condition"], case["seed"]])
                    return validate_trial(case, directory, result, reference, baseline_dir / (key + ".json"),
                                          active, runtime.visual_mismatch, predictor)
                finally:
                    active.clear()
            return trial
        return run_plan(output, plan, {m: wrap(cb) for m, cb in callbacks.items()}, **kwargs)

    args = SimpleNamespace(runtime=Path(config["runtime"]), core=Path(config["core"]),
                           calibration=Path(config["calibration"]), output=root, mode=domain, deadline=deadline)
    with ExitStack() as stack:
        for obj, name, value in ((runtime, "OnlineCorrector", ConfiguredRuntime), (runtime, "guided_velocity", guided),
                                 (engine, "registry", lambda: registry), (engine, "cases", lambda *_: wanted),
                                 (batch, "make_plan", protocol_plan), (batch, "run_plan", audited_plan)):
            stack.enter_context(patch.object(obj, name, value))
        engine.run(args, domain, holder, conditions)


def final_summary(root):
    result = {}
    for domain in DOMAINS:
        trials = [read(p) for p in sorted((root / domain / "trials").glob("*.json"))]
        groups = {m: {} for m in ARMS}
        for t in trials:
            if t["status"] != "completed":
                raise ValueError("incomplete trial during summary")
            c = t["case"]
            group = groups[c["method"]]
            key = c["condition"], c["seed"]
            if key in group:
                raise ValueError("duplicate ablation condition")
            group[key] = t
        base = groups["cosine_base"]
        if len(base) != 8 or any(set(group) != set(base) for group in groups.values()):
            raise ValueError("missing or duplicate ablation conditions")
        summary = {}
        for method, group in groups.items():
            rows = [t["result"] for t in group.values()]
            diffs = [dict(condition=k[0], seed=k[1], success=t["result"]["success"],
                          baseline=base[k]["result"]["success"]) for k, t in group.items()
                     if t["result"]["success"] != base[k]["result"]["success"]]
            summary[method] = dict(completed=len(group), successes=sum(r["success"] for r in rows),
                                   gained=sum(d["success"] for d in diffs), lost=sum(d["baseline"] for d in diffs),
                                   flips=diffs, **{"mean_"+k: statistics.mean(r[k] for r in rows) for k in (
                                   "episode_wall_s", "policy_s", "monitor_s", "event_replans", "guidance_calls",
                                   "guidance_compute_s", "peak_torch_allocated_gib")})
        result[domain] = dict(methods=summary, historical_baseline_agreement={
            key: sum(t["result"]["historical_match"][key] for t in base.values())
            for key in next(iter(base.values()))["result"]["historical_match"]})
    write(root / "summary.json", dict(status="completed", completed=64, domains=result))


def status(root):
    result = read(root / "status.json") if (root / "status.json").exists() else {}
    result["domains"] = {d: read(path) for d in DOMAINS if (path := root / d / "execution.json").exists()}
    return result


def run_queue(root):
    with (root / "worker.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop = root / "STOP"
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stop.touch())
        # Only launch the predeclared 64 cases. Hold one GPU process at a time.
        state = {"status": "running", "phase": "preflight", "pid": os.getpid()}
        deadline = time.time() + 3*3600
        try:
            for domain in DOMAINS:
                current = status(root)["domains"].get(domain, {})
                if (current.get("status"), current.get("completed"), current.get("errors")) == ("completed", 32, 0):
                    continue
                if stop.exists():
                    state.update(status="paused")
                    return
                gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                               "--format=csv,noheader"], text=True)
                if gpu.strip():
                    raise RuntimeError("GPU occupied")
                log = root / f"{domain}_{time.time_ns()}.log"
                with log.open("x") as output:
                    process = subprocess.Popen([sys.executable, "-u", __file__, domain, str(root),
                                                "--deadline", str(deadline)],
                                               stdout=output, stderr=subprocess.STDOUT)
                    state.update(phase=domain, child_pid=process.pid, log=str(log))
                    write(root / "status.json", state)
                    code = process.wait()
                state["child_pid"] = None
                if stop.exists() or status(root)["domains"].get(domain, {}).get("status") == "paused":
                    state.update(status="paused")
                    return
                if code:
                    raise RuntimeError(f"{domain} exited {code}, see {log}")
            final_summary(root)
            state.update(status="completed", phase="completed")
        except BaseException as exc:
            state.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write(root / "status.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("status", "queue", *DOMAINS))
    parser.add_argument("root", type=Path)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    if args.mode == "queue":
        run_queue(args.root)
    elif args.mode == "status":
        print(json.dumps(status(args.root), ensure_ascii=False, indent=2))
    else:
        evaluate(args.root, args.mode, args.deadline)
