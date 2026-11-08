"""把本地脚本快照写入Corrector Notebook；不执行单元格、不清除本地实验输出。

本地.py修改后先运行本命令，再在远程内核执行Notebook首个同步格。
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "WORKBENCH_SOURCE": "scripts/smolvla_training_workbench.py",
    "DATA_SOURCE": "scripts/corrector_data_workbench.py",
    "SIM_SOURCE": "scripts/smolvla_libero_tools.py",
    "BASELINE_SOURCE": "scripts/smolvla_baseline_benchmark.py",
    "BATCH_SOURCE": "studies/policy_generalization/corrector/batch_evaluation.py",
    "INVENTORY_SOURCE": "scripts/inspect_robot_datasets.py",
    "TRAIN_SOURCE": "scripts/corrector_train.py",
}


def sync(notebook, root=ROOT):
    value = json.loads(notebook.read_text())
    cell = next(c for c in value["cells"] if c.get("id") == "corrector-data-1")
    source = "".join(cell["source"])
    replacements, found = [], set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in SOURCES:
                text = (root / SOURCES[name]).read_text()
                ast.parse(text)
                replacements.append((node.lineno - 1, node.end_lineno, f"{name} = {text!r}\n"))
                found.add(name)
    if found != set(SOURCES):
        raise ValueError(f"Notebook模板缺少源码字段：{set(SOURCES) - found}")
    lines = source.splitlines(keepends=True)
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    cell["source"] = "".join(lines).splitlines(keepends=True)
    temporary = notebook.with_name(notebook.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")
    temporary.replace(notebook)
    for name, path in SOURCES.items():
        print(path, hashlib.sha256((root / path).read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=ROOT / "scripts/corrector_data_prepare.ipynb")
    sync(parser.parse_args().notebook)
