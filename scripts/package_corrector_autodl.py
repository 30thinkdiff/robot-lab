"""将当前工作区实际代码打包；不依赖尚未push的GitHub版本，不包含凭据/模型/实验输出。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

from sync_corrector_notebook import ROOT, SOURCES


def build(output, root=ROOT):
    files = {}
    for key, relative in SOURCES.items():
        name = 'corrector_online_practice.py' if key == 'ONLINE_PRACTICE_SOURCE' else Path(relative).name
        text = (root / relative).read_text()
        if name.endswith('.json'):json.loads(text)
        else:ast.parse(text)
        if name in files:raise ValueError('部署文件名冲突：' + name)
        files[name] = text.encode()
    runner = 'corrector_autodl.py'
    files[runner] = (root / 'scripts' / runner).read_bytes()
    ast.parse(files[runner])
    files['START_HERE.txt'] = '''AutoDL部署包（没有任何账号、令牌或SSH配置）
1. 解压到/root/autodl-tmp/corrector/runtime；Python入口使用已有bootstrap环境。
2. prepare只装依赖和检查真实渲染，不需要权重，不训练、不提取特征。
   /root/autodl-tmp/corrector/bootstrap/bin/python corrector_autodl.py prepare --proxy-url http://127.0.0.1:17890
3. 从Drive的corrector_stream_task24_001/predictor/paper_width_001下载best_model.pt，
   上传为runtime/checkpoints/best_model.pt；SHA256应为
   19f73f07cc75a32500aaaeb6f059f30b11ce8cc5b3626eff84a33f4b8ca7492c
4. 在tmux中执行：
   /root/autodl-tmp/corrector/bootstrap/bin/python corrector_autodl.py night --proxy-url http://127.0.0.1:17890
   默认原始LIBERO object/task0，5真实初始状态×2种子×5方法=50正式回合。
   之前另有1个强制事件单回合，仅作OGG调用检查，不计正式成功率。
   当前建议先小规模：在night命令后加
   --initial-state-ids 0 1 --seeds 1000 --output results/pilot_001
   得到5方法×2初始化×1种子=10回合；检查视频/触发/耗时后再用新output扩大。
5. stdout和所有阶段完整日志在runtime/logs；结果、视频、分数在runtime/results/night_001。
   正式批次成功前自动检查smoke；默认HF离线，失败视频留存，成功率按真实完成回合统计。
6. 中断后，原机器/原代码/原参数重跑同一night命令；已完成回合验视频SHA后跳过。
   改方法/种子/初始状态/保留规则请用新--output；不要编辑已有冻结计划。
7. 只看结果不需要GPU：python corrector_autodl.py summary
8. 环境/实验出错时收集小日志包：python corrector_autodl.py diagnostics
   diagnostics_*.zip不含权重/视频/账号目录，可取回本地进行定位。
联网下载阶段依赖SSH反向代理时，保持本地隧道开启；tmux只保证任务不因SSH退出而结束，
不会让本地代理永久在线。night中通过smoke后，默认批次阶段不再请求HF网络。
如果127.0.0.1:17890报Connection refused，先在本地重建ssh -R反向代理并保持开启，
远端curl --proxy返回200后重跑prepare；不要删除半成品venv或误判成pip版本不存在。
本包是准备好的入口，尚未证明AutoDL上的真实完整实验已通过。
'''.encode()
    manifest = {'format':1,'scope':'current source snapshot, no model or credentials',
                'files':{name:hashlib.sha256(content).hexdigest() for name,content in files.items()}}
    files['bundle_manifest.json'] = (json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    output = Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_name(output.name+'.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():z.writestr(name,content)
    temporary.replace(output)
    print(output.resolve(),output.stat().st_size,'bytes')
    print('SHA256:',hashlib.sha256(output.read_bytes()).hexdigest())
    return manifest


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/autodl_setup/corrector_autodl.zip')
    build(p.parse_args().output)
