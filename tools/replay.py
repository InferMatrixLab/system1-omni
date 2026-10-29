"""Relocate the archived campaign scripts without changing workload/protocol logic."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--kind',choices=['public','shapes','parity'],required=True)
p.add_argument('--s1-repo',type=Path,required=True)
p.add_argument('--vllm-repo',type=Path,required=True)
p.add_argument('--model-root',type=Path,required=True)
p.add_argument('--baseline-bin',type=Path,required=True)
a,args=p.parse_known_args()
if args and args[0]=='--':args=args[1:]
root=Path(__file__).resolve().parent.parent
name={'public':'public_campaign.py','shapes':'shape_campaign.py','parity':'parity_probe.py'}[a.kind]
source=root/'tools/original'/name
text=source.read_text()
# Only absolute path prefixes change. Archived original sources remain intact.
relocations = [('/home/psv/jev-optimization/baseline/bin',a.baseline_bin),
                ('/home/psv/system1-omni',a.s1_repo),('/home/psv/vllm-jev',a.vllm_repo),
                ('/home/psv/jev-comparison',a.model_root),('/home/psv/jev-optimization',root)]
for i,(old,new) in enumerate(relocations):
 new=str(new.resolve())
 if any(c in new for c in "'\"\\\n\r"):raise ValueError('Use paths without quotes, backslashes or newlines')
 text=text.replace(old,f"__REPLAY_PATH_{i}__")
for i,(_,new) in enumerate(relocations):
 text=text.replace(f"__REPLAY_PATH_{i}__",str(new.resolve()))
sys.path.insert(0,str(source.parent))
sys.argv=[str(source),*args]
print(json.dumps({'archived_script':name,'s1_repo':str(a.s1_repo),'vllm_repo':str(a.vllm_repo),'model_root':str(a.model_root),'argv':args}),flush=True)
exec(compile(text,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
