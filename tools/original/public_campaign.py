"""Pinned Public231 campaign using the repository's schema-validating HTTP runner.
Run only inside a GPU reservation. All server children are cleaned in finally.
"""
import argparse,asyncio,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import socket
import httpx
P=argparse.ArgumentParser();P.add_argument('--backend',choices=['system1','vllm-jev','both'],required=True);P.add_argument('--bin-dir',type=Path,default=Path('/home/psv/jev-optimization/baseline/bin'));P.add_argument('--out',type=Path,required=True);P.add_argument('--concurrency',default='1,8,16,32');P.add_argument('--baseline-screen',type=Path);P.add_argument('--screen-only',action='store_true');P.add_argument('--exact-prefix',choices=['0','1'],default='0');P.add_argument('--vllm-prefix',choices=['0','1'],default='0');P.add_argument('--frozen-metadata',type=Path);P.add_argument('--native-graph',choices=['0','1'],default='0');A=P.parse_args()
R=Path('/home/psv/jev-optimization');S=Path('/home/psv/system1-omni');V=Path('/home/psv/vllm-jev');C=Path('/home/psv/jev-comparison');M=R/'workloads/public231.jsonl';PY=V/'.venv/bin/python';RUNNER=S/'benchmarks/bench.py'
A.out.mkdir(parents=True,exist_ok=False)
assert os.environ.get('CUDA_VISIBLE_DEVICES'), 'Use gpu run'
plan={'workload':str(M),'sha256':hashlib.sha256(M.read_bytes()).hexdigest(),'concurrencies':[int(x) for x in A.concurrency.split(',')],'rounds':3,'warmup':5,'precision':'BF16','prefix_policy':{'native_exact_request_local':A.exact_prefix,'vllm_engine_prefix':A.vllm_prefix,'historical_request_reuse':'none: native recomputes each request, vLLM endpoint generates fresh cache salt'},'graph_policy':('native optional graphs enabled; vLLM default compile/graph' if A.native_graph=='1' else 'native eager; vLLM default compile/graph'),'clock':'post_to_body_json_and_schema_validation','numerical_acceptance_vs_frozen_native':{'decision_flips':0,'probability_max_abs':.001,'score_max_abs':.001},'candidate_must_win':'All predefined cells: lower pooled P50/P95, higher QPS; no failed or skipped requests. Screen cannot accept a change.','retries':0,'screen_only':A.screen_only}
(A.out/'plan.json').write_text(json.dumps(plan,indent=2))
procs=[]
def stop_all():
 for p in reversed(procs):
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(25)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 procs.clear()
def spawn(cmd,env,log,cwd):
 with log.open('w') as f:p=subprocess.Popen(cmd,env=env,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 procs.append(p)
def run_cli(cmd,log):
 with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(f'Runner failed ({r.returncode}): {log}')
async def backend(name):
 out=A.out/name;out.mkdir();env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',RAYON_NUM_THREADS='4',TOKENIZERS_PARALLELISM='true')
 if name=='system1':
  env.update(OPEN_JEV_MODEL=str(C/'system1-9b'),OPEN_JEV_CUDA_LIB=str(A.bin_dir/'libqwen3_5_cuda.so'),OPEN_JEV_PORT='19901',OMNI_JEV_BIND='127.0.0.1:19902',OMNI_JEV_BACKEND_URL='http://127.0.0.1:19901',CUA_S1_GRAPH=A.native_graph,OPEN_JEV_PREFIX='0',OPEN_JEV_EXACT_PREFIX=A.exact_prefix)
  commands=[[str(A.bin_dir/'omni-open-jev-native')],[str(A.bin_dir/'omni-jev')]];cwd=S
 else:
  commands=[[str(V/'.venv/bin/vllm-jev'),'serve','ZefanCai/Open-Jev-9B','--workspace',str(C/'vllm-workspace'),'--host','127.0.0.1','--port','19902','--max-model-len','4096','--max-num-seqs','128','--max-num-batched-tokens','8192','--gpu-memory-utilization','0.45','--enable-prefix-caching' if A.vllm_prefix=='1' else '--no-enable-prefix-caching','--dtype','bfloat16']];cwd=V
 gpu_query=subprocess.check_output(['nvidia-smi','-i',env['CUDA_VISIBLE_DEVICES'],'--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap','--format=csv,noheader'],text=True).strip()
 hw=json.loads(subprocess.check_output([str(PY),'-c','import torch,json; p=torch.cuda.get_device_properties(0); print(json.dumps(dict(name=p.name,sm=p.major*10+p.minor,sm_count=p.multi_processor_count,memory=p.total_memory,cuda=torch.version.cuda)))'],text=True))
 revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=cwd,text=True).strip()
 metadata={'native_graph': A.native_graph if name=='system1' else None,'gpu':hw['name'],'gpu_ids':env['CUDA_VISIBLE_DEVICES'],'device_evidence':gpu_query,'compute_capability':'sm_'+str(hw['sm']),'driver':gpu_query.split(',')[3].strip(),'cuda':hw['cuda'],'precision':'BF16','model_revision':'ZefanCai/Open-Jev-9B@47e966881e489511c0c7f5633a9e1960a676a551','runtime_revision':revision,'exact_request_local_prefix':name=='system1' and A.exact_prefix=='1','vllm_engine_prefix':name=='vllm-jev' and A.vllm_prefix=='1','cache_policy':('request-local exact prefix; recomputed every request; no response cache' if name=='system1' and A.exact_prefix=='1' else 'engine prefix enabled; fresh per-request salt; no response cache' if name=='vllm-jev' and A.vllm_prefix=='1' else 'prefix disabled; no response cache'),'cuda_evidence':{'hardware':hw,'worker_log':'server-0.log'},'reservation':'gpu run, inherited CUDA_VISIBLE_DEVICES='+env['CUDA_VISIBLE_DEVICES'],'commands':commands,'patch':subprocess.check_output(['git','diff','HEAD'],cwd=cwd,text=True),'binary_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in A.bin_dir.iterdir() if p.is_file()} if name=='system1' else None}
 if name=='system1' and A.frozen_metadata:
  frozen=json.loads(A.frozen_metadata.read_text())
  assert metadata['native_graph']==frozen.get('native_graph','0')
  assert metadata['binary_hashes']==frozen['binary_hashes'],'frozen provenance must match every binary'
  assert metadata['exact_request_local_prefix']==frozen.get('exact_request_local_prefix',False)
  metadata['runtime_revision']=frozen['runtime_revision'];metadata['patch']=frozen['patch']
  metadata['frozen_provenance_source']=str(A.frozen_metadata)
 (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
 try:
  for port in (19902, *([19901] if name=='system1' else [])):
   with socket.socket() as probe:probe.bind(('127.0.0.1',port))
  for i,cmd in enumerate(commands):spawn(cmd,env,out/f'server-{i}.log',cwd)
  async with httpx.AsyncClient(timeout=2,trust_env=False) as client:
   for _ in range(900):
    if any(p.poll() is not None for p in procs):raise RuntimeError('server exited')
    try:
     r=await client.get('http://127.0.0.1:19902/health')
     if r.status_code==200:break
    except httpx.HTTPError:pass
    await asyncio.sleep(1)
   else:raise RuntimeError('readiness timeout')
  print(name,'ready',flush=True)
  def run(manifest,dest,c=1,extra=()):
   cmd=[str(PY),str(RUNNER),'run',str(manifest),'--endpoint','http://127.0.0.1:19902/v1/systemone','--model','open-jev' if name=='system1' else 'ZefanCai/Open-Jev-9B','--metadata',str(out/'metadata.json'),'--phase','measured','--concurrency',str(c),'--warmup','5','--output',str(out/dest),*extra]
   run_cli(cmd,out/(dest+'.log'))
   summary=json.loads((out/dest/'summary.json').read_text());print(name,dest,json.dumps(summary),flush=True)
  run(S/'benchmarks/smoke.jsonl','smoke')
  if name=='system1':
   extra=['--fast-screen']
   if A.baseline_screen:extra+=['--baseline-screen',str(A.baseline_screen)]
   run(M,'screen',extra=extra)
   s=json.loads((out/'screen/summary.json').read_text())
   if s.get('fast_screen_decision') in ['drop','inconclusive']:raise RuntimeError('Screen did not permit full campaign')
  if A.screen_only:return
  for c in plan['concurrencies']:
   for i in range(1,4):run(M,f'public-c{c}-r{i}',c)
 finally:stop_all()
async def main():
 for b in (['system1','vllm-jev'] if A.backend=='both' else [A.backend]):await backend(b)
try:asyncio.run(main())
finally:stop_all()
