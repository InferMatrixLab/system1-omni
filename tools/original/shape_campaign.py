"""Pinned Public231 campaign using the repository's schema-validating HTTP runner.
Run only inside a GPU reservation. All server children are cleaned in finally.
"""
import argparse,asyncio,hashlib,json,os,signal,subprocess,sys,time,random,statistics,math,importlib.util
from pathlib import Path
import socket
import httpx
from duration_runner import collect
P=argparse.ArgumentParser();P.add_argument('--backend',choices=['system1','vllm-jev','both'],required=True);P.add_argument('--bin-dir',type=Path,default=Path('/home/psv/jev-optimization/baseline/bin'));P.add_argument('--out',type=Path,required=True);P.add_argument('--concurrency',default='1,8,16,32');P.add_argument('--baseline-screen',type=Path);P.add_argument('--screen-only',action='store_true');P.add_argument('--native-revision');P.add_argument('--rounds',type=int,default=2);P.add_argument('--exact-prefix',choices=['0','1'],default='1');P.add_argument('--vllm-prefix',choices=['0','1'],default='1');P.add_argument('--max-model-len',type=int,default=8192);P.add_argument('--native-model',type=Path,default=Path('/home/psv/jev-comparison/system1-9b-cap8192'));P.add_argument('--frozen-metadata',type=Path);P.add_argument('--native-graph',choices=['0','1'],default='0');A=P.parse_args()
assert A.backend=='vllm-jev' or A.native_revision,'record the frozen native binary revision'
if A.backend!='vllm-jev': assert json.loads((A.native_model/'open_jev_export.json').read_text())['max_length']==A.max_model_len
R=Path('/home/psv/jev-optimization');S=Path('/home/psv/system1-omni');V=Path('/home/psv/vllm-jev');C=Path('/home/psv/jev-comparison');M=R/'workloads/shapes.jsonl';PY=V/'.venv/bin/python';RUNNER=S/'benchmarks/bench.py'
A.out.mkdir(parents=True,exist_ok=False)
harness=Path(__file__).read_bytes();(A.out/'harness_at_launch.py').write_bytes(harness);harness_hash=hashlib.sha256(harness).hexdigest()
assert os.environ.get('CUDA_VISIBLE_DEVICES'), 'Use gpu run'
plan=json.loads((R/'workloads/shapes-plan.json').read_text())
plan.update(native_graph=A.native_graph,graph_policy=('native optional graphs enabled' if A.native_graph=='1' else 'native eager')+'; vLLM default compile/graph',workload=str(M),sha256=hashlib.sha256(M.read_bytes()).hexdigest(),rounds=A.rounds,native_revision=A.native_revision,exact_request_local_prefix=A.exact_prefix,vllm_engine_prefix=A.vllm_prefix,max_model_len=A.max_model_len,native_model=str(A.native_model),capacity_protocol='Both engines reserve 8192 context capacity; unchanged input manifest max4096, no truncation. VJ exact4096 capacity boundary hangs; independent diagnostic020 showed8192 resolves.')
spec=importlib.util.spec_from_file_location('official_bench',RUNNER);bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
cases=[json.loads(line) for line in M.read_text().splitlines()]
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
async def backend(name,round_no):
 out=A.out/f'{name}-r{round_no}';out.mkdir();env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',RAYON_NUM_THREADS='4',TOKENIZERS_PARALLELISM='true')
 if name=='system1':
  env.update(OPEN_JEV_MODEL=str(A.native_model),OPEN_JEV_CUDA_LIB=str(A.bin_dir/'libqwen3_5_cuda.so'),OPEN_JEV_PORT='21901',OMNI_JEV_BIND='127.0.0.1:21902',OMNI_JEV_BACKEND_URL='http://127.0.0.1:21901',CUA_S1_GRAPH=A.native_graph,OPEN_JEV_PREFIX='0',OPEN_JEV_EXACT_PREFIX=A.exact_prefix)
  commands=[[str(A.bin_dir/'omni-open-jev-native')],[str(A.bin_dir/'omni-jev')]];cwd=S
 else:
  commands=[[str(V/'.venv/bin/vllm-jev'),'serve','ZefanCai/Open-Jev-9B','--workspace',str(C/'vllm-workspace'),'--host','127.0.0.1','--port','21902','--max-model-len',str(A.max_model_len),'--max-num-seqs','128','--max-num-batched-tokens','8192','--gpu-memory-utilization','0.45','--enable-prefix-caching' if A.vllm_prefix=='1' else '--no-enable-prefix-caching','--dtype','bfloat16']];cwd=V
 gpu_query=subprocess.check_output(['nvidia-smi','-i',env['CUDA_VISIBLE_DEVICES'],'--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap','--format=csv,noheader'],text=True).strip()
 hw=json.loads(subprocess.check_output([str(PY),'-c','import torch,json; p=torch.cuda.get_device_properties(0); print(json.dumps(dict(name=p.name,sm=p.major*10+p.minor,sm_count=p.multi_processor_count,memory=p.total_memory,cuda=torch.version.cuda)))'],text=True))
 revision=A.native_revision if name=='system1' else subprocess.check_output(['git','rev-parse','HEAD'],cwd=cwd,text=True).strip()
 metadata={'native_graph': A.native_graph if name=='system1' else None,'gpu':hw['name'],'gpu_ids':env['CUDA_VISIBLE_DEVICES'],'device_evidence':gpu_query,'compute_capability':'sm_'+str(hw['sm']),'driver':gpu_query.split(',')[3].strip(),'cuda':hw['cuda'],'precision':'BF16','model_revision':'ZefanCai/Open-Jev-9B@47e966881e489511c0c7f5633a9e1960a676a551','runtime_revision':revision,'exact_request_local_prefix':name=='system1' and A.exact_prefix=='1','vllm_engine_prefix':name=='vllm-jev' and A.vllm_prefix=='1','cache_policy':'per-request prefix reuse configured explicitly; no historical request or answer reuse','cuda_evidence':{'hardware':hw,'worker_log':'server-0.log'},'reservation':'gpu run, inherited CUDA_VISIBLE_DEVICES='+env['CUDA_VISIBLE_DEVICES'],'commands':commands,'patch':subprocess.check_output(['git','diff','HEAD'],cwd=cwd,text=True),'binary_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in A.bin_dir.iterdir() if p.is_file()} if name=='system1' else None}
 metadata['harness_sha256']=harness_hash
 metadata['max_model_len']=A.max_model_len
 if name=='system1':
  metadata['native_model_manifest_sha256']=hashlib.sha256((A.native_model/'open_jev_export.json').read_bytes()).hexdigest()
  if A.frozen_metadata:
   frozen=json.loads(A.frozen_metadata.read_text())
   assert metadata['native_graph']==frozen.get('native_graph','0')
   assert metadata['binary_hashes']==frozen['binary_hashes']
   assert A.native_revision==frozen['runtime_revision']
   assert metadata['exact_request_local_prefix']==frozen['exact_request_local_prefix']
   metadata['patch']=frozen['patch'];metadata['frozen_provenance_source']=str(A.frozen_metadata)
 (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
 try:
  for port in (21902, *([21901] if name=='system1' else [])):
   with socket.socket() as probe:probe.bind(('127.0.0.1',port))
  for i,cmd in enumerate(commands):spawn(cmd,env,out/f'server-{i}.log',cwd)
  async with httpx.AsyncClient(timeout=2,trust_env=False) as client:
   for _ in range(900):
    if any(p.poll() is not None for p in procs):raise RuntimeError('server exited')
    try:
     r=await client.get('http://127.0.0.1:21902/health')
     if r.status_code==200:break
    except httpx.HTTPError:pass
    await asyncio.sleep(1)
   else:raise RuntimeError('readiness timeout')
  print(name,'ready',flush=True)
  model='open-jev' if name=='system1' else 'ZefanCai/Open-Jev-9B'
  endpoint='http://127.0.0.1:21902/v1/systemone'
  async with httpx.AsyncClient(timeout=180,trust_env=False,limits=httpx.Limits(max_connections=64,max_keepalive_connections=64)) as client:
   async def measure(items,c,duration,path):
    async def send(case):return await bench.request_one(client,endpoint,case,model,180)
    with path.open('w') as file:
     def record(row):file.write(json.dumps(row)+'\n');file.flush()
     return await collect(send,items,c,duration,on_result=record,stop_on_failure=True)
   for length in plan['lengths']:
    items=[x for x in cases if x['prompt_tokens']==length];random.Random(plan['seed']).shuffle(items)
    for c in plan['concurrencies']:
     dest=out/f't{length}-c{c}';dest.mkdir()
     (dest/'requests.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in items))
     warm,_=await measure(items,c,None,dest/'warmup.jsonl')
     if any(x['error'] or x['state']!='completed' for x in warm):raise RuntimeError('warmup failed')
     rows,elapsed=await measure(items,c,plan['duration_seconds_per_cell'],dest/'responses.jsonl')
     good=[r for r in rows if not r['error'] and r['state']=='completed'];lat=sorted(r['latency_ms'] for r in good)
     summary=dict(requests=len(rows),success=len(good),errors=len(rows)-len(good),wall_seconds=elapsed,dispatch_duration_seconds=plan['duration_seconds_per_cell'],qps=len(good)/elapsed,p50_ms=statistics.median(lat) if lat else None,p95_ms=lat[math.ceil(.95*len(lat))-1] if lat else None,latency_basis=plan['latency_basis'],request_counts={x['id']:sum(r['id']==x['id'] for r in rows) for x in items})
     (dest/'summary.json').write_text(json.dumps(summary,indent=2));print(name,round_no,length,c,json.dumps({k:v for k,v in summary.items() if k!='request_counts'}),flush=True)
     if len(good)!=len(rows):raise RuntimeError('measurement failure; raw failures retained')
 finally:stop_all()
async def main():
 for round_no in range(1,A.rounds+1):
  names=['system1','vllm-jev'] if A.backend=='both' else [A.backend]
  if round_no%2==0:names.reverse()
  for name in names:await backend(name,round_no)
try:asyncio.run(main())
finally:stop_all()
