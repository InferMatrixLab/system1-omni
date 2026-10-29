"""Fresh-process baseline/candidate parity on one reserved device; diagnostic timings only."""
import argparse,asyncio,json,os,signal,subprocess,sys,time,hashlib
from pathlib import Path
import httpx
sys.path.insert(0,'/home/psv/system1-omni/benchmarks');from bench import load_cases,read_answers
P=argparse.ArgumentParser();P.add_argument('--candidate',type=Path,required=True);P.add_argument('--out',type=Path,required=True);P.add_argument('--manifest',type=Path,default=Path('/home/psv/jev-optimization/workloads/public231.jsonl'));P.add_argument('--exact-prefix',choices=['0','1'],default='0');P.add_argument('--prefix',choices=['0','1','auto'],default='0');P.add_argument('--candidate-model',type=Path,default=Path('/home/psv/jev-comparison/system1-9b'));P.add_argument('--candidate-graph',choices=['0','1'],default='0');A=P.parse_args();A.out.mkdir(parents=True,exist_ok=False)
cases=load_cases(A.manifest)
(A.out/'config.json').write_text(json.dumps({'args':vars(A),'manifest_sha256':hashlib.sha256(A.manifest.read_bytes()).hexdigest(),'precision':'BF16','decision_flips':0,'max_probability_or_score_abs_drift':0.001,'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES')},default=str,indent=2))
async def run(name,bin_dir):
 out=A.out/name;out.mkdir();env=os.environ.copy();env.update(OPEN_JEV_MODEL=str(A.candidate_model) if name=='candidate' else '/home/psv/jev-comparison/system1-9b',OPEN_JEV_CUDA_LIB=str(bin_dir/'libqwen3_5_cuda.so'),OPEN_JEV_PORT='20901',OMNI_JEV_BIND='127.0.0.1:20902',OMNI_JEV_BACKEND_URL='http://127.0.0.1:20901',RAYON_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUA_S1_GRAPH=A.candidate_graph if name=='candidate' else '0',OPEN_JEV_PREFIX=A.prefix if name=='candidate' else '0',OPEN_JEV_EXACT_PREFIX=A.exact_prefix if name=='candidate' else '0')
 procs=[]
 try:
  for exe in ['omni-open-jev-native','omni-jev']:
   with (out/(exe+'.log')).open('w') as f:p=subprocess.Popen([str(bin_dir/exe)],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   procs.append(p)
  rows=[]
  async with httpx.AsyncClient(timeout=120,trust_env=False) as client:
   for _ in range(120):
    if any(p.poll()is not None for p in procs):raise RuntimeError('server exited')
    try:
     r=await client.get('http://127.0.0.1:20902/health',timeout=1)
     if r.status_code==200:break
    except httpx.HTTPError:pass
    await asyncio.sleep(.5)
   else:raise RuntimeError('readiness timeout')
   for case in cases:
    t=time.perf_counter();r=await client.post('http://127.0.0.1:20902/v1/systemone',json=case['request']);body=r.json();assert r.status_code==200,(r.status_code,body)
    ans=read_answers(case,body);rows.append({'id':case['id'],'latency_ms':1000*(time.perf_counter()-t),'answers':ans,'body':body})
  (out/'responses.json').write_text(json.dumps(rows,indent=2));print(name,'complete',len(rows),flush=True);return rows
 finally:
  for p in reversed(procs):
   if p.poll()is None:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(15)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
async def main():
 base=await run('baseline',Path('/home/psv/jev-optimization/baseline/bin'));cand=await run('candidate',A.candidate)
 diffs=[]
 for case,b,c in zip(cases,base,cand):
  assert case['id']==b['id']==c['id'];bp=b['answers']['q']['probabilities'];cp=c['answers']['q']['probabilities'];kind=case['request']['questions']['q']['type'];bv=b['answers']['q']['value'];cv=c['answers']['q']['value']
  bd=(bv>=.5) if kind=='noul' else bv if kind=='choice' else max(bp,key=bp.get)
  cd=(cv>=.5) if kind=='noul' else cv if kind=='choice' else max(cp,key=cp.get)
  diffs.append({'id':case['id'],'probability_delta':max(abs(bp[k]-cp[k]) for k in bp),'score_delta':abs(bv-cv) if kind=='score' else 0,'decision_flip':bd!=cd,'baseline_ms':b['latency_ms'],'candidate_ms':c['latency_ms']})
 summary={'requests':len(diffs),'decision_flips':sum(x['decision_flip'] for x in diffs),'max_probability_delta':max(x['probability_delta'] for x in diffs),'max_score_delta':max(x['score_delta'] for x in diffs),'GPU_VISIBLE':os.environ.get('CUDA_VISIBLE_DEVICES'),'failures':[x for x in diffs if x['decision_flip'] or x['probability_delta']>.001 or x['score_delta']>.001],'note':'Exploratory fresh-process outputs; timing is not acceptance evidence.'}
 (A.out/'differences.json').write_text(json.dumps(diffs,indent=2));(A.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True);return bool(summary['failures'])
raise SystemExit(asyncio.run(main()))
