"""Audit complete fixed-duration cells; pool samples, never average percentiles."""
import argparse,json,math,statistics
from collections import Counter
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('--reference',type=Path,default=Path('/home/psv/jev-optimization/runs/019-v2-shapes-parity/baseline/responses.json'));a=p.parse_args()
plan=json.loads((a.campaign/'plan.json').read_text());cases={x['id']:x for x in map(json.loads,Path(plan['workload']).read_text().splitlines())};expected={x['id']:x for x in json.loads(a.reference.read_text())};results=[]
for backend in ['system1','vllm-jev']:
 launches=sorted(a.campaign.glob(backend+'-r*'))
 if not launches:continue
 assert len(launches)==plan['rounds'], 'incomplete launch count'
 metadata=[json.loads((d/'metadata.json').read_text()) for d in launches]
 for key in ['runtime_revision','device_evidence','compute_capability','precision','cache_policy']:
  assert len({m[key] for m in metadata})==1,(backend,key)
 for length in plan['lengths']:
  for concurrency in plan['concurrencies']:
   all_rows=[];elapsed=0;round_metrics=[]
   for launch in launches:
    cell=launch/f't{length}-c{concurrency}';summary=json.loads((cell/'summary.json').read_text());rows=list(map(json.loads,(cell/'responses.jsonl').read_text().splitlines()));warm=list(map(json.loads,(cell/'warmup.jsonl').read_text().splitlines()))
    ids={k for k,v in cases.items() if v['prompt_tokens']==length};counts=Counter(x['id'] for x in rows)
    assert len(warm)==len(ids) and {x['id'] for x in warm}==ids
    assert all(x['state']=='completed' and not x['error'] for x in warm+rows)
    assert set(counts)==ids and len(set(counts.values()))==1,'unequal request mix'
    assert summary['requests']==len(rows) and summary['errors']==0
    all_rows.extend(rows);elapsed+=summary['wall_seconds'];round_metrics.append({k:summary[k] for k in ['requests','qps','p50_ms','p95_ms']})
   lat=sorted(x['latency_ms'] for x in all_rows);flips=0;max_probability=0;max_score=0;bad_ids=set()
   for row in all_rows:
    kind=cases[row['id']]['request']['questions']['q']['type'];ref=expected[row['id']]['answers']['q'];actual=row['answers']['q'];rp=ref['probabilities'];cp=actual['probabilities'];prob=max(abs(rp[k]-cp[k]) for k in rp);max_probability=max(max_probability,prob)
    score=abs(ref['value']-actual['value']) if kind=='score' else 0;max_score=max(max_score,score)
    def decision(x):return (x['value']>=.5) if kind=='noul' else x['value'] if kind=='choice' else max(x['probabilities'],key=x['probabilities'].get)
    flip=decision(ref)!=decision(actual);flips+=flip
    if flip or prob>.001 or score>.001:bad_ids.add(row['id'])
    if backend=='system1':assert json.loads(row['response'])['usage']['input_tokens']==cases[row['id']]['total_candidate_tokens']
   result=dict(backend=backend,length=length,concurrency=concurrency,requests=len(all_rows),errors=0,p50_ms=statistics.median(lat),p95_ms=lat[math.ceil(.95*len(lat))-1],qps=len(all_rows)/elapsed,rounds=round_metrics,device=metadata[0]['device_evidence'],runtime_revision=metadata[0]['runtime_revision'],precision_vs_original_native=dict(decision_flips=flips,unique_failed_ids=sorted(bad_ids),max_probability=max_probability,max_score=max_score),native_precision_pass=not bad_ids if backend=='system1' else None)
   results.append(result);print(json.dumps({k:v for k,v in result.items() if k not in ['rounds','precision_vs_original_native']}))
(a.campaign/'pooled-audit.json').write_text(json.dumps({'protocol':plan,'cells':results,'note':'vLLM numerical differences vs original native are diagnostic; native precision tolerance0flips,maxabs.001.'},indent=2))
