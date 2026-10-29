import json,math,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);result=[]
for backend in ['system1','vllm-jev']:
 p=root/backend
 if not p.exists():continue
 meta=json.loads((p/'metadata.json').read_text())
 for concurrency in [1,8,16,32]:
  rounds=[p/f'public-c{concurrency}-r{i}' for i in range(1,4)]
  if not all((d/'summary.json').exists() for d in rounds):continue
  rows=[];seconds=0;errors=0;per_round=[]
  for d in rounds:
   summary=json.loads((d/'summary.json').read_text());seconds+=summary['wall_seconds'];errors+=summary['failed_requests'];per_round.append({k:summary[k] for k in ['successful_latency_p50_ms','successful_latency_p95_ms','requests_per_second']});rows.extend(json.loads(x) for x in (d/'responses.jsonl').read_text().splitlines())
  v=sorted(r['latency_ms'] for r in rows if r['error'] is None)
  entry={'backend':backend,'concurrency':concurrency,'requests':len(rows),'errors':errors,'p50_ms':statistics.median(v),'p95_ms':v[math.ceil(.95*len(v))-1],'QPS':len(v)/seconds,'rounds':per_round,'device':meta['gpu'],'compute_capability':meta['compute_capability'],'runtime_revision':meta['runtime_revision']};result.append(entry);print(json.dumps(entry))
(root/'pooled-summary.json').write_text(json.dumps(result,indent=2))
