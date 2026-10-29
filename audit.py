"""CPU-only independent audit of the published samples (Python standard library)."""
import hashlib,json,math,re,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 inventory=json.loads((ROOT/'SHA256.json').read_text())
 for name,expected in inventory.items():assert digest(ROOT/name)==expected,name
 answers=json.loads((ROOT/'answers.json').read_text())
 refs={}
 for corpus in ['public','shapes']:
  p=ROOT/f'parity/068-{corpus}-parity/baseline/responses.json'
  for row in json.loads(p.read_text()):refs[row['id']]=row['answers']
 pooled={};count=0;parity_count=0;max_delta=0.;flips=0
 for run in json.loads((ROOT/'runs.json').read_text()):
  path=ROOT/'measurements'/run
  rows=[json.loads(s) for s in (path/'samples.jsonl').read_text().splitlines()]
  summary=json.loads((path/'summary.json').read_text())
  assert len(rows)==summary.get('requests',summary.get('attempted_requests')),run
  good=[x for x in rows if x.get('status')==200 and not x.get('error') and x.get('state','completed')=='completed']
  assert len(good)==len(rows),(run,'failed requests')
  lat=[x['latency_ms'] for x in rows];wall=summary['wall_seconds']
  metrics=dict(p50_ms=statistics.median(lat),p95_ms=sorted(lat)[math.ceil(.95*len(lat))-1],qps=len(lat)/wall)
  for key,long in [('p50_ms','successful_latency_p50_ms'),('p95_ms','successful_latency_p95_ms'),('qps','requests_per_second')]:
   expected=summary.get(key,summary.get(long))
   if expected is not None:assert math.isclose(metrics[key],expected,abs_tol=1e-8),(run,key)
  # Pool rounds, never average percentiles. Keep launches/backend names distinct.
  group=re.sub(r'-r[12]/t', '/t',run)
  group=re.sub(r'/public-c(\d+)-r[123]$',r'/public-c\1',group)
  bucket=pooled.setdefault(group,{'lat':[],'wall':0.})
  bucket['lat'].extend(lat);bucket['wall']+=wall;count+=len(rows)
  if run.startswith(('068-public231-vector-residual-norm/','069-shapes-native-vector-norm/')):
   for row in rows:
    got=answers[row['answers_ref']];ref=refs[row['id']]
    assert got==ref,(run,row['id'],'native answer differs from original')
    parity_count+=1
 result={'runs':len(json.loads((ROOT/'runs.json').read_text())),'samples':count,
         'formal_native_answers_exact_to_original':parity_count,
         'manifest_sha256':{n:digest(ROOT/'workloads'/n) for n in ['public231.jsonl','shapes.jsonl']},
         'pooled':{k:{'requests':len(v['lat']),'p50_ms':statistics.median(v['lat']),
          'p95_ms':sorted(v['lat'])[math.ceil(.95*len(v['lat']))-1],
          'qps':len(v['lat'])/v['wall']} for k,v in pooled.items()}}
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
