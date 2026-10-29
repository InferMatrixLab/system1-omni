"""Freeze Public231 and generated shape workloads before optimization.
Lengths use the native export's complete per-candidate template, without truncation.
"""
import collections,hashlib,json,sys
from pathlib import Path
from tokenizers import Tokenizer
R=Path('/home/psv/jev-optimization/workloads')
J=R/'jevbench-7ce310c7262ed49cc85853339a8a42459298e3f3'
sys.path.insert(0,str(J))
from jevbench.tasks import load_jsonl,dataset_hash
from jevbench.adapters.base import build_question
M=Path('/home/psv/jev-comparison/system1-9b')
meta=json.loads((M/'open_jev_export.json').read_text());tok=Tokenizer.from_file(str(M/'tokenizer.json'))
def render(v):return v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,sort_keys=True,allow_nan=False)
def lengths(request):
 out=[]
 for q in request['questions'].values():
  ins=render(q['instructions']);crit=q.get('criteria');kind=q['type']
  if kind=='noul' and crit is not None:ins+='\nYes means: '+render(crit['true'])+'\nNo means: '+render(crit['false'])
  prefix='Context:\n'+render(request['state'])+'\n\nQuestion: '+ins+'\n'
  if kind=='noul':prompts=[prefix+'Is the answer to this question yes? Answer Yes or No.']
  else:
   opts=[k if v is None else k+': '+render(v) for k,v in crit.items()] if kind=='choice' else [render(v) for v in crit]
   prompts=[prefix+'Proposed answer: '+v+'\nIs this proposed answer correct? Answer Yes or No.' for v in opts]
  out += [len(tok.encode(meta['chat_prefix']+p.strip()+meta['chat_suffix']).ids) for p in prompts]
 return out

def enrich(case):
 seq=lengths(case['request']);case.update(prompt_tokens=max(seq),candidate_tokens=seq,total_candidate_tokens=sum(seq));return case

def save(name,rows):
 path=R/name;data=''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rows);path.write_text(data);return {'path':str(path),'sha256':hashlib.sha256(data.encode()).hexdigest(),'count':len(rows),'max_candidate_tokens':max(x['prompt_tokens'] for x in rows)}
if __name__=='__main__':
 tasks=sum([load_jsonl(str(J/'datasets/public'/f'{name}.jsonl')) for name in ['easy','original','hard']],[])
 assert len(tasks)==231
 canonical=dataset_hash(tasks)
 assert canonical=='dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51',canonical
 public=[]
 for t in tasks:
  expected=(t.expected=='yes') if t.question['type']=='noul' else t.expected
  public.append(enrich({'id':t.id,'family':t.family,'source':'JevBench@7ce310c7262ed49cc85853339a8a42459298e3f3','request':{'state':t.state,'questions':{'q':build_question(t)}},'expected':{'q':expected}}))
 out={'public231':save('public231.jsonl',public),'canonical_dataset_sha256':canonical,'unsupported_context':[{'id':x['id'],'tokens':x['prompt_tokens']} for x in public if x['prompt_tokens']>4096], 'type_counts':dict(collections.Counter(x['request']['questions']['q']['type'] for x in public)),'length_buckets':dict(collections.Counter(next((b for b in [128,256,512,1024,2048,4096] if x['prompt_tokens']<b),8192) for x in public))}
 (R/'provenance.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
