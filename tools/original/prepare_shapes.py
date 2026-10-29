"""Controlled length/option-count stress cases, separate from labelled Public231."""
import json,random
from pathlib import Path
from prepare_workloads import enrich,lengths,save,R
rng=random.Random(231)
topics=[('billing','Invoice and payment problems','My card was charged twice; please correct the invoice.'),('technical','Application failures','The application crashes on launch; please repair it.'),('security','Unauthorized access','An unknown person accessed my account; please secure it.'),('shipping','Package delivery','My parcel has not arrived; please locate the shipment.'),('returns','Product returns','I want to send the item back; please provide return instructions.'),('account','Profile changes','Please change the name shown on my profile.'),('sales','Purchase advice','I need advice on choosing a product before purchase.'),('feedback','Product suggestions','Please add an export button in a future release.')]
levels=['No service impact','Minor inconvenience','Partial loss of service','Major loss of service','Complete service outage']
rows=[]
for target in [128,512,2048,4096]:
 for kind,count in [('noul',1),('choice',2),('choice',4),('choice',8),('score',3),('score',5)]:
  for i in range(8):
   if kind=='choice':
    selected=rng.sample(topics,count);answer=selected[i%count];q={'type':kind,'instructions':'Classify the final ticket; ignore the archive.','criteria':{key:desc for key,desc,_ in selected}};ticket=answer[2];expected=answer[0]
   elif kind=='noul':
    expected=i%2==0;ticket='Please refund the duplicate charge.' if expected else 'Please change my profile name; no refund is requested.';q={'type':kind,'instructions':'Does the final ticket request a refund? Ignore the archive.'}
   else:
    criteria=levels if count==5 else [levels[0],levels[2],levels[4]];expected=i%count;ticket='Observed current impact: '+criteria[expected]+'.';q={'type':kind,'instructions':'Rate the impact of the final ticket; ignore the archive.','criteria':criteria}
   request={'state':'','questions':{'q':q}}
   def set_padding(n):request['state']=f'Archive record {i}: '+(' entry'*n)+'\nFinal ticket: '+ticket
   set_padding(0);minimum=max(lengths(request));assert minimum<=target,(kind,count,minimum,target)
   set_padding(target-minimum);delta=target-max(lengths(request));set_padding(target-minimum+delta)
   case=enrich({'id':f'shape-{target}-{kind}{count}-{i}','slice':f't{target}','kind':kind,'candidate_count':count,'expected':{'q':expected},'request':request,'label_scope':'Controlled synthetic rubric; used primarily for parity and performance, not general model quality.'})
   assert case['prompt_tokens']==target,(case['id'],case['prompt_tokens']);rows.append(case)
print(json.dumps(save('shapes.jsonl',rows),indent=2))
plan={'seed':231,'lengths':[128,512,2048,4096],'candidate_counts':{'noul':[1],'choice':[2,4,8],'score':[3,5]},'cases_per_kind_count_length':8,'concurrencies':[1,8,16,32],'duration_seconds_per_cell':30,'independent_service_launch_rounds':2,'arrival_model':'closed-loop with fixed worker count; stop new dispatch at time limit then drain','warmup':'one complete traversal of each slice before its measurement','request_order':'deterministic seeded shuffle per slice; repeated cyclically','latency_basis':'post_to_body_json_and_schema_validation','caches':'no response caching; prefix reuse disabled on both','overflow_cases':'test separately; never truncate or include expected failures in successful throughput','precision_reference':'Original frozen native baseline, max probability/Score abs drift 0.001 and zero decisions flipped'}
(R/'shapes-plan.json').write_text(json.dumps(plan,indent=2))
