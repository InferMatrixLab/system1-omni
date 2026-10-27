"""Pinned three-way Kev acceptance: reference, exported-weight baseline, native.

GPU subcommands must be invoked through the execution host's reservation tool.
Never changes thresholds, chooses easier cases, or overwrites previous outputs.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('kev_fixture', ROOT / 'tests/kev/generate_processing_fixture.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def reference(args):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    sys.path.insert(0, str(Path(__file__).parent))
    import export
    export.verify_files(args.base, export.BASE_FILES)
    export.verify_files(args.adapter, export.CHECKPOINT_FILES)
    manifest = json.loads((args.export / 'kev_export.json').read_text())
    export.verify_files(args.export, manifest['files'])
    api, model = fixture.official(args.reference)
    tok = model.load_tokenizer(args.base)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    meta = torch.load(args.adapter / 'head.pt', weights_only=True, map_location='cpu')
    export.pointer_head(meta)
    provenance = {'source_hashes': fixture.SOURCE_HASHES,
                  'runtime_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'runtime_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT)).hexdigest(), 'reference_revision': export.REFERENCE_REVISION,
                  'base_revision': export.BASE_REVISION, 'adapter_revision': export.REVISION,
                  'temperature': meta['temperature'], 'torch_cuda': torch.version.cuda,
                  'device': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
                  'dependencies': {n: importlib.metadata.version(n) for n in
                                   ('torch', 'transformers', 'peft', 'accelerate', 'pydantic', 'tokenizers', 'safetensors', 'numpy')}}
    corpus = json.loads((ROOT / 'tests/kev/data/processing.json').read_text())['cases']
    for kind in ('official', 'fp32-merged', 'merged'):
        path = args.out / f'{kind}.json'
        if path.exists():
            raise ValueError(f'preserve previous evidence: {path} exists')
        print(f'Loading {kind}', flush=True)
        if kind == 'fp32-merged':
            # Isolate PEFT merge arithmetic from BF16 export/compute precision.
            m.lm = m.lm.merge_and_unload(safe_merge=True)
        else:
            m = model.DecisionModel(str(args.base), tok, 'cuda', dtype=torch.float32 if kind == 'official' else torch.bfloat16, attn='eager')
        if kind == 'official':
            m.lm = PeftModel.from_pretrained(m.lm, str(args.adapter), torch_device='cuda').to('cuda')
        elif kind == 'merged':
            # Exact exporter values, including BF16 norms. No PEFT merge is rerun here.
            weights = {}
            for name in manifest['files']:
                if name.startswith('model.') and name.endswith('.safetensors'):
                    weights.update({k.removeprefix('model.language_model.'): v for k, v in load_file(args.export / name).items()})
            m.lm.load_state_dict(weights, strict=True)
            del weights
        m.head.load_state_dict(meta['head'])
        m.head.temperature = meta['temperature']
        m.eval()
        results = []
        with torch.inference_mode():
            for case in corpus:
                req = api.SystemOneRequest.model_validate(case['request'])
                rec, mapping = api.to_record(req)
                enc = model.encode(tok, rec, max_state=8192, max_branch=16384, strict=True)
                assert len(enc['ids']) == case['input_tokens']
                logits = m(enc)  # official independent complete causal rows, no prefix reuse
                ps = [torch.softmax(z, -1).double().cpu().tolist() for z in logits]
                answers = api.to_answers(ps, mapping)
                results.append({'name': case['name'], 'probabilities': ps,
                                'calibrated_logits': [z.double().cpu().tolist() for z in logits],
                                'body': {'model': req.model, 'answers': answers,
                                         'usage': {'input_tokens': len(enc['ids']), 'output_tokens': api.output_tokens(tok, answers)}}})
                save(path, {'kind': kind, 'provenance': provenance, 'cases': results})
                print(kind, case['name'], 'questions', len(ps), flush=True)
        if kind != 'official':
            del m
            gc.collect()
            torch.cuda.empty_cache()


def compare(args):
    corpus = json.loads((ROOT / 'tests/kev/data/processing.json').read_text())['cases']
    outputs = {kind: json.loads((args.out / f'{kind}.json').read_text()) for kind in ('official', 'fp32-merged', 'merged', 'native')}
    report = {'gates': {'probability_max_abs': 0.01, 'choice_margin': 0.02, 'noul_distance': 0.01},
              'comparisons': {}}
    for left, right in (('official', 'fp32-merged'), ('fp32-merged', 'merged'), ('official', 'merged'), ('merged', 'native'), ('official', 'native')):
        records, failures, boundary = [], [], []
        a, b = outputs[left]['cases'], outputs[right]['cases']
        assert len(a) == len(corpus) and len(b) >= len(corpus), 'incomplete corpus'
        for expected, ca, cb in zip(corpus, a, b):
            assert expected['name'] == ca['name'] == cb['name']
            assert ca['body']['usage']['input_tokens'] == cb['body']['usage']['input_tokens'] == expected['input_tokens']
            assert len(ca['probabilities']) == len(cb['probabilities']) == len(expected['meta'])
            for meta, pa, pb in zip(expected['meta'], ca['probabilities'], cb['probabilities']):
                assert len(pa) == len(pb) == len(meta['keys'])
                error = max(abs(x-y) for x, y in zip(pa, pb))
                row = {'case': expected['name'], 'question': meta['id'], 'type': meta['type'], 'probability_max_abs': error}
                if error > 0.01:
                    failures.append({**row, 'gate': 'probability'})
                if meta['type'] == 'choice':
                    top = sorted(range(len(pa)), key=lambda i: -pa[i])
                    margin = pa[top[0]] - (pa[top[1]] if len(top) > 1 else 0)
                    agree = top[0] == max(range(len(pb)), key=lambda i: pb[i])
                    row.update(margin=margin, decision_agrees=agree)
                    if not agree:
                        dest = failures if margin > 0.02 else boundary
                        dest.append({**row, 'gate': 'choice', 'left': pa, 'right': pb})
                if meta['type'] == 'noul':
                    distance = abs(pa[1]-0.5)
                    agree = (pa[1] >= 0.5) == (pb[1] >= 0.5)
                    row.update(threshold_distance=distance, decision_agrees=agree)
                    if not agree:
                        dest = failures if distance > 0.01 else boundary
                        dest.append({**row, 'gate': 'noul', 'left': pa[1], 'right': pb[1]})
                if meta['type'] == 'score':
                    row['score_abs_error'] = abs(sum(i*(x-y) for i, (x, y) in enumerate(zip(pa, pb))))
                records.append(row)
        report['comparisons'][f'{left}-to-{right}'] = {'max_abs': max(x['probability_max_abs'] for x in records),
            'questions': len(records), 'failures': failures, 'near_boundary_disagreements': boundary, 'records': records}
    native = outputs['native']['cases']
    assert len(native) == len(corpus)+3, 'missing repeat captures'
    first = next(c for c in native if c['name'] == 'mixed')
    report['repeat_exact'] = all(c['body'] == first['body'] and c['probabilities'] == first['probabilities'] for c in native[len(corpus):])
    save(args.out / 'comparison.json', report)
    for name, result in report['comparisons'].items():
        print(name, 'max_abs', result['max_abs'], 'failures', len(result['failures']), 'boundary', len(result['near_boundary_disagreements']))
    return 0 if report['repeat_exact'] and all(not c['failures'] for c in report['comparisons'].values()) else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    ref = sub.add_parser('reference')
    for flag in ('reference', 'base', 'adapter', 'export', 'out'):
        ref.add_argument('--'+flag, type=Path, required=True)
    cmp = sub.add_parser('compare')
    cmp.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == 'reference':
        reference(args)
        return 0
    return compare(args)


if __name__ == '__main__':
    raise SystemExit(main())
