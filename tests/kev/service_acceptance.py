"""Pinned default-service acceptance. GPU subcommands MUST run under gpu run.

The official server is unchanged: observers only save its unrounded response
inputs and count graph replays. No precision, cache, graph or kernel override.
"""
import argparse
import collections
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = '2074cc44256200bb37540103fa4572901faa10f3'
SOURCES = {
    'v4/transfer-v4': 'ff374c49c6c9f15f8a56fb274b4a4857d20497eb8dd1ac07ce01560e682a5f2e',
    'v4/decision-v4': '8d5765d7aec4d08c61854f4664ca79ec2ba44ed092e9b967eaf61ed86c496d9c',
    'hard-v1': '3cdbf12a6b7c3b70c73e8c39f2677127ea61ef32f0ea75b55a43556d884b2d98',
}
GATES = {
    'probability_max_abs': 0.01, 'score_max_abs': 0.1,
    'choice_margin': 0.02, 'noul_distance': 0.01,
    'minimum_boundary_per_type': 10,
    'accuracy_drop': 0.01, 'per_type_accuracy_drop': 0.02,
    'brier_increase': 0.01, 'normalized_score_mae_increase': 0.005,
}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().split('\n') if line.strip()]


def request_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def prepare(args):
    """Choose every task row before inference; discovery never uses native scores."""
    if args.out.exists():
        raise ValueError('preserve previous corpus')
    fixture = module('kev_fixture', ROOT/'tests/kev/generate_processing_fixture.py')
    api, model = fixture.official(args.reference)
    tok = model.load_tokenizer(args.base)
    cases = [{**c, 'population': 'contract', 'labels': {}, 'source': 'contract'} for c in fixture.cases()]
    for suite, sha in SOURCES.items():
        path = args.reference/'evals'/suite/'development.jsonl'
        assert digest(path) == sha, path
        for record in read_lines(path):
            meta = record['_meta']
            if meta['variant'] != 'clean':
                continue
            if suite == 'v4/decision-v4' and not any(q['type'] == 'score' for q in record['questions'].values()):
                continue
            if suite == 'hard-v1' and meta['source'] not in ('hard_ambiguous', 'hard_probability'):
                continue
            questions, labels = {}, {}
            for qid, q in record['questions'].items():
                if suite == 'v4/decision-v4' and q['type'] != 'score':
                    continue
                questions[qid] = {k: q[k] for k in ('type', 'instructions', 'criteria') if k in q}
                keys = api.question_keys(q['type'], q.get('criteria'))
                labels[qid] = keys.index(q['label']) if q['type'] == 'choice' else int(q['label'])
            cases.append({'name': f'{suite}:{meta["id"]}', 'population': 'discovery' if suite == 'hard-v1' else 'task',
                          'source': meta['source'], 'group': meta.get('group_id', meta['id']), 'labels': labels,
                          'request': {'model': 'kev-latest', 'state': record['state'], 'questions': questions}})
    # Explicitly synthetic stress inputs, kept separate from labeled task quality.
    # A fixed sweep is retained in full, even when it does not yield a boundary.
    for i in range(101):
        chance = 45 + i/10
        cases.append({'name': f'boundary-probe-{i:03}', 'population': 'discovery', 'source': 'synthetic_probability', 'labels': {},
                      'request': {'model': 'kev-latest', 'state': f'A lottery has exactly {chance:.1f}% winning tickets. One ticket is drawn uniformly at random. Its outcome has not been revealed.',
                                  'questions': {'outcome': {'type': 'choice', 'instructions': 'Predict the outcome of the ticket.',
                                                           'criteria': {'win': 'The ticket wins.', 'lose': 'The ticket does not win.'}},
                                                'wins': {'type': 'noul', 'instructions': 'The ticket wins.'}}}})
    mixed = next(c for c in cases if c['name'] == 'mixed')
    cases += [{**mixed, 'name': f'mixed-repeat-{i}', 'population': 'repeat'} for i in range(3)]
    captured = []
    for case in cases:
        captured.append(fixture.capture(case, api, model, tok))
        assert len(captured[-1]['rows']) <= 64
        assert sum(len(row['ids']) for row in captured[-1]['rows']) <= 65536
    assert len({c['name'] for c in captured}) == len(captured)
    counts = collections.Counter(c['population'] for c in captured)
    save(args.out, {'reference': REFERENCE, 'sources': SOURCES, 'gates': GATES, 'counts': counts, 'cases': captured})
    print('frozen corpus', dict(counts), 'sha256', digest(args.out), flush=True)


def official_server(args):
    """Run the official main with environment defaults, recording raw probabilities."""
    overrides = sorted(k for k in os.environ if k.startswith('KEV_'))
    if overrides:
        raise ValueError(f'official defaults require no KEV_* environment variables: {overrides}')
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.reference, text=True).strip() == REFERENCE
    assert not subprocess.check_output(['git', 'diff', 'HEAD', '--', 'kev'], cwd=args.reference)
    sys.path.insert(0, str(args.reference))
    import torch
    from kev import serve
    from kev.checkpoint import fused_available
    replay_count = 0
    original_replay = torch.cuda.CUDAGraph.replay

    def observed_replay(graph):
        nonlocal replay_count
        result = original_replay(graph)
        replay_count += 1
        return result

    torch.cuda.CUDAGraph.replay = observed_replay
    original_body = serve.Server._body
    with args.out.open('x') as stream:
        def observed_body(server, req, meta, probabilities, stats):
            body = original_body(server, req, meta, probabilities, stats)
            entry = {'request_digest': request_digest(req.model_dump()), 'probabilities': probabilities, 'body': body,
                     'stats': stats, 'graph_replays': replay_count, 'graphs': server.model.graphs.stats(),
                     'runtime': {'dtype': server.model.dtype, 'head_dtype': str(next(server.model.head.parameters()).dtype),
                                 'attention': server.model.lm.config._attn_implementation,
                                 'fused_available': fused_available(),
                                 'remaining_lora_modules': sum(hasattr(m, 'lora_A') for m in server.model.lm.modules()),
                                 'temperature': server.model.head.temperature, 'device': torch.cuda.get_device_name(),
                                 'capability': torch.cuda.get_device_capability(), 'torch_cuda': torch.version.cuda}}
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            return body
        serve.Server._body = observed_body
        sys.argv = ['kev.serve', '--run', str(args.adapter), '--device', 'cuda', '--host', '127.0.0.1', '--port', str(args.port)]
        serve.main()


def boundary_supplement(args):
    """Predeclared full-range Noul sweep after the first pool's coverage failure."""
    if args.out.exists():
        raise ValueError('preserve previous corpus')
    fixture = module('kev_fixture', ROOT/'tests/kev/generate_processing_fixture.py')
    api, model = fixture.official(args.reference)
    tok = model.load_tokenizer(args.base)
    cases = []
    for template in ('tickets', 'balls'):
        for i in range(1001):
            chance = i/10
            state = (f'A lottery has exactly {chance:.1f}% winning tickets. One ticket is drawn uniformly at random. Its outcome has not been revealed.'
                     if template == 'tickets' else
                     f'A bag contains red and blue balls. Exactly {chance:.1f}% of the balls are red. One ball is drawn uniformly at random without looking. Its color has not been revealed.')
            instruction = 'The ticket wins.' if template == 'tickets' else 'The drawn ball is red.'
            case = {'name': f'boundary-supplement-{template}-{i:04}', 'population': 'discovery',
                    'source': f'synthetic_{template}_full_range', 'labels': {},
                    'request': {'model': 'kev-latest', 'state': state, 'questions': {'event': {'type': 'noul', 'instructions': instruction}}}}
            cases.append(fixture.capture(case, api, model, tok))
    save(args.out, {'reference': REFERENCE, 'sources': {}, 'gates': GATES, 'counts': {'discovery': len(cases)}, 'cases': cases})
    print('frozen supplement', len(cases), 'sha256', digest(args.out), flush=True)


def reference(args):
    http = module('kev_http', ROOT/'recipe/kev/validate_http.py')
    args.out.mkdir(parents=True, exist_ok=False)
    corpus = json.loads(args.corpus.read_text())
    assert corpus['gates'] == GATES
    exporter = module('kev_exporter', ROOT/'recipe/kev/export.py')
    exporter.verify_files(args.adapter, exporter.CHECKPOINT_FILES)
    exporter.verify_files(args.adapter/'base', exporter.BASE_FILES)
    port = http.port()
    url = f'http://127.0.0.1:{port}'
    env = {k: v for k, v in os.environ.items() if not k.startswith('KEV_')}
    command = [sys.executable, str(Path(__file__).resolve()), 'official-server', '--reference', str(args.reference),
               '--adapter', str(args.adapter), '--out', str(args.out/'observations.jsonl'), '--port', str(port)]
    save(args.out/'provenance.json', {'command': command, 'reference': REFERENCE, 'corpus_sha256': digest(args.corpus),
                                    'harness_sha256': digest(__file__), 'gates': GATES,
                                    'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                                    'source_diff': subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT, text=True),
                                    'dependencies': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}})
    with (args.out/'server.log').open('w') as log, (args.out/'responses.jsonl').open('x') as responses:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'official server exited {process.returncode}; see server.log')
                try:
                    status, models = http.request(url+'/v1/models')
                    if status == 200:
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('official readiness deadline')
                time.sleep(0.5)
            save(args.out/'models-before.json', models)
            for i, case in enumerate(corpus['cases']):
                status, body = http.request(url+'/v1/systemone', case['request'])
                responses.write(json.dumps({'name': case['name'], 'status': status, 'body': body}, ensure_ascii=False)+'\n')
                responses.flush()
                if status != 200:
                    raise ValueError((case['name'], status, body))
                if i % 25 == 0:
                    print('official', i+1, '/', len(corpus['cases']), case['name'], flush=True)
            save(args.out/'models-after.json', http.request(url+'/v1/models')[1])
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    observations = read_lines(args.out/'observations.jsonl')
    responses = read_lines(args.out/'responses.jsonl')
    assert len(observations) == len(responses) == len(corpus['cases'])
    fixture = module('kev_fixture', ROOT/'tests/kev/generate_processing_fixture.py')
    api, _ = fixture.official(args.reference)
    captured = []
    for c, obs, response in zip(corpus['cases'], observations, responses):
        assert obs['body'] == response['body']
        assert obs['request_digest'] == request_digest(api.SystemOneRequest.model_validate(c['request']).model_dump())
        assert obs['runtime']['dtype'] == 'bfloat16' and obs['runtime']['head_dtype'] == 'torch.float32'
        assert obs['runtime']['remaining_lora_modules'] == 0
        captured.append({'name': c['name'], **obs})
    save(args.out/'official-service.json', {'cases': captured})
    boundary = []
    for case, record in zip(corpus['cases'], captured):
        for meta, p in zip(case['meta'], record['probabilities']):
            if meta['type'] == 'choice' and len(p) > 1:
                ranked = sorted(p, reverse=True)
                distance, limit = ranked[0]-ranked[1], GATES['choice_margin']
            elif meta['type'] == 'noul':
                distance, limit = abs(p[1]-0.5), GATES['noul_distance']
            else:
                continue
            if distance <= limit:
                boundary.append({'case': case['name'], 'question': meta['id'], 'type': meta['type'],
                                 'source': case['source'], 'distance': distance, 'probabilities': p})
    save(args.out/'boundary-membership.json', {'selection': 'official outputs only, before native', 'items': boundary})
    print('official default capture complete', len(captured), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for flag in ('reference', 'base', 'out'):
        prep.add_argument('--'+flag, type=Path, required=True)
    supplement = sub.add_parser('boundary-supplement')
    for flag in ('reference', 'base', 'out'):
        supplement.add_argument('--'+flag, type=Path, required=True)
    server = sub.add_parser('official-server')
    for flag in ('reference', 'adapter', 'out'):
        server.add_argument('--'+flag, type=Path, required=True)
    server.add_argument('--port', type=int, required=True)
    ref = sub.add_parser('reference')
    for flag in ('reference', 'adapter', 'corpus', 'out'):
        ref.add_argument('--'+flag, type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'boundary-supplement': boundary_supplement,
     'official-server': official_server, 'reference': reference}[args.command](args)


if __name__ == '__main__':
    main()
