"""Compare the frozen default BF16 service campaign; never infer new gates."""
import argparse
import collections
import json
import math
from pathlib import Path

from service_acceptance import GATES, ROOT, digest, module, read_lines, save


def distribution(p, size):
    return (len(p) == size and all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in p)
            and abs(sum(p)-1) <= 1e-4)


def question(meta, p, n, official, native):
    """Independent probability, scalar and decision gates, including wire rounding."""
    kind, size = meta['type'], len(meta['keys'])
    if not distribution(p, size) or not distribution(n, size):
        return {'failures': ['distribution'], 'type': kind}
    error = max(abs(a-b) for a, b in zip(p, n))
    result = {'type': kind, 'levels': size, 'probability_max_abs': error, 'failures': []}
    if error > GATES['probability_max_abs']:
        result['failures'].append('probability')
    if kind == 'score':
        ref_score, native_score = (sum(i*x for i, x in enumerate(row)) for row in (p, n))
        raw_error = abs(ref_score-native_score)
        wire_error = abs(official['score']-native['score'])
        result.update(official_score=ref_score, native_score=native_score, score_abs=raw_error,
                      score_wire_abs=wire_error, score_normalized_abs=raw_error/(size-1) if size > 1 else 0)
        if max(raw_error, wire_error) > GATES['score_max_abs'] or (size == 1 and max(abs(official['score']), abs(native['score'])) != 0):
            result['failures'].append('score')
    else:
        if kind == 'choice':
            ordered = sorted(p, reverse=True)
            distance = ordered[0]-ordered[1] if size > 1 else 1.0
            limit = GATES['choice_margin']
            a, b = (max(range(size), key=row.__getitem__) for row in (p, n))
            wire_flip = official['choice'] != native['choice']
        else:
            distance, limit = abs(p[1]-0.5), GATES['noul_distance']
            a, b = p[1] >= 0.5, n[1] >= 0.5
            wire_flip = (official['noul'] >= 0.5) != (native['noul'] >= 0.5)
        result.update(distance=distance, near_boundary=distance <= limit, raw_flip=a != b, wire_flip=wire_flip)
        if distance > limit and (a != b or wire_flip):
            result['failures'].append('decision')
    return result


def labeled_metrics(kind, p, gold):
    if not 0 <= gold < len(p):
        raise ValueError('invalid frozen gold label')
    result = {'accuracy': float(max(range(len(p)), key=p.__getitem__) == gold),
              'brier': sum((x-float(i == gold))**2 for i, x in enumerate(p)),
              'nll': -math.log(max(p[gold], 1e-9))}
    if kind == 'score':
        error = abs(sum(i*x for i, x in enumerate(p))-gold)
        result.update(score_mae=error, normalized_score_mae=error/max(1, len(p)-1))
    return result


def aggregate(rows):
    import numpy as np
    result = {'questions': len(rows)}
    for metric in ('accuracy', 'brier', 'nll', 'score_mae', 'normalized_score_mae'):
        chosen = [r for r in rows if metric in r['official']]
        if not chosen:
            continue
        a = sum(r['official'][metric] for r in chosen)/len(chosen)
        b = sum(r['native'][metric] for r in chosen)/len(chosen)
        # Pair and resample whole source records, retaining all questions in a record.
        groups = collections.defaultdict(list)
        for r in chosen:
            groups[r['group']].append(r['native'][metric]-r['official'][metric])
        sums = np.array([sum(g) for g in groups.values()])
        counts = np.array([len(g) for g in groups.values()])
        rng = np.random.default_rng(27)
        draws = rng.integers(0, len(groups), size=(2000, len(groups)))
        boot = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
        result[metric] = {'official': a, 'native': b, 'native_minus_official': b-a,
                          'paired_record_bootstrap_95pct': np.quantile(boot, [0.025, 0.975]).tolist(),
                          'questions': len(chosen), 'groups': len(groups)}
    return result


def compare(corpus, official, native):
    expected = [c['name'] for c in corpus['cases']]
    if len(set(expected)) != len(expected) or any([c['name'] for c in rows] != expected for rows in (official, native)):
        raise ValueError('missing, duplicate, or reordered cases')
    if corpus['gates'] != GATES:
        raise ValueError('gates changed after freeze')
    results, metrics, errors, boundaries = [], [], [], []
    for case, ref, nat in zip(corpus['cases'], official, native):
        metas = case['meta']
        ids = [m['id'] for m in metas]
        for record in (ref, nat):
            if len(record['probabilities']) != len(metas) or list(record['body']['answers']) != ids:
                raise ValueError(f'question count/order: {case["name"]}')
            if record['body']['usage']['input_tokens'] != case['input_tokens']:
                errors.append({'case': case['name'], 'failure': 'input_tokens'})
        for meta, p, n in zip(metas, ref['probabilities'], nat['probabilities']):
            qid, kind = meta['id'], meta['type']
            row = {'case': case['name'], 'question': qid, 'source': case['source'], 'population': case['population'],
                   **question(meta, p, n, ref['body']['answers'][qid], nat['body']['answers'][qid])}
            if row['failures']:
                errors.append(row)
            if row.get('near_boundary'):
                boundaries.append({**row, 'official_probabilities': p, 'native_probabilities': n,
                                   'official_answer': ref['body']['answers'][qid], 'native_answer': nat['body']['answers'][qid]})
            results.append(row)
            if case['population'] == 'task' and 'distribution' not in row['failures']:
                metrics.append({'case': case['name'], 'question': qid, 'type': kind, 'source': case['source'],
                                'group': case['name'].split(':')[0]+':'+case['group'],
                                'gold': case['labels'][qid],
                                'official': labeled_metrics(kind, p, case['labels'][qid]),
                                'native': labeled_metrics(kind, n, case['labels'][qid])})
    quality = {'all': aggregate(metrics),
               'by_type': {k: aggregate([r for r in metrics if r['type'] == k]) for k in ('choice', 'noul', 'score')},
               'by_source': {k: aggregate([r for r in metrics if r['source'] == k]) for k in sorted({r['source'] for r in metrics})}}
    for group, values in [('all', quality['all']), *quality['by_type'].items()]:
        if not values['questions']:
            errors.append({'failure': 'missing_task_type', 'group': group})
            continue
        limit = GATES['accuracy_drop' if group == 'all' else 'per_type_accuracy_drop']
        if values['accuracy']['native_minus_official'] < -limit:
            errors.append({'failure': 'task_accuracy', 'group': group, 'limit': limit})
    if quality['all'].get('brier', {}).get('native_minus_official', 0) > GATES['brier_increase']:
        errors.append({'failure': 'task_brier'})
    if quality['all'].get('normalized_score_mae', {}).get('native_minus_official', 0) > GATES['normalized_score_mae_increase']:
        errors.append({'failure': 'task_score_mae'})
    counts = collections.Counter(r['type'] for r in boundaries)
    for kind in ('choice', 'noul'):
        if counts[kind] < GATES['minimum_boundary_per_type']:
            errors.append({'failure': 'boundary_coverage', 'type': kind, 'observed': counts[kind]})
    repeats = []
    for records, label in ((official, 'official'), (native, 'native')):
        mixed = next(r for r in records if r['name'] == 'mixed')
        for r in records:
            if r['name'].startswith('mixed-repeat-'):
                error = max(abs(a-b) for p, n in zip(mixed['probabilities'], r['probabilities']) for a, b in zip(p, n))
                same = mixed['body']['answers'] == r['body']['answers']
                repeats.append({'kind': label, 'case': r['name'], 'probability_max_abs': error, 'answers_equal': same})
                if label == 'native' and (error != 0 or not same):
                    errors.append({'failure': 'native_repeat', 'case': r['name']})
    return {'gates': GATES, 'requests': len(expected), 'questions': len(results), 'results': results,
            'boundaries': boundaries, 'boundary_counts': counts, 'task_quality': quality, 'labeled_rows': metrics,
            'repeats': repeats, 'failures': errors, 'passed': not errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('corpus', 'official', 'native', 'reference', 'base', 'out'):
        parser.add_argument('--'+flag, type=Path, required=True)
    for flag in ('supplement-corpus', 'supplement-official', 'supplement-native'):
        parser.add_argument('--'+flag, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('preserve previous report')
    corpus = json.loads(args.corpus.read_text())
    official = json.loads(args.official.read_text())['cases']
    native = read_lines(args.native)
    runs = [official]
    extra = (args.supplement_corpus, args.supplement_official, args.supplement_native)
    if any(extra):
        if not all(extra):
            parser.error('all three supplement files are required together')
        supplement = json.loads(args.supplement_corpus.read_text())
        if supplement['gates'] != GATES:
            raise ValueError('supplement gates differ')
        additional = json.loads(args.supplement_official.read_text())['cases']
        corpus['cases'] += supplement['cases']
        runs.append(additional)
        official = official + additional
        native += read_lines(args.supplement_native)
    report = compare(corpus, official, native)
    fixture = module('kev_fixture', ROOT/'tests/kev/generate_processing_fixture.py')
    api, model = fixture.official(args.reference)
    tok = model.load_tokenizer(args.base)
    for kind, records in (('official', official), ('native', native)):
        for case, record in zip(corpus['cases'], records):
            body = record['body']
            if body['usage']['output_tokens'] != api.output_tokens(tok, body['answers']):
                report['failures'].append({'failure': 'output_tokens', 'kind': kind, 'case': case['name']})
            expected = api.to_answers(record['probabilities'], case['meta'])
            if body['answers'] != expected:
                report['failures'].append({'failure': 'postprocess', 'kind': kind, 'case': case['name'], 'expected': expected, 'actual': body['answers']})
    report['official_execution'] = [{**{k: run[-1][k] for k in ('runtime', 'graphs', 'graph_replays')},
                                     'requests': len(run), 'cache_hits': sum(r['stats']['prefix_cache_hit'] for r in run)} for run in runs]
    if (any(not run[-1]['graph_replays'] or not run[-1]['graphs']['captured'] for run in runs)
            or not sum(r['stats']['prefix_cache_hit'] for r in official)):
        report['failures'].append({'failure': 'official_execution_coverage'})
    for record in official:
        runtime = record['runtime']
        if (runtime['dtype'] != 'bfloat16' or runtime['head_dtype'] != 'torch.float32'
                or runtime['attention'] != 'sdpa' or runtime['fused_available']
                or runtime['remaining_lora_modules'] != 0 or runtime['temperature'] != 2.406050072164233):
            report['failures'].append({'failure': 'official_runtime', 'case': record['name'], 'runtime': runtime})
    report['provenance'] = {k: {'path': str(getattr(args, k)), 'sha256': digest(getattr(args, k))}
                            for k in ('corpus', 'official', 'native', 'supplement_corpus', 'supplement_official', 'supplement_native') if getattr(args, k)}
    report['passed'] = not report['failures']
    save(args.out, report)
    print(json.dumps({'passed': report['passed'], 'requests': report['requests'], 'questions': report['questions'],
                      'boundary_counts': report['boundary_counts'], 'failure_counts': collections.Counter(r.get('failure', f) for r in report['failures'] for f in r.get('failures', ['other']))}, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
