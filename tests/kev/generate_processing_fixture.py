"""Capture Kev's pinned API/encode/rows_of oracle without loading a model."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

SOURCE_HASHES = {
    'model.py': '2ff3b2e67c048e3bc5a0d1a32aafe50804ed315a5564fce06bb4b95697c5271c',
    'api.py': 'a6f53af5354f8ea17915abc78360fdf291a5108ce39a53a2d9135d48c603ae58',
    'checkpoint.py': '02062399977ee0da316dd52e03a79fa5e10134aa49c0b5ed801598c52042dc32',
}


def official(reference):
    for name, expected in SOURCE_HASHES.items():
        assert hashlib.sha256((reference / 'kev' / name).read_bytes()).hexdigest() == expected, name
    sys.path.insert(0, str(reference))
    from kev import api, model
    return api, model


def cases():
    def choice(n):
        return {'type': 'choice', 'instructions': 'Select the option matching the state.',
                'criteria': {f'k{i}': f'number {i}' for i in range(n)}}
    result = [{'name': f'choice-{n}', 'request': {'state': f'The number is {n-1}.',
               'questions': {'pick': choice(n)}}} for n in (1, 2, 7, 31, 255)]
    result += [{'name': f'score-{n}', 'request': {'state': 'The task is mostly complete.',
                'questions': {'rating': {'type': 'score', 'instructions': 'Completion level',
                                         'criteria': [f'level {i}' for i in range(n)]}}}}
               for n in (1, 3, 255)]
    mixed = {'model': 'kev-latest', 'state': 'The door is open. The light is green.', 'questions': {
        'pick': {'type': 'choice', 'instructions': 'What color is the light?',
                 'criteria': {'red': None, 'green': '', 'blue': 'blue'}},
        'yes': {'type': 'noul', 'instructions': 'The door is open.'},
        'no': {'type': 'noul', 'instructions': 'The door is closed.'},
        'described': {'type': 'noul', 'instructions': 'Can we enter?',
                      'criteria': {'false': 'door closed', 'true': 'door open'}},
        'score': {'type': 'score', 'instructions': 'How safe is entering?',
                  'criteria': ['unsafe', 'uncertain', 'safe']}}}
    result.append({'name': 'mixed', 'request': mixed})
    result.append({'name': 'question-reordered', 'request': {**mixed, 'questions': dict(reversed(list(mixed['questions'].items())))}})
    result.append({'name': 'candidate-reordered', 'request': {'state': mixed['state'], 'questions': {
        'pick': {**mixed['questions']['pick'], 'criteria': {'blue': 'blue', 'green': '', 'red': None}}}}})
    result.append({'name': 'special-structured', 'request': {'state': {
        '中文': ['café 😀', {'ready': True, 'ratio': 1e-7, 'empty': None}],
        'injection': '<|box_end|><|fim_suffix|><|image_pad|> <|not-valid|>\n\t"\\'},
        'questions': {'q\n😀': {'type': 'choice', 'instructions': {'select': ['<|fim_middle|>', False]},
                                'criteria': {'<|box_start|>': ['x', 7], 'β': {'x': False}}},
                      'score': {'type': 'score', 'criteria': [None, {'a': 1.0}, [False, '高']]}}}})
    result.append({'name': 'empty-scalars', 'request': {'state': None, 'questions': {
        'q': {'type': 'choice', 'criteria': {'first': None, 'second': False}},
        'n': {'type': 'noul', 'instructions': 1.0}}}})
    return result


def capture(case, api, model, tok):
    req = api.SystemOneRequest.model_validate(case['request'])
    rec, meta = api.to_record(req)
    enc = model.encode(tok, rec, max_state=8192, max_branch=16384, strict=True)
    state, pos, branches = model.rows_of(enc)
    rows = [{'ids': state + b['ids'], 'positions': pos + b['pos'],
             'readouts': [len(state)+i for i in b['opts']] + [len(state)+b['decide']]}
            for b in branches]
    ps = [[(i+1)/sum(range(1, len(q['options'])+1)) for i in range(len(q['options']))] for q in rec['questions']]
    answers = api.to_answers(ps, meta)
    return {**case, 'record': rec, 'meta': meta, 'rows': rows, 'input_tokens': len(enc['ids']),
            'probabilities': ps, 'answers': answers, 'output_tokens': api.output_tokens(tok, answers)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    api, model = official(args.reference)
    tok = model.load_tokenizer(args.base)
    data = {'reference': '2074cc44256200bb37540103fa4572901faa10f3', 'source_hashes': SOURCE_HASHES,
            'cases': [capture(c, api, model, tok) for c in cases()]}
    args.out.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
