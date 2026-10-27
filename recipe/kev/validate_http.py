"""Start the real Kev worker and frontend; test readiness, parity and HTTP failures.

Run the entire script under gpu run; child processes are always stopped before exit.
"""
import argparse
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def request(url, value=None, raw=None, content_type='application/json'):
    data = raw if raw is not None else (json.dumps(value, ensure_ascii=False).encode() if value is not None else None)
    req = urllib.request.Request(url, data=data, headers={'Content-Type': content_type})
    try:
        with HTTP.open(req, timeout=180) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def ready(process, url):
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'server exited {process.returncode}; see captured log')
        try:
            status, body = request(url+'/health')
            if status == 200:
                return body
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    raise TimeoutError('worker readiness deadline')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('worker-bin', 'frontend-bin', 'export', 'cuda-lib', 'out', 'native'):
        p.add_argument('--'+flag, type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    worker_port, frontend_port = port(), port()
    worker_url, frontend_url = f'http://127.0.0.1:{worker_port}', f'http://127.0.0.1:{frontend_port}'
    env = {**os.environ, 'KEV_MODEL': str(args.export), 'KEV_CUDA_LIB': str(args.cuda_lib), 'KEV_PORT': str(worker_port),
           'KEV_HOST': '127.0.0.1', 'OMNI_JEV_BIND': f'127.0.0.1:{frontend_port}', 'OMNI_JEV_BACKEND_URL': worker_url}
    corpus = json.loads((ROOT/'tests/kev/data/processing.json').read_text())['cases']
    native = {case['name']: case['body'] for case in json.loads(args.native.read_text())['cases']}
    processes = []
    report = {'cases': [], 'errors': []}
    started = time.monotonic()
    with (args.out/'worker.log').open('w') as wlog, (args.out/'frontend.log').open('w') as flog:
        try:
            worker = subprocess.Popen([str(args.worker_bin)], env=env, stdout=wlog, stderr=subprocess.STDOUT)
            processes.append(worker)
            report['worker_health'] = ready(worker, worker_url)
            report['load_and_real_warmup_seconds'] = time.monotonic()-started
            # The very first external request after readiness includes all question types.
            mixed = next(c for c in corpus if c['name'] == 'mixed')
            status, first = request(worker_url+'/v1/systemone', mixed['request'])
            assert status == 200 and first == native['mixed'], (status, first)
            report['first_after_ready_exact'] = True
            frontend = subprocess.Popen([str(args.frontend_bin)], env=env, stdout=flog, stderr=subprocess.STDOUT)
            processes.append(frontend)
            report['frontend_health'] = ready(frontend, frontend_url)
            for case in corpus + [mixed]*3:
                ds, direct = request(worker_url+'/v1/systemone', case['request'])
                fs, front = request(frontend_url+'/v1/systemone', case['request'])
                assert ds == fs == 200 and direct == front == native[case['name']], case['name']
                report['cases'].append({'name': case['name'], 'direct': direct, 'frontend': front})
            invalid = [
                ('malformed', 400, b'{', 'application/json'),
                ('duplicate', 400, b'{"state":1,"state":2}', 'application/json'),
                ('media-type', 415, b'{}', 'text/plain'),
                ('oversize', 413, b' '*(4*1024*1024+1), 'application/json'),
                ('unknown-model', 422, json.dumps({**mixed['request'], 'model': 'other'}).encode(), 'application/json'),
                ('256-options', 422, json.dumps({'state': '', 'questions': {'q': {'type': 'score', 'criteria': ['x']*256}}}).encode(), 'application/json'),
                ('empty-questions', 422, b'{"state":"", "questions":{}}', 'application/json'),
                ('oversize-state', 422, json.dumps({'state': 'x '*9000, 'questions': {'q': {'type': 'noul'}}}).encode(), 'application/json'),
            ]
            for name, expected, raw, content_type in invalid:
                ds, direct = request(worker_url+'/v1/systemone', raw=raw, content_type=content_type)
                fs, front = request(frontend_url+'/v1/systemone', raw=raw, content_type=content_type)
                assert ds == fs == expected and direct == front, (name, ds, fs, direct, front)
                report['errors'].append({'name': name, 'status': ds, 'body': direct})
            report['passed'] = True
        finally:
            for process in reversed(processes):
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            (args.out/'http.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print('direct/frontend/native equality:', len(report['cases']), 'requests; HTTP errors:', len(report['errors']))


if __name__ == '__main__':
    main()
