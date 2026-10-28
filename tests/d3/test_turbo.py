"""CPU contract tests: no checkpoint imports, network or accelerator required."""
import hashlib
import http.client
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontend import d3
from models.d3 import engine
from models.d3.protocol import RequestError


def turbo_module():
    from models.d3 import turbo
    return turbo


class FakeTurbo:
    profile = 'turbo'
    model_name = 'd3-turbo'
    max_length = 32768
    modalities = ['text', 'image']
    image_unavailable = None

    def __init__(self):
        self.calls = []
        self.closed = False
        self.response = None

    def load_images(self, images):
        self.calls.append(('images', images))
        return ['decoded']

    def system_one(self, **kwargs):
        self.calls.append(('system_one', kwargs))
        return self.response or {'model': self.model_name, 'answers': {
            name: {'type': q['type'], 'error': 'invalid_question', 'reason': 'invalid_criteria', 'message': 'bad criteria'}
            for name, q in kwargs['questions'].items()}, 'usage': {'input_tokens': 21, 'output_tokens': 0}}

    def close(self):
        self.closed = True


def raw(**changes):
    value = {'state': {'anything': [1, False, None]}, 'questions': {'q': {'type': 'choice', 'criteria': {}}}}
    value.update(changes)
    return json.dumps(value).encode()


def test_cli_profile_is_explicit_and_legacy_by_default():
    assert d3.parse_args(['--model', '/tmp/model']).profile == 'legacy'
    assert d3.parse_args(['--model', '/tmp/model', '--profile', 'turbo']).profile == 'turbo'


def test_turbo_dispatch_preserves_official_per_question_errors_and_usage():
    executor = FakeTurbo()
    result = d3.decide(raw(), executor)
    assert result['answers']['q']['error'] == 'invalid_question'
    assert result['usage']['input_tokens'] == 21
    assert executor.calls == [('system_one', {'state': {'anything': [1, False, None]}, 'questions': {'q': {'type': 'choice', 'criteria': {}}}, 'images': []})]


@pytest.mark.parametrize('field,value', [('assets', {}), ('videos', []), ('long_context', False), ('extra', None)])
def test_unsupported_fields_reject_even_empty(field, value):
    executor = FakeTurbo()
    with pytest.raises(RequestError) as error:
        d3.decide(raw(**{field: value}), executor)
    assert error.value.status == 422
    assert executor.calls == []


@pytest.mark.parametrize('question', [{'type': 'box'}, {'type': 'track'}, {'type': 'unknown'}, {}, {'type': 'choice', 'images': []}, {'type': 'choice', 'instructions': {'type': 'image', 'image_url': '/tmp/a.png'}}])
def test_unsupported_question_media_and_kinds_reject_whole(question):
    with pytest.raises(RequestError):
        d3.decide(raw(questions={'q': question}), FakeTurbo())


@pytest.mark.parametrize('images', [['file.png'], ['https://example.com/a.png'], ['data:image/webp;base64,YQ=='], ['data:image/png;base64,YQ=='] * 2, 'bad'])
def test_unsupported_images_never_reach_runtime(images):
    executor = FakeTurbo()
    with pytest.raises(RequestError):
        d3.decide(raw(images=images), executor)
    assert executor.calls == []


def test_inline_image_strict_loader_and_arbitrary_json_descriptions():
    executor = FakeTurbo()
    q = {'q': {'type': 'choice', 'criteria': {'a': {'image': 'a descriptive property', 'nested': [1, 2]}}}}
    d3.decide(raw(questions=q, images=['data:image/png;base64,YQ==']), executor)
    assert executor.calls[0] == ('images', ['data:image/png;base64,YQ=='])
    assert executor.calls[1][1]['images'] == ['decoded']


@pytest.mark.parametrize('response', [
    {'model': 'x', 'answers': {}, 'usage': {'input_tokens': 0, 'output_tokens': 0}},
    {'model': 'x', 'answers': {'q': {}}, 'usage': {'input_tokens': -1, 'output_tokens': 0}},
    {'model': 'x', 'answers': {'q': {}}, 'usage': {'input_tokens': True, 'output_tokens': 0}},
])
def test_turbo_checks_response_order_and_envelope(response):
    executor = FakeTurbo(); executor.response = response
    with pytest.raises(RuntimeError):
        d3.decide(raw(), executor)


def test_actual_http_server_routes_turbo_and_recovers_after_error():
    executor = FakeTurbo()
    server = d3.WorkerServer(('127.0.0.1', 0), executor)
    thread = threading.Thread(target=server.serve_forever); thread.start()
    try:
        for body, status in [(raw(videos=[]), 422), (raw(), 200)]:
            connection = http.client.HTTPConnection(*server.server_address)
            connection.request('POST', '/v1/systemone', body, {'Content-Type': 'application/json'})
            response = connection.getresponse(); response.read(); connection.close()
            assert response.status == status
    finally:
        server.shutdown(); server.server_close(); thread.join()
    assert executor.closed


def test_manifest_pin_and_no_verification_bypass(tmp_path):
    turbo = turbo_module()
    fixture = Path(__file__).parent / 'fixtures/turbo-manifest.json'
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == turbo.MANIFEST_SHA256
    with pytest.raises(ValueError, match='fast.*full'):
        turbo.verify_package(tmp_path, 'none')
    (tmp_path / engine.MANIFEST_NAME).write_bytes(fixture.read_bytes() + b' ')
    with pytest.raises(ValueError, match='manifest.*pin|pinned.*manifest'):
        turbo.verify_package(tmp_path)


def make_package(tmp_path, monkeypatch):
    turbo = turbo_module()
    files = {'d3_runtime.py': b'# runtime\n', 'modeling_d3.py': b'# wrapper\n', 'd3_compose.py': b'# compose\n', 'config.json': b'{}', 'tokenizer.json': b'{}', 'readout.safetensors': b'abc', 'model-00001.safetensors': b'abc'}
    hashes = {key: hashlib.sha256(value).hexdigest() for key, value in files.items()}
    manifest = {'schema': engine.MANIFEST_SCHEMA, 'repo_id': 'vllm-sr/d3-turbo', 'model_name': 'd3-turbo', 'runtime': {'files_sha256': {name: value for name, value in hashes.items() if name.endswith('.py')}}, 'files_sha256': dict(hashes, **{'README.md': '0' * 64}), 'files_bytes': {key: len(value) for key, value in files.items()}}
    encoded = json.dumps(manifest).encode()
    for name, value in files.items():
        (tmp_path / name).write_bytes(value)
    (tmp_path / engine.MANIFEST_NAME).write_bytes(encoded)
    monkeypatch.setattr(turbo, 'MANIFEST_SHA256', hashlib.sha256(encoded).hexdigest())
    return turbo


@pytest.mark.parametrize('name', ['d3_runtime.py', 'modeling_d3.py', 'd3_compose.py', 'config.json', 'tokenizer.json', 'readout.safetensors'])
def test_every_runtime_and_nonweight_artifact_is_checked_before_import(tmp_path, monkeypatch, name):
    turbo = make_package(tmp_path, monkeypatch)
    (tmp_path / name).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='verification'):
        turbo.verify_package(tmp_path)


def test_official_loaded_files_excludes_stale_readme(tmp_path, monkeypatch):
    turbo = make_package(tmp_path, monkeypatch)
    assert turbo.verify_package(tmp_path)['repo_id'] == 'vllm-sr/d3-turbo'


@pytest.mark.parametrize('name', ['transformers.py', 'torch.pyc', 'hidden.abi3.so', 'processor_config.json'])
def test_turbo_refuses_shadow_imports_and_unlisted_data(tmp_path, monkeypatch, name):
    turbo = make_package(tmp_path, monkeypatch)
    (tmp_path / name).write_text('unlisted')
    with pytest.raises(ValueError, match='unexpected|unlisted'):
        turbo.verify_package(tmp_path)


@pytest.mark.parametrize('first,second', [('legacy', 'turbo'), ('turbo', 'legacy')])
def test_profile_package_isolation_both_directions(tmp_path, monkeypatch, first, second):
    assert hasattr(engine, 'runtime_import_guard'), 'shared runtime import guard is required'
    monkeypatch.setattr(engine, '_RUNTIME_SELECTION', None)
    with engine.runtime_import_guard(tmp_path, first, 'digest', ['d3_runtime.py']):
        pass
    with pytest.raises(RuntimeError, match='one.*profile|selected|restart'):
        with engine.runtime_import_guard(tmp_path, second, 'digest2', ['d3_runtime.py']):
            pass


@pytest.mark.parametrize('name', ['d3_runtime', 'transformers_modules.old.d3_runtime', 'transformers_modules.old.modeling_d3', 'd3_fast_0123456789abcdef'])
def test_preimported_modules_rejected_even_same_origin(tmp_path, monkeypatch, name):
    assert hasattr(engine, 'runtime_import_guard'), 'shared runtime import guard is required'
    monkeypatch.setattr(engine, '_RUNTIME_SELECTION', None)
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(__file__=str(tmp_path / 'd3_runtime.py')))
    with pytest.raises(RuntimeError, match='already imported|preimported'):
        with engine.runtime_import_guard(tmp_path, 'turbo', 'digest', ['d3_runtime.py', 'modeling_d3.py']):
            pass


def fake_loaded_executor(tmp_path, monkeypatch, *, fail_load=False):
    turbo = make_package(tmp_path, monkeypatch)
    monkeypatch.setattr(engine, '_RUNTIME_SELECTION', None)
    events = []

    class Runtime:
        image_unavailable = None
        model_name = 'd3-turbo'

        def load_images(self, images, *, strict):
            events.append(('load_images', strict))
            return [SimpleNamespace(format=None)]

        def warmup(self):
            events.append('warmup')

    runtime = Runtime()

    class Model:
        model_name = 'd3-turbo'
        max_input_tokens = 32768

        def system_one(self, **kwargs):
            events.append(('system_one', kwargs))
            return {'model': runtime.model_name, 'answers': {'q': {'type': 'noul', 'noul': 0.625}}, 'usage': {'input_tokens': 3, 'output_tokens': 0}}

    model = Model(); model.runtime = runtime

    def load(*args, **kwargs):
        events.append(('load', args, kwargs))
        if fail_load:
            raise RuntimeError('failed load')
        return model

    dynamic = SimpleNamespace(HF_MODULES_CACHE='user-owned-cache')
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=load), dynamic_module_utils=dynamic))
    return turbo, model, runtime, events


def test_executor_uses_public_automodel_warmup_then_real_typed_inference(tmp_path, monkeypatch):
    turbo, model, runtime, events = fake_loaded_executor(tmp_path, monkeypatch)
    executor = turbo.D3TurboExecutor(tmp_path, device='cpu', batch_size=3)
    assert executor.model is model and executor.runtime is runtime
    assert events[0][2] == {'trust_remote_code': True, 'local_files_only': True, 'device': 'cpu', 'batch_size': 3, 'verify': 'fast'}
    executor.warmup()
    assert events[1] == 'warmup'
    assert events[2][0] == 'system_one'
    assert events[2][1]['questions'] == {'q': {'type': 'noul'}}
    executor.close(); executor.close()
    assert executor.model is None and executor.runtime is None
    with pytest.raises(RuntimeError, match='closed'):
        executor.system_one(state='', questions={})


def test_failed_load_cannot_be_retried_in_same_process(tmp_path, monkeypatch):
    turbo, _, _, _ = fake_loaded_executor(tmp_path, monkeypatch, fail_load=True)
    with pytest.raises(RuntimeError, match='failed load'):
        turbo.D3TurboExecutor(tmp_path)
    with pytest.raises(RuntimeError, match='restart'):
        turbo.D3TurboExecutor(tmp_path)


def test_repurposed_root_and_dynamic_module_injection_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, '_RUNTIME_SELECTION', None)
    with engine.runtime_import_guard(tmp_path, 'turbo', 'old', ['d3_runtime.py']):
        pass
    with pytest.raises(RuntimeError, match='restart'):
        with engine.runtime_import_guard(tmp_path, 'turbo', 'changed', ['d3_runtime.py']):
            pass
    monkeypatch.setitem(sys.modules, 'transformers_modules.injected.d3_runtime', SimpleNamespace())
    with pytest.raises(RuntimeError, match='preimported'):
        with engine.runtime_import_guard(tmp_path, 'turbo', 'old', ['d3_runtime.py']):
            pass


def test_disguised_webp_rejected_after_runtime_strict_validation(tmp_path, monkeypatch):
    turbo, _, _, events = fake_loaded_executor(tmp_path, monkeypatch)
    executor = turbo.D3TurboExecutor(tmp_path)
    import base64
    disguised = 'data:image/png;base64,' + base64.b64encode(b'RIFFxxxxWEBPpayload').decode()
    with pytest.raises(ValueError, match='PNG.*JPEG'):
        executor.load_images([disguised])
    assert ('load_images', True) in events


def test_main_finishes_warmup_before_bind_and_closes_on_warmup_failure(monkeypatch):
    turbo = turbo_module()
    events = []

    class Executor(FakeTurbo):
        def __init__(self, *args, **kwargs):
            super().__init__(); events.append('load')

        def warmup(self):
            events.append('warmup'); raise RuntimeError('bad warmup')

        def close(self):
            events.append('close')

    monkeypatch.setattr(turbo, 'D3TurboExecutor', Executor)
    monkeypatch.setattr(d3, 'WorkerServer', lambda *args: events.append('bind'))
    with pytest.raises(RuntimeError, match='bad warmup'):
        d3.main(['--model', '/tmp/x', '--profile', 'turbo'])
    assert events == ['load', 'warmup', 'close']


def test_turbo_failure_releases_shared_worker_busy_lock():
    executor = FakeTurbo()
    original = executor.system_one
    executor.system_one = lambda **kwargs: (_ for _ in ()).throw(RuntimeError('private detail'))
    server = d3.WorkerServer(('127.0.0.1', 0), executor)
    thread = threading.Thread(target=server.serve_forever); thread.start()
    try:
        for expected in [500, 200]:
            connection = http.client.HTTPConnection(*server.server_address)
            connection.request('POST', '/v1/systemone', raw(), {'Content-Type': 'application/json'})
            response = connection.getresponse(); body = response.read(); connection.close()
            assert response.status == expected
            if expected == 500:
                assert b'private detail' not in body
                executor.system_one = original
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_private_dynamic_cache_is_clean_scoped_and_lifecycle_owned(tmp_path, monkeypatch):
    turbo, _, _, _ = fake_loaded_executor(tmp_path, monkeypatch)
    dynamic = sys.modules['transformers'].dynamic_module_utils
    old = sys.modules['transformers'].AutoModel.from_pretrained
    observed = []
    def load(*args, **kwargs):
        directory = Path(dynamic.HF_MODULES_CACHE)
        observed.append(directory)
        assert directory.is_dir() and not list(directory.rglob('*.pyc'))
        assert str(directory) in sys.path
        return old(*args, **kwargs)
    sys.modules['transformers'].AutoModel.from_pretrained = load
    executor = turbo.D3TurboExecutor(tmp_path)
    assert dynamic.HF_MODULES_CACHE == 'user-owned-cache'
    assert observed[0].exists()
    executor.close()
    assert not observed[0].exists()
    assert str(observed[0]) not in sys.path


def test_private_cache_is_cleaned_when_loading_fails(tmp_path, monkeypatch):
    turbo, _, _, _ = fake_loaded_executor(tmp_path, monkeypatch)
    dynamic = sys.modules['transformers'].dynamic_module_utils
    observed = []
    def load(*args, **kwargs):
        observed.append(Path(dynamic.HF_MODULES_CACHE))
        raise RuntimeError('failed load')
    sys.modules['transformers'].AutoModel.from_pretrained = load
    with pytest.raises(RuntimeError, match='failed load'):
        turbo.D3TurboExecutor(tmp_path)
    assert dynamic.HF_MODULES_CACHE == 'user-owned-cache'
    assert not observed[0].exists()
    assert str(observed[0]) not in sys.path


def test_fast_checks_large_weight_size_full_checks_weight_hash(tmp_path, monkeypatch):
    turbo = make_package(tmp_path, monkeypatch)
    shard = tmp_path / 'model-00001.safetensors'
    with shard.open('wb') as stream:
        stream.truncate((64 << 20) + 1)
    manifest_path = tmp_path / engine.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest['files_bytes'][shard.name] = shard.stat().st_size
    # Deliberately wrong hash. Fast mode does not hash a large shard; full
    # must catch corruption even when the size agrees with the pinned table.
    encoded = json.dumps(manifest).encode(); manifest_path.write_bytes(encoded)
    monkeypatch.setattr(turbo, 'MANIFEST_SHA256', hashlib.sha256(encoded).hexdigest())
    turbo.verify_package(tmp_path, 'fast')
    with pytest.raises(ValueError, match='SHA-256'):
        turbo.verify_package(tmp_path, 'full')
    with shard.open('ab') as stream:
        stream.write(b'x')
    with pytest.raises(ValueError, match='size'):
        turbo.verify_package(tmp_path, 'fast')


def test_preimport_guard_is_used_by_legacy_executor(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, '_RUNTIME_SELECTION', None)
    manifest = {'runtime': {'files_sha256': {'d3_runtime.py': 'x'}}}
    monkeypatch.setattr(engine, 'load_manifest', lambda root: manifest)
    monkeypatch.setattr(engine, 'verify_pinned_package', lambda manifest: None)
    monkeypatch.setattr(engine, 'verify_runtime_files', lambda *args: [])
    monkeypatch.setattr(engine, '_sha256_safe', lambda path: 'digest')
    monkeypatch.setitem(sys.modules, 'transformers_modules.old.modeling_d3', SimpleNamespace())
    with pytest.raises(RuntimeError, match='already imported'):
        engine.D3Executor(tmp_path)


def test_exact_public_response_is_not_recomputed_or_calibrated():
    executor = FakeTurbo()
    executor.response = {'model': 'd3-turbo', 'answers': {'q': {'type': 'choice', 'choice': 'b', 'probabilities': {'a': 0.125, 'b': 0.875}, 'confidence': 0.349102641}}, 'usage': {'input_tokens': 139, 'output_tokens': 0}}
    assert d3.decide(raw(), executor) is executor.response
