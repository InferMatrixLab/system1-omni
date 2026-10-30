"""Pinned d3-turbo AutoModel executor; serving stays in the existing D3 worker.

All executable and non-weight load artifacts hash before importing any package
code. Fast mode checks shard sizes and hashes shards up to 64 MiB; full mode
also hashes every large shard. Use full for first-time package admission:
fast cannot detect same-size corruption in large weight shards.
"""
from __future__ import annotations

import base64
import json
import math
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from .engine import MANIFEST_NAME, MANIFEST_SCHEMA, _sha256_safe, runtime_import_guard, verify_runtime_files
from .turbo_protocol import finish_response

REFERENCE_REPO = 'vllm-sr/d3-turbo'
REFERENCE_REVISION = 'a0f3e9e59ebd1202fe1d4de1c4c0f9a986758cbb'
MANIFEST_SHA256 = '54a65af0871c21b3440cbdadaccfd752a8e2173b86334d1b3290f005e49699fe'
VERIFY_MODES = ('fast', 'full')
_NOT_LOADED = {'LICENSE', 'NOTICE', 'requirements.txt', '.gitattributes'}


def verify_package(root: Path, verify: str = 'fast') -> dict[str, Any]:
    """Authenticate manifest/load inputs; large weight hashes require full mode."""
    if verify not in VERIFY_MODES:
        raise ValueError('turbo verify must be fast or full')
    root = Path(root)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file() or _sha256_safe(manifest_path) != MANIFEST_SHA256:
        raise ValueError('turbo manifest does not match the pinned manifest SHA-256')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema') != MANIFEST_SCHEMA or manifest.get('repo_id') != REFERENCE_REPO:
        raise ValueError('turbo manifest has an unexpected package identity')
    verify_runtime_files(root, manifest)
    listed = manifest['files_sha256']
    problems = []
    for name, digest in sorted(listed.items()):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'unsafe package file name {name!r}')
        if name in _NOT_LOADED or name.endswith('.md') or name.startswith(('assets/', 'LICENSES/')):
            continue
        target = root / relative
        if not target.is_file():
            problems.append(f'missing {name}')
            continue
        size = target.stat().st_size
        if size != manifest['files_bytes'][name]:
            problems.append(f'{name}: size differs from manifest')
            continue
        weight = name.startswith('model-') and name.endswith('.safetensors')
        if not weight or verify == 'full' or size <= 64 << 20:
            if _sha256_safe(target) != digest:
                problems.append(f'{name}: SHA-256 differs from manifest')
    for entry in root.iterdir():
        if entry.is_file() and entry.name not in listed and entry.name != MANIFEST_NAME and not entry.name.startswith('.'):
            problems.append(f'unlisted package file {entry.name}')
    if problems:
        raise ValueError('turbo package verification failed: ' + '; '.join(problems[:8]))
    return manifest


class D3TurboExecutor:
    """One public AutoModel with the bundled state-once runtime semantics."""
    profile = 'turbo'

    def __init__(self, package: str | Path, *, device: str | None = None,
                 batch_size: int = 8, verify: str = 'fast', model_name: str | None = None):
        root = Path(package).resolve()
        manifest = verify_package(root, verify)
        self._lock = threading.RLock()
        self._closed = False
        with runtime_import_guard(root, self.profile, MANIFEST_SHA256, list(manifest['runtime']['files_sha256'])):
            from transformers import AutoModel, dynamic_module_utils
            # AutoModel imports copied sources from this module constant (setting
            # an environment variable after transformers was imported is too
            # late). A clean private cache prevents a stale cached .pyc or a
            # same-size/mtime source copy from replacing verified package code.
            self._module_cache = tempfile.TemporaryDirectory(prefix="d3-turbo-modules-")
            cache = self._module_cache.name
            saved_cache = dynamic_module_utils.HF_MODULES_CACHE
            namespace = sys.modules.get("transformers_modules")
            saved_namespace_path = getattr(namespace, "__path__", None)
            if saved_namespace_path is not None:
                namespace.__path__ = [str(Path(cache) / "transformers_modules"), *saved_namespace_path]
            dynamic_module_utils.HF_MODULES_CACHE = cache
            Path(cache, "__init__.py").touch()
            Path(cache, "transformers_modules").mkdir()
            Path(cache, "transformers_modules", "__init__.py").touch()
            sys.path.insert(0, cache)
            try:
                self.model = AutoModel.from_pretrained(
                    str(root), trust_remote_code=True, local_files_only=True,
                    device=device, batch_size=batch_size, verify=verify,
                )
                self.runtime = self.model.runtime
            except BaseException:
                self._module_cache.cleanup()
                raise
            finally:
                dynamic_module_utils.HF_MODULES_CACHE = saved_cache
                sys.path[:] = [entry for entry in sys.path if entry != cache]
                if saved_namespace_path is not None:
                    namespace.__path__ = saved_namespace_path
        if model_name is not None:
            self.runtime.model_name = model_name
        self.model_name = self.model.model_name
        self.max_length = self.model.max_input_tokens
        self.image_unavailable = self.runtime.image_unavailable

    @property
    def modalities(self):
        return ['text', 'image'] if self.image_unavailable is None else ['text']

    def _require_open(self):
        if self._closed:
            raise RuntimeError('D3 turbo executor is closed')

    def load_images(self, images):
        with self._lock:
            self._require_open()
            decoded = self.runtime.load_images(images, strict=True)
            # RGB conversion removes PIL's format marker. Inspect only the
            # signature after the runtime has strictly checked the full payload,
            # integrity and byte/pixel budgets; a PNG header cannot admit WebP.
            for image in images:
                signature = base64.b64decode(image.partition(",")[2][:16])
                if not (signature.startswith(b"\x89PNG\r\n\x1a\n") or signature.startswith(b"\xff\xd8\xff")):
                    raise ValueError("turbo images must be PNG or JPEG")
            return decoded

    def system_one(self, *, state, questions, images=None):
        with self._lock:
            self._require_open()
            return self.model.system_one(state=state, questions=questions, images=images)

    def warmup(self):
        with self._lock:
            self._require_open()
            self.runtime.warmup()
            result = self.model.system_one(state='warm-up', questions={'q': {'type': 'noul'}})
            finish_response(('q',), result)
            answer = result['answers']['q']
            if (answer.get('type') != 'noul' or 'error' in answer
                    or type(answer.get('noul')) not in (int, float)
                    or not math.isfinite(answer['noul']) or not 0 <= answer['noul'] <= 1):
                raise RuntimeError('turbo readiness inference did not produce a typed answer')

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.model = None
            self.runtime = None
            self._module_cache.cleanup()
