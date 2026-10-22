#!/usr/bin/env python3
"""Tier-2 build of every declared CUDA backend. No GPU and no self-hosted runner.

Discovers every ``*.backend.json`` under ``src/backends/cuda/``, runs the backend's
own build script with an output directory and the compute capability the manifest
declares, and requires the library the manifest names. Nothing here runs what it
builds: nvcc cross-compiles for a target it is not running on, and the toolkit is
the caller's to supply.

One backend cannot hide another:

  * a manifest that will not parse, or one that is not an object, is reported
    against its directory and the sweep continues,
  * a ``planned`` backend is skipped the way ``contract.md`` says the checker
    skips it -- the directory is declared and nothing is built,
  * a missing ``build`` block or field, a missing or non-executable script, a
    non-zero exit and a missing library are each reported against that backend,
  * every remaining backend still builds, and the run fails once at the end with
    the whole list rather than at the first defect.

Each build writes into its own directory, keyed by position as well as by name,
so two manifests that share a ``name`` and an output filename cannot pass on each
other's artifact.

It does not compile for more than ``build.default_arch``; the later tiers and the
architectures a backend really ships are the backend owner's to state.

Usage:
    python3 src/backends/cuda/build_backends.py [--repo-root PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

MANIFEST_SUFFIX = ".backend.json"
BACKENDS_DIR = os.path.join("src", "backends", "cuda")
# contract.md: "planned -- directory only; the checker skips everything else."
SKIPPED_STATUS = "planned"


class Outcome:
    """What happened to one manifest."""

    def __init__(self, label, directory, detail=""):
        self.label = label
        self.directory = directory
        self.detail = detail

    def describe(self):
        if self.detail:
            return "%s (%s): %s" % (self.label, self.directory, self.detail)
        return "%s (%s)" % (self.label, self.directory)


def discover(repo_root):
    """Return the manifest paths, in a stable order.

    Both layouts are accepted: a backend may be a subdirectory named after
    itself, or sit directly in ``src/backends/cuda/`` the way Laya's ``kernels/``
    and ``tools/`` do. Only the manifest's contents are fixed.
    """
    root = os.path.join(repo_root, BACKENDS_DIR)
    return sorted(glob.glob(os.path.join(root, "*" + MANIFEST_SUFFIX))
                  + glob.glob(os.path.join(root, "*", "*" + MANIFEST_SUFFIX)))


def _safe(label):
    """A directory component that keeps distinct labels distinct."""
    kept = "".join(char if char.isalnum() or char in "-_." else "_" for char in label)
    return kept.strip(".") or "backend"


def _fields(build):
    """Return (script, output, arch, problem) for a build block."""
    script = build.get("script")
    if not isinstance(script, str) or not script:
        return None, None, None, "build.script is missing or not a string"
    output = build.get("output")
    if not isinstance(output, str) or not output:
        return None, None, None, "build.output is missing or not a string"
    arch = build.get("default_arch")
    if not isinstance(arch, int) or isinstance(arch, bool):
        return None, None, None, "build.default_arch is missing or not an integer"
    return script, output, arch, None


def build_all(repo_root, out_dir):
    """Build every declared backend.

    Returns ``(built, skipped, failed)``, each a list of :class:`Outcome`. Every
    manifest is attempted; the caller decides what a non-empty ``failed`` means.
    """
    built, skipped, failed = [], [], []
    for index, path in enumerate(discover(repo_root)):
        directory = os.path.dirname(os.path.abspath(path))
        label = os.path.basename(path)[: -len(MANIFEST_SUFFIX)]
        shown = os.path.relpath(directory, os.path.abspath(repo_root))

        try:
            with open(path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, ValueError) as error:
            failed.append(Outcome(label, shown, "%s cannot be read: %s"
                                  % (os.path.basename(path), error)))
            continue
        if not isinstance(manifest, dict):
            failed.append(Outcome(label, shown,
                                  "%s must contain a JSON object" % os.path.basename(path)))
            continue

        name = manifest.get("name")
        if isinstance(name, str) and name:
            label = name

        if manifest.get("status") == SKIPPED_STATUS:
            skipped.append(Outcome(label, shown, "planned; nothing is built"))
            continue

        build = manifest.get("build")
        if not isinstance(build, dict):
            failed.append(Outcome(label, shown, "no build block"))
            continue
        script_name, output, arch, problem = _fields(build)
        if problem is not None:
            failed.append(Outcome(label, shown, problem))
            continue

        script = os.path.join(directory, script_name)
        if not os.path.isfile(script):
            failed.append(Outcome(label, shown, "build script %s does not exist" % script_name))
            continue
        if not os.access(script, os.X_OK):
            failed.append(Outcome(label, shown, "build script %s is not executable" % script_name))
            continue

        # Keyed by position as well as name: two manifests may declare the same
        # backend name and the same library filename, and neither may be credited
        # with the other's artifact.
        target = os.path.join(out_dir, "%03d-%s" % (index, _safe(label)))
        os.makedirs(target, exist_ok=True)
        print("== %s: %s -> %s (sm_%s)" % (shown, script_name, output, arch), flush=True)
        try:
            code = subprocess.run([script, target, str(arch)], cwd=directory).returncode
        except OSError as error:
            failed.append(Outcome(label, shown, "could not run %s: %s" % (script_name, error)))
            continue
        if code != 0:
            failed.append(Outcome(label, shown, "%s exited %d" % (script_name, code)))
            continue
        if not os.path.isfile(os.path.join(target, output)):
            failed.append(Outcome(label, shown, "%s was not produced" % output))
            continue
        built.append(Outcome(label, shown))
    return built, skipped, failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=".",
                        help="repository root to build (default: current directory)")
    parser.add_argument("--out", default=None,
                        help="output directory, created if absent (default: a temporary one)")
    args = parser.parse_args(argv)

    repo_root = os.path.abspath(args.repo_root)
    if not os.path.isdir(os.path.join(repo_root, BACKENDS_DIR)):
        print("%s does not exist" % BACKENDS_DIR, file=sys.stderr)
        return 1
    out_dir = os.path.abspath(args.out or tempfile.mkdtemp(prefix="cuda-build-"))
    os.makedirs(out_dir, exist_ok=True)

    built, skipped, failed = build_all(repo_root, out_dir)
    if not (built or skipped or failed):
        print("no backend manifests under %s" % BACKENDS_DIR, file=sys.stderr)
        return 1

    for outcome in built:
        print("built    %s" % outcome.describe())
    for outcome in skipped:
        print("skipped  %s" % outcome.describe())
    for outcome in failed:
        print("FAILED   %s" % outcome.describe(), file=sys.stderr)
    print("\n%d built, %d skipped, %d failed" % (len(built), len(skipped), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
