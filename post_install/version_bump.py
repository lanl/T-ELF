#!/usr/bin/env python3
"""
Version bump + optional docs rebuild for TELF.

Examples:
  # set explicit version and preview the changes
  telf-version bump --new 0.0.45 --preview

  # bump patch (reads current from pyproject.toml), write changes, rebuild docs
  telf-version bump --bump patch --rebuild-docs

  # strict mode (error if a target file had no replacement)
  telf-version bump --new 0.0.45 --strict
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

# Python 3.11 has stdlib tomllib
try:
    import tomllib  # type: ignore
except Exception:  # pragma: no cover
    tomllib = None

# Files & line prefixes to update (your original list)
UPDATE_LOCATIONS: Dict[str, Dict[str, str]] = {
    "CITATION.cff":           {"match": "version: ",     "type": "float"},
    "setup.py":               {"match": "__version__ = ", "type": "str"},
    "pyproject.toml":         {"match": "version = ",     "type": "str"},
    "TELF/version.py":        {"match": "__version__ = ", "type": "str"},
    "docs/source/conf.py":    {"match": "release = ",     "type": "str"},
}

DOCS_DEPS = [
    "sphinx", "sphinx-book-theme", "sphinxcontrib-bibtex", "sphinx-automodapi"
]

def read_current_version_from_pyproject(root: Path) -> str:
    """Read [tool.poetry].version from pyproject.toml."""
    pyproj = root / "pyproject.toml"
    if not pyproj.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproj}")
    if tomllib is None:
        raise RuntimeError("tomllib not available; need Python 3.11+")
    with pyproj.open("rb") as f:
        data = tomllib.load(f)
    try:
        return str(data["tool"]["poetry"]["version"])
    except Exception as e:
        raise KeyError("Could not read tool.poetry.version from pyproject.toml") from e

def bump_semver(v: str, which: str) -> str:
    """
    Bump major/minor/patch for a simple semantic version 'X.Y.Z'.
    Keeps pre-release/build metadata off; you can add as needed.
    """
    parts = v.split(".")
    if len(parts) < 3 or not all(p.isdigit() for p in parts[:3]):
        raise ValueError(f"Unsupported version format for bump: {v}")
    major, minor, patch = map(int, parts[:3])
    if which == "major":
        return f"{major+1}.0.0"
    if which == "minor":
        return f"{major}.{minor+1}.0"
    if which == "patch":
        return f"{major}.{minor}.{patch+1}"
    raise ValueError(f"Unknown bump type: {which}")

def _replace_line(line: str, match: str, typ: str, new_version: str) -> Tuple[str, bool]:
    """Replace version in a single line; tries robust regex first, then falls back."""
    if match not in line:
        return line, False

    if typ == "str":
        # e.g., version = "0.0.44"  OR  __version__ = '0.0.44'
        pat = re.compile(rf"({re.escape(match)}\s*)(['\"])(.*?)\2")
        m = pat.search(line)
        if m:
            start, end = m.span(3)
            return line[:start] + new_version + line[end:], True
        # fallback: write quoted value after match
        idx = line.find(match) + len(match)
        prefix = line[:idx]
        suffix = "" if line.endswith("\n") else "\n"
        return prefix + f"\"{new_version}\"" + suffix, True

    elif typ == "float":
        # e.g., version: 0.0.44  (YAML in CITATION.cff)
        pat = re.compile(rf"({re.escape(match)}\s*)([0-9]+(?:\.[0-9]+)*)")
        m = pat.search(line)
        if m:
            start, end = m.span(2)
            return line[:start] + new_version + line[end:], True
        # fallback: write unquoted number after match
        idx = line.find(match) + len(match)
        prefix = line[:idx]
        suffix = "" if line.endswith("\n") else "\n"
        return prefix + new_version + suffix, True

    else:
        raise ValueError(f"Unsupported type specifier: {typ}")

def update_version_in_files(root_dir: Path,
                            update_locations: Dict[str, Dict[str, str]],
                            new_version: str,
                            preview: bool = False,
                            strict: bool = False,
                            make_backup: bool = True) -> None:
    """
    Iterate target files, replacing version strings.
    - preview: print planned changes but do not write
    - strict:  error if any file matched exists but had 0 replacements
    - make_backup: save .bak files before writing
    """
    print(f"Updating version to {new_version} in {len(update_locations)} files...")
    any_errors = False
    for rel_path, settings in update_locations.items():
        full_path = (root_dir / rel_path).resolve()
        if not full_path.exists():
            print(f"[skip] not found: {full_path}")
            continue

        match = settings["match"]
        typ = settings["type"]

        try:
            original = full_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[error] cannot read {full_path}: {e}")
            any_errors = True
            continue

        new_lines = []
        replaced_count = 0
        for line in original.splitlines(keepends=True):
            nline, changed = _replace_line(line, match, typ, new_version)
            new_lines.append(nline)
            if changed:
                replaced_count += 1

        if replaced_count == 0:
            msg = f"[warn] no replacements made in {full_path}"
            if strict:
                print(msg)
                any_errors = True
            else:
                print(msg)

        if preview:
            print(f"[preview] {full_path}: {replaced_count} line(s) would change")
            continue

        new_content = "".join(new_lines)
        if new_content != original:
            if make_backup:
                try:
                    shutil.copy2(full_path, full_path.with_suffix(full_path.suffix + ".bak"))
                except Exception as e:
                    print(f"[warn] failed to create backup for {full_path}: {e}")
            try:
                full_path.write_text(new_content, encoding="utf-8")
                print(f"[ok] updated {full_path} ({replaced_count} line(s))")
            except Exception as e:
                print(f"[error] cannot write {full_path}: {e}")
                any_errors = True
        else:
            print(f"[ok] unchanged {full_path}")

    if any_errors and strict and not preview:
        raise SystemExit("Errors/warnings occurred with --strict; aborting.")

def rebuild_docs(root_dir: Path, install_deps: bool = False, clean: bool = True) -> None:
    """
    Rebuild Sphinx docs similar to your shell snippet,
    but using Python stdlib for portability.
    """
    docs_dir = root_dir / "docs"
    if not docs_dir.exists():
        print(f"[docs] no docs/ directory at {docs_dir}; skipping.")
        return

    if install_deps:
        print("[docs] installing Sphinx dependencies...")
        cmd = [sys.executable, "-m", "pip", "install", *DOCS_DEPS]
        subprocess.run(cmd, check=True)

    # Optional: ensure project is installed in editable mode for autodoc imports
    # subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(root_dir)], check=True)

    # make clean / make html
    build_dir = docs_dir / "build"
    if clean and build_dir.exists():
        shutil.rmtree(build_dir)

    print("[docs] building HTML…")
    # Prefer 'make html' if Makefile exists; otherwise use sphinx-build directly
    makefile = docs_dir / "Makefile"
    if makefile.exists():
        subprocess.run(["make", "html"], cwd=str(docs_dir), check=True)
    else:
        # Fallback: sphinx-build -b html source build/html
        src = docs_dir / "source"
        out = build_dir / "html"
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(["sphinx-build", "-b", "html", str(src), str(out)], check=True)

    # Copy build/html/* into docs root (and doctrees) like your snippet
    html_src = build_dir / "html"
    if html_src.exists():
        print("[docs] copying built HTML into docs/ …")
        for item in html_src.iterdir():
            dest = docs_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    doctrees_src = build_dir / "doctrees"
    if doctrees_src.exists():
        shutil.copytree(doctrees_src, docs_dir / "doctrees", dirs_exist_ok=True)

    print("[docs] done.")

def _in_git_repo(path: Path) -> bool:
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                       cwd=str(path), stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def git_commit_and_tag(root_dir: Path, new_version: str, do_tag: bool) -> None:
    if not _in_git_repo(root_dir):
        print("[git] not a git repo; skipping commit/tag.")
        return
    try:
        subprocess.run(["git", "add", "-A"], cwd=str(root_dir), check=True)
        subprocess.run(["git", "commit", "-m", f"Bump version to {new_version}"], cwd=str(root_dir), check=True)
        if do_tag:
            subprocess.run(["git", "tag", f"v{new_version}"], cwd=str(root_dir), check=True)
        print("[git] committed version bump" + (" and tagged" if do_tag else ""))
    except subprocess.CalledProcessError as e:
        print(f"[git] warning: {e}")

def main(argv=None):
    parser = argparse.ArgumentParser(description="TELF version bumper + docs helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("bump", help="Update version strings across project files")
    p.add_argument("--root", type=Path, default=Path.cwd(),
                   help="Project root (default: current working dir)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--new", dest="new_version", help="Explicit version, e.g. 0.0.45")
    group.add_argument("--bump", choices=["major", "minor", "patch"], help="Semver bump relative to pyproject.toml")
    p.add_argument("--preview", action="store_true", help="Show changes without writing")
    p.add_argument("--strict", action="store_true", help="Error if any target file received no replacement")
    p.add_argument("--no-backup", action="store_true", help="Do not create .bak files")
    p.add_argument("--rebuild-docs", action="store_true", help="Rebuild Sphinx docs after bump")
    p.add_argument("--install-docs-deps", action="store_true", help="pip install Sphinx deps before building docs")
    p.add_argument("--git-commit", action="store_true", help="git commit the changes")
    p.add_argument("--git-tag", action="store_true", help="git tag v<version> after commit (implies --git-commit)")

    args = parser.parse_args(argv)

    root = args.root.resolve()

    if args.new_version:
        new_version = args.new_version
    else:
        current = read_current_version_from_pyproject(root)
        new_version = bump_semver(current, args.bump)

    update_version_in_files(
        root,
        UPDATE_LOCATIONS,
        new_version,
        preview=args.preview,
        strict=args.strict,
        make_backup=not args.no_backup,
    )

    if not args.preview and args.rebuild_docs:
        rebuild_docs(root, install_deps=args.install_docs_deps, clean=True)

    if not args.preview and (args.git_commit or args.git_tag):
        git_commit_and_tag(root, new_version, do_tag=args.git_tag)

if __name__ == "__main__":
    main()
