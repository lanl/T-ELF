#!/usr/bin/env python3

# EXAMPLE USAGE
# python archiver.py pack some_HNMFK_dir/ ./backups/some_HNMFK_dir.tar.gz --method tar-gz --level 6 --split --split-size 1.5G
# python archiver.py unpack ./backups/some_HNMFK_dir.tar.gz ./restore

"""
archiver.py — strong compression + progress + root-folder packing + optional split parts

What's new (split support)
- Add --split to write the archive into .partNNN segments (default size <2GB).
- Add --split-size to control per-part target size ("2G", "1500M", "512MB", "800K", or raw bytes).
- Unpack auto-detects and reassembles parts before extraction.

Speed & usability improvements
- Default is now `--method tar-xz --level 6` (balanced speed/ratio).
- Optional multi-threaded xz: `--xz-threads N` (non-split only; shells out to `xz -T{N}`).
- Avoid streaming tar when not needed; uses seekable modes for better throughput.
- Larger copy buffers where supported.

Existing features
- Packs with a top-level root folder by default (the source dir’s basename).
- Methods: tar-xz-max (strongest), tar-xz, tar-gz, tar-bz2, zip-lzma, zip-bzip2, zip-deflate
- Default excludes: *.npz, __pycache__/, *.log, *.tmp, build/, archive/
- Progress bars for pack & unpack (bytes, speed, ETA)
- Safe extraction (prevents path traversal), overwrite checks, --verbose
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import lzma
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional

DEFAULT_EXCLUDES = ["*.npz", "__pycache__/", "*.log", "*.tmp", "build/", "archive/"]
MANIFEST_BASENAME = "PACKAGER_MANIFEST.txt"

# ---------------- Progress helpers ----------------
def fmt_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.1f} {units[i]}"

class ByteProgress:
    def __init__(self, total_bytes: int, prefix: str = ""):
        self.total = max(int(total_bytes), 0)
        self.done = 0
        self.t0 = time.time()
        self.prefix = prefix
        self._last_draw = 0.0

    def add(self, n: int):
        self.done += int(n)
        now = time.time()
        if now - self._last_draw >= 0.05:  # ~20 FPS
            self._draw(now)

    def _draw(self, now: float | None = None):
        if now is None:
            now = time.time()
        elapsed = max(now - self.t0, 1e-6)
        speed = self.done / elapsed  # bytes/s
        pct = (self.done / self.total * 100.0) if self.total else 100.0
        remain = max(self.total - self.done, 0)
        eta = remain / speed if speed > 0 else 0.0
        bar_len = 24
        filled = int(bar_len * pct / 100.0)
        bar = "#" * filled + "-" * (bar_len - filled)
        line = (
            f"{self.prefix}[{bar}] {pct:6.2f}%  "
            f"{fmt_bytes(self.done)}/{fmt_bytes(self.total)}  "
            f"{fmt_bytes(speed)}/s  ETA {int(eta)//60:02d}:{int(eta)%60:02d}"
        )
        sys.stdout.write("\r" + line[:120])
        sys.stdout.flush()
        self._last_draw = now

    def done_line(self, message: str = "Done."):
        self._draw()
        sys.stdout.write("\n" + message + "\n")
        sys.stdout.flush()

# ---------------- Split helpers ----------------
def parse_size(s: str) -> int:
    """
    Accepts numbers (bytes) or human strings like '2G', '1500M', '512MB', '800K', case-insensitive.
    """
    s = str(s).strip().lower().replace("ib", "").replace("b", "")
    if s.endswith("k"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1024**2)
    if s.endswith("g"):
        return int(float(s[:-1]) * 1024**3)
    if s.endswith("t"):
        return int(float(s[:-1]) * 1024**4)
    return int(float(s))

class PartWriter(io.RawIOBase):
    """
    File-like writer that splits output into .partNNN files up to max_bytes each.
    """
    def __init__(self, base_path: Path, max_bytes: int):
        if max_bytes <= 0:
            raise ValueError("split size must be > 0")
        self.base_path = Path(base_path)
        self.max_bytes = int(max_bytes)
        self.part_no = 1
        self.current = None
        self.bytes_in_part = 0
        self.closed_flag = False
        self._open_next()

    def _part_path(self, n: int) -> Path:
        return self.base_path.with_name(self.base_path.name + f".part{n:03d}")

    def _open_next(self):
        if self.current:
            self.current.close()
        p = self._part_path(self.part_no)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.current = open(p, "wb")
        self.bytes_in_part = 0
        self.part_no += 1

    def writable(self):
        return True

    def write(self, b: bytes) -> int:
        mv = memoryview(b)
        total = len(mv)
        off = 0
        while off < total:
            room = self.max_bytes - self.bytes_in_part
            if room <= 0:
                self._open_next()
                room = self.max_bytes
            chunk = mv[off: off + room]
            self.current.write(chunk)
            self.bytes_in_part += len(chunk)
            off += len(chunk)
        return total

    def flush(self):
        if self.current:
            self.current.flush()

    def close(self):
        if self.closed_flag:
            return
        try:
            if self.current:
                self.current.close()
        finally:
            self.closed_flag = True

def find_split_parts(base_path: Path) -> List[Path]:
    """
    Return ordered list of .partNNN files if they exist; else [].
    """
    parts = []
    n = 1
    while True:
        p = base_path.with_name(base_path.name + f".part{n:03d}")
        if p.exists():
            parts.append(p)
            n += 1
        else:
            break
    return parts

def join_parts_to_temp(parts: List[Path]) -> Path:
    """
    Concatenate parts into a temp file and return its path.
    """
    tmp = Path(tempfile.mkstemp(prefix="archiver_join_", suffix=".bin")[1])
    with open(tmp, "wb") as out:
        for p in parts:
            with open(p, "rb") as inp:
                while True:
                    chunk = inp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
    return tmp

# Wrap a file object to count bytes as tarfile reads it
class ProgressReader(io.RawIOBase):
    def __init__(self, f, progress_add):
        self.f = f
        self.progress_add = progress_add
    def read(self, n: int = -1):
        b = self.f.read(n)
        if b:
            self.progress_add(len(b))
        return b
    def readable(self):
        return True

# ---------------- Excludes & scanning ----------------
def should_exclude(path: Path, root: Path, patterns: List[str]) -> bool:
    rel_posix = path.relative_to(root).as_posix()
    name = path.name
    is_dir = path.is_dir()
    for pat in patterns:
        if pat.endswith("/"):  # directory pattern
            if is_dir and (fnmatch.fnmatch(rel_posix + "/", pat) or fnmatch.fnmatch(name + "/", pat)):
                return True
        else:
            if fnmatch.fnmatch(rel_posix, pat) or fnmatch.fnmatch(name, pat):
                return True
    return False

def iter_files(root: Path, exclude_patterns: List[str]) -> Tuple[List[Path], List[Path]]:
    files, excluded = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirpath_p = Path(dirpath)
        # prune excluded dirs
        pruned = []
        for d in list(dirnames):
            p = dirpath_p / d
            if should_exclude(p, root, exclude_patterns):
                pruned.append(d)
                excluded.append(p)
        for d in pruned:
            dirnames.remove(d)
        # files
        for fname in filenames:
            p = dirpath_p / fname
            if should_exclude(p, root, exclude_patterns):
                excluded.append(p)
            else:
                files.append(p)
    return files, excluded

def total_input_bytes(files: List[Path]) -> int:
    total = 0
    for f in files:
        try:
            total += f.stat().st_size
        except FileNotFoundError:
            pass
    return total

# ---------------- Manifest ----------------
def manifest_text(root: Path, excluded_paths: List[Path], patterns: List[str]) -> str:
    lines = []
    lines.append("Packager Manifest")
    lines.append(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S %z')}")
    lines.append(f"Root: {root}")
    lines.append("")
    lines.append("Exclude patterns:")
    for pat in patterns:
        lines.append(f"  - {pat}")
    lines.append("")
    lines.append("Excluded paths (first 500 shown if many):")
    max_list = 500
    for i, p in enumerate(excluded_paths):
        if i >= max_list:
            lines.append(f"... ({len(excluded_paths) - max_list} more)")
            break
        try:
            relp = p.relative_to(root).as_posix()
        except Exception:
            relp = str(p)
        lines.append(f"  - {relp}")
    return "\n".join(lines) + "\n"

def manifest_arcname(root_prefix: Optional[str]) -> str:
    return (root_prefix.rstrip("/") + "/" + MANIFEST_BASENAME) if root_prefix else MANIFEST_BASENAME

def add_manifest_to_tar(tf: tarfile.TarFile, root_prefix: Optional[str],
                        root: Path, excluded: List[Path], patterns: List[str]) -> None:
    data = manifest_text(root, excluded, patterns).encode("utf-8")
    name = manifest_arcname(root_prefix)
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = int(time.time())
    tf.addfile(tarinfo=info, fileobj=io.BytesIO(data))

# ---------------- Packing helpers (xz external) ----------------
def _xz_available() -> bool:
    return shutil.which("xz") is not None

def _tar_write_with_progress(tf: tarfile.TarFile, src_dir: Path, files: List[Path],
                             excluded: List[Path], verbose: bool, root_prefix: Optional[str],
                             prog: ByteProgress):
    # Larger copy buffer helps reduce Python overhead (if supported)
    try:
        tf.copybufsize = 1024 * 1024  # 1 MiB
    except Exception:
        pass

    add_manifest_to_tar(tf, root_prefix, src_dir, excluded, DEFAULT_EXCLUDES)
    for f in files:
        arc = arcname_for(f, src_dir, root_prefix)
        if verbose:
            sys.stdout.write(f"\n+ {arc}\n")
        try:
            st = f.stat()
        except FileNotFoundError:
            continue
        info = tf.gettarinfo(str(f), arcname=arc)
        info.size = st.st_size
        with open(f, "rb") as fin:
            tf.addfile(info, fileobj=ProgressReader(fin, prog.add))

def pack_tar_xz_external(src_dir: Path, out_path: Path, files: List[Path], excluded: List[Path],
                         verbose: bool, root_prefix: Optional[str], level: int,
                         extreme: bool, threads: int) -> None:
    """
    Use the system xz (multi-threaded) for best performance.
    Non-split only (writes directly to out_path).
    """
    if not _xz_available():
        raise RuntimeError("xz not found on PATH")

    args = ["xz", f"-T{threads if threads > 0 else 0}", f"-{max(0, min(9, level))}"]
    if extreme:
        args.append("-e")
    args.append("-c")  # write compressed data to stdout

    total_bytes = total_input_bytes(files)
    prog = ByteProgress(total_bytes, prefix=f"Packing ({'tar-xz-max' if extreme else 'tar-xz'} MT) ")

    with open(out_path, "wb") as out, \
         subprocess.Popen(args, stdin=subprocess.PIPE, stdout=out) as proc:
        try:
            # Stream tar into xz stdin
            with tarfile.open(fileobj=proc.stdin, mode="w|") as tf:
                _tar_write_with_progress(tf, src_dir, files, excluded, verbose, root_prefix, prog)
        finally:
            # Close stdin so xz can finish & flush
            if proc.stdin:
                proc.stdin.close()
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(f"xz exited with code {ret}")
    prog.done_line("Packing complete.")

# ---------------- Packing (with progress & splitting) ----------------
def arcname_for(f: Path, src_dir: Path, root_prefix: Optional[str]) -> str:
    rel = f.relative_to(src_dir).as_posix()
    return f"{root_prefix.rstrip('/')}/{rel}" if root_prefix else rel

def _maybe_part_writer(archive_path: Path, split: bool, split_size: int):
    if not split:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        return open(archive_path, "wb"), None  # fileobj, parts_base(None)
    return PartWriter(archive_path, split_size), archive_path

def pack_tar_xz_max(src_dir: Path, out_path: Path, files: List[Path], excluded: List[Path],
                    verbose: bool, root_prefix: Optional[str],
                    split: bool, split_size: int, xz_threads: int) -> None:
    """
    Strongest compression (xz -9e). Uses multi-threaded xz when possible (non-split).
    """
    preset = 9 | lzma.PRESET_EXTREME
    total_bytes = total_input_bytes(files)

    # Fast path: external xz for non-split
    if not split and xz_threads != 0 and _xz_available():
        pack_tar_xz_external(src_dir, out_path, files, excluded, verbose, root_prefix,
                             level=9, extreme=True, threads=xz_threads)
        return

    prog = ByteProgress(total_bytes, prefix="Packing (tar-xz-max) ")

    if not split:
        # Seekable mode with built-in lzma (usually faster than manual stream)
        with tarfile.open(out_path, mode="w:xz", preset=preset) as tf:
            _tar_write_with_progress(tf, src_dir, files, excluded, verbose, root_prefix, prog)
        prog.done_line("Packing complete.")
    else:
        # Streaming through our PartWriter when splitting
        fileobj, _ = _maybe_part_writer(out_path, split, split_size)
        try:
            with tarfile.open(fileobj=fileobj, mode="w:xz", preset=preset) as tf:
                _tar_write_with_progress(tf, src_dir, files, excluded, verbose, root_prefix, prog)
        finally:
            fileobj.close()
        prog.done_line("Packing complete.")

def pack_tar_stream(src_dir: Path, out_path: Path, files: List[Path], excluded: List[Path],
                    mode: str, level: int, verbose: bool, root_prefix: Optional[str],
                    split: bool, split_size: int, xz_threads: int) -> None:
    """
    Handles tar-gz, tar-bz2, tar-xz (non-extreme).
    Uses external xz for tar-xz when non-split and xz-threads given.
    """
    if mode == "w:xz" and (not split) and (xz_threads != 0) and _xz_available():
        pack_tar_xz_external(src_dir, out_path, files, excluded, verbose, root_prefix,
                             level=level, extreme=False, threads=xz_threads)
        return

    # Pick the correct keyword for each compressor
    if mode == "w:gz":
        open_kwargs = {"compresslevel": level}         # gzip 1..9
    elif mode == "w:bz2":
        open_kwargs = {"compresslevel": level}         # bzip2 1..9
    elif mode == "w:xz":
        open_kwargs = {"preset": level}                # xz/lzma 0..9
    else:
        open_kwargs = {}

    total_bytes = total_input_bytes(files)
    prog = ByteProgress(total_bytes, prefix=f"Packing ({mode}) ")

    if not split:
        with tarfile.open(out_path, mode=mode, **open_kwargs) as tf:
            _tar_write_with_progress(tf, src_dir, files, excluded, verbose, root_prefix, prog)
        prog.done_line("Packing complete.")
    else:
        fileobj, _ = _maybe_part_writer(out_path, split, split_size)
        try:
            with tarfile.open(fileobj=fileobj, mode=mode, **open_kwargs) as tf:
                _tar_write_with_progress(tf, src_dir, files, excluded, verbose, root_prefix, prog)
        finally:
            fileobj.close()
        prog.done_line("Packing complete.")

def pack_zip_like(src_dir: Path, out_path: Path, files: List[Path], excluded: List[Path],
                  method: str, level: int, verbose: bool, root_prefix: Optional[str],
                  split: bool, split_size: int) -> None:
    comp_map = {
        "zip-deflate": zipfile.ZIP_DEFLATED,
        "zip-bzip2": zipfile.ZIP_BZIP2,
        "zip-lzma": zipfile.ZIP_LZMA,
    }
    comp = comp_map[method]
    kw = dict(allowZip64=True)
    try:
        kw["compresslevel"] = level
    except TypeError:
        pass

    total_bytes = total_input_bytes(files)
    prog = ByteProgress(total_bytes, prefix=f"Packing ({method}) ")

    fileobj, _ = _maybe_part_writer(out_path, split, split_size)
    try:
        with zipfile.ZipFile(file=fileobj, mode="w", compression=comp, **kw) as zf:
            zf.writestr(manifest_arcname(root_prefix), manifest_text(src_dir, excluded, DEFAULT_EXCLUDES))
            for f in files:
                arc = arcname_for(f, src_dir, root_prefix)
                if verbose:
                    sys.stdout.write(f"\n+ {arc}\n")
                try:
                    size = f.stat().st_size
                except FileNotFoundError:
                    size = 0
                zf.write(f, arc)
                prog.add(size)
    finally:
        fileobj.close()
    prog.done_line("Packing complete.")

def pack(src_dir: Path, archive_path: Path, method: str, level: int,
         extra_excludes: List[str], dry_run: bool, verbose: bool,
         root_prefix: Optional[str], split: bool, split_size: int, xz_threads: int) -> None:
    exclude_patterns = list(DEFAULT_EXCLUDES) + (extra_excludes or [])
    files, excluded = iter_files(src_dir, exclude_patterns)

    if dry_run:
        print(f"[DRY RUN] include={len(files)} exclude={len(excluded)}")
        show = 20
        for p in files[:show]:
            print(" +", arcname_for(p, src_dir, root_prefix))
        for p in excluded[:show]:
            try:
                relp = p.relative_to(src_dir).as_posix()
            except Exception:
                relp = str(p)
            print(" -", relp)
        return

    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if method == "tar-xz-max":
        pack_tar_xz_max(src_dir, archive_path, files, excluded, verbose, root_prefix, split, split_size, xz_threads)
    elif method in ("tar-xz", "tar-gz", "tar-bz2"):
        mode = {"tar-gz": "w:gz", "tar-bz2": "w:bz2", "tar-xz": "w:xz"}[method]
        pack_tar_stream(src_dir, archive_path, files, excluded, mode, level, verbose, root_prefix, split, split_size, xz_threads)
    elif method in ("zip-lzma", "zip-bzip2", "zip-deflate"):
        pack_zip_like(src_dir, archive_path, files, excluded, method, level, verbose, root_prefix, split, split_size)
    else:
        raise SystemExit(f"Unknown method: {method}")

# ---------------- Safe paths & unpacking (with join) ----------------
def _is_within_directory(base: Path, target: Path) -> bool:
    try:
        return os.path.commonpath([base.resolve()]) == os.path.commonpath([base.resolve(), target.resolve()])
    except Exception:
        return False

def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def _set_times_and_perms(path: Path, mtime: Optional[float], mode_bits: Optional[int]):
    try:
        if mtime is not None:
            os.utime(path, (mtime, mtime), follow_symlinks=False)
    except Exception:
        pass
    try:
        if mode_bits is not None:
            os.chmod(path, mode_bits & 0o777, follow_symlinks=False)
    except Exception:
        pass

def _resolve_archive_or_parts(archive_path: Path) -> Path:
    """
    If archive_path exists, return it.
    If not, but split parts <archive>.part001... exist, join them to a temp file and return that path.
    """
    if archive_path.exists():
        return archive_path
    parts = find_split_parts(archive_path)
    if parts:
        print(f"Detected split parts ({len(parts)}). Joining for extraction...")
        tmp = join_parts_to_temp(parts)
        return tmp
    raise SystemExit(f"Archive not found: {archive_path} (no .part files found either)")

def unpack_zip(archive_path: Path, dest_dir: Path, overwrite: bool, verbose: bool):
    actual = _resolve_archive_or_parts(archive_path)
    with zipfile.ZipFile(actual, "r") as zf:
        members = zf.infolist()
        total_bytes = sum(m.file_size for m in members if not m.is_dir())
        prog = ByteProgress(total_bytes, prefix="Unpacking (zip) ")

        # collision check & safety
        collisions = []
        for m in members:
            tgt = (dest_dir / m.filename).resolve()
            if not _is_within_directory(dest_dir, tgt):
                raise SystemExit(f"Unsafe path in archive: {m.filename}")
            if not m.is_dir() and tgt.exists() and not overwrite:
                collisions.append(m.filename)
        if collisions:
            print("Refusing to overwrite existing files. Use --overwrite to replace.")
            print("Collisions (first 50):")
            for m in collisions[:50]:
                print("  -", m)
            return

        for m in members:
            out_path = dest_dir / m.filename
            if verbose:
                sys.stdout.write(f"\n* {m.filename}\n")
            if m.is_dir():
                out_path.mkdir(parents=True, exist_ok=True)
                continue
            _ensure_parent(out_path)
            with zf.open(m, "r") as src, open(out_path, "wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
                    prog.add(len(chunk))
            # perms & times
            mtime = time.mktime(m.date_time + (0, 0, -1))
            mode_bits = (m.external_attr >> 16) & 0o7777 if m.external_attr else None
            _set_times_and_perms(out_path, mtime, mode_bits)
        prog.done_line("Unpacking complete.")

def unpack_tar(archive_path: Path, dest_dir: Path, mode: str, verbose: bool, overwrite: bool):
    actual = _resolve_archive_or_parts(archive_path)
    with tarfile.open(actual, mode) as tf:
        members = tf.getmembers()
        total_bytes = sum(m.size for m in members if m.isfile())
        prog = ByteProgress(total_bytes, prefix=f"Unpacking ({mode}) ")

        # collision check & safety
        collisions = []
        for m in members:
            name = m.name.lstrip("/")  # normalize
            tgt = (dest_dir / name).resolve()
            if not _is_within_directory(dest_dir, tgt):
                raise SystemExit(f"Unsafe path in archive: {m.name}")
            if m.isfile() and tgt.exists() and not overwrite:
                collisions.append(m.name)
        if collisions:
            print("Refusing to overwrite existing files. Use --overwrite to replace.")
            print("Collisions (first 50):")
            for m in collisions[:50]:
                print("  -", m)
            return

        for m in members:
            name = m.name.lstrip("/")
            out_path = dest_dir / name
            if verbose:
                sys.stdout.write(f"\n* {name}\n")

            if m.isdir():
                out_path.mkdir(parents=True, exist_ok=True)
                continue
            if m.issym() or m.islnk():
                sys.stdout.write("  (skipping link for safety)\n")
                continue
            if m.isfile():
                _ensure_parent(out_path)
                src = tf.extractfile(m)
                if src is None:
                    continue
                with src, open(out_path, "wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
                        prog.add(len(chunk))
                mtime = float(getattr(m, "mtime", time.time()))
                mode_bits = m.mode if isinstance(m.mode, int) else None
                _set_times_and_perms(out_path, mtime, mode_bits)
        prog.done_line("Unpacking complete.")

def unpack(archive_path: Path, dest_dir: Path, overwrite: bool, verbose: bool) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()

    if name.endswith(".zip"):
        unpack_zip(archive_path, dest_dir, overwrite, verbose)
        return

    # TAR family
    if name.endswith((".tar.xz", ".txz")):
        unpack_tar(archive_path, dest_dir, "r:xz", verbose, overwrite)
        return
    if name.endswith((".tar.gz", ".tgz")):
        unpack_tar(archive_path, dest_dir, "r:gz", verbose, overwrite)
        return
    if name.endswith((".tar.bz2", ".tbz2")):
        unpack_tar(archive_path, dest_dir, "r:bz2", verbose, overwrite)
        return
    if name.endswith(".tar"):
        unpack_tar(archive_path, dest_dir, "r:", verbose, overwrite)
        return

    raise SystemExit("Unsupported archive extension. Use .zip, .tar, .tar.gz/.tgz, .tar.bz2/.tbz2, .tar.xz/.txz")

# ---------------- CLI ----------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Pack/unpack with strong compression and progress. "
                    "Defaults exclude: " + ", ".join(DEFAULT_EXCLUDES)
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pack = sub.add_parser("pack", help="Create an archive")
    p_pack.add_argument("src_dir", type=Path, help="Directory to package")
    p_pack.add_argument("archive_path", type=Path, help="Output archive path (e.g., /tmp/a.tar.xz or /tmp/a.zip)")
    p_pack.add_argument("--method",
        choices=["tar-xz-max", "tar-xz", "tar-gz", "tar-bz2", "zip-lzma", "zip-bzip2", "zip-deflate"],
        default="tar-xz",  # changed default (balanced)
        help="Compression method/format (tar-xz-max = smallest; slowest).")
    p_pack.add_argument("--level", type=int, default=6,
        help="Compression level for non-max modes (1=fast..9=small). Ignored by tar-xz-max when using extreme.")
    p_pack.add_argument(
        "--xz-threads",
        dest="xz_threads",   # <-- explicit
        type=int,
        default=0,
        help="Use multi-threaded system xz with this many threads (0 = all cores). "
            "Only for tar-xz/tar-xz-max and when NOT using --split."
    )
    p_pack.add_argument("--exclude", "-e", action="append", default=[],
        help="Additional glob excludes (defaults already include: " + ", ".join(DEFAULT_EXCLUDES) + ")")
    p_pack.add_argument("--dry-run", action="store_true", help="Preview includes/excludes and exit")
    p_pack.add_argument("--verbose", action="store_true", help="Print each file as it’s processed")

    # Root folder controls
    root_grp = p_pack.add_mutually_exclusive_group()
    root_grp.add_argument("--no-root", action="store_true",
        help="Do NOT include a top-level folder; store files flat (current directory layout).")
    root_grp.add_argument("--root", type=str, default=None,
        help="Custom name for the top-level folder inside the archive (default: source dir name).")

    # Split options
    p_pack.add_argument("--split", action="store_true",
        help="Write output into numbered parts (.part001, .part002, ...) instead of a single file.")
    p_pack.add_argument("--split-size", type=str, default="2G",
        help="Max size per part when using --split (e.g., 2G, 1500M, 512MB, 800K, or raw bytes).")

    p_unpack = sub.add_parser("unpack", help="Extract an archive")
    p_unpack.add_argument("archive_path", type=Path, help="Archive to extract (or the base name of .part files)")
    p_unpack.add_argument("dest_dir", type=Path, help="Destination directory (will receive the archive’s top folder)")
    p_unpack.add_argument("--overwrite", action="store_true", help="Allow overwriting existing files")
    p_unpack.add_argument("--verbose", action="store_true", help="Print each file as it’s processed")

    args = parser.parse_args(argv)
    if args.cmd == "pack":
        src_dir = args.src_dir.resolve()
        if args.no_root:
            root_prefix = None
        elif args.root:
            root_prefix = args.root
        else:
            root_prefix = src_dir.name  # default: include top-level folder

        split = bool(args.split)
        split_size = parse_size(args.split_size) if split else 0

        if split and args.xz_threads != 0:
            print("[info] --xz-threads is ignored when --split is used (falling back to built-in compressor).")

        pack(src_dir, args.archive_path, args.method, args.level,
             args.exclude, args.dry_run, args.verbose, root_prefix,
             split, split_size, args.xz_threads)

    elif args.cmd == "unpack":
        unpack(args.archive_path, args.dest_dir, overwrite=args.overwrite, verbose=args.verbose)

if __name__ == "__main__":
    main()

