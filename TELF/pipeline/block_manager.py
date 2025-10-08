from __future__ import annotations

from pathlib import Path
import sys, io, contextlib, datetime as _dt, traceback
from typing import List, Tuple, Sequence, Any, Dict, Set

from IPython.display import display, HTML

from .blocks.base_block import AnimalBlock
from .blocks.data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY, SOURCE_DIR_BUNDLE_KEY

import jsonpickle
import re
import time


class BlockManager:
    """
    Orchestrates a list of `AnimalBlock` instances, moving a single
    `DataBundle` through the pipeline.

    The manager is **namespace-aware**:

        - Each block owns a unique `tag` (defaults to the class name;
          duplicates automatically receive a numeric suffix).
        - A block that writes `provides = ("df",)` actually puts its data
          under the key  ``"<tag>.df"``  *and* updates the bundle’s
          “latest” pointer for ``"df"``.
        - A block may depend on either the generic key ( `"df"` ) or any
          fully-qualified key ( e.g. `"Clean.df"` ).
    """


    def __init__(
        self,
        blocks: List[AnimalBlock],
        databundle: DataBundle | None = None,
        *,
        verbose: bool = True,
        progress: bool = True,
        capture_output: str | None = "file",
        force_checkpoint: bool | None = None,
    ):
        self.verbose = bool(verbose)
        self.progress = bool(progress)
        self.capture_output = capture_output   # None | "memory" | "file"
        self.force_checkpoint = force_checkpoint  # None: default; True: force load; False: skip load
        self.block_logs: Dict[str, str] = {}

        self.blocks: List[AnimalBlock] = blocks
        self.bundle: DataBundle = databundle or DataBundle()

        self._assign_unique_tags()
        self._ensure_result_path()
        self.check_io_consistency()
        if self.verbose:
            self.describe_io()

    def __call__(self) -> DataBundle:
        base = Path(self.bundle[SAVE_DIR_BUNDLE_KEY])

        # 1) Preflight: rename block directories and update checkpoint paths on disk
        #    MUST return:
        #      - base_to_display: {"SemanticHNMFk": "06_SemanticHNMFk", ...}
        #      - prefix_map: {"/.../07_SemanticHNMFk": "/.../06_SemanticHNMFk", ...}
        base_to_display, prefix_map = _renumber_dirs_and_update_ckpts(base, self.blocks)

        # 2) Update in-memory objects to new prefixes BEFORE any block runs
        # 2a) Rewrite any stored paths inside the bundle values
        try:
            for base_key, bucket in list(self.bundle._store.items()):
                for tag, val in list(bucket.items()):
                    if tag == "_latest":
                        continue
                    # Apply all variants
                    new_val = val
                    for old_pref, new_pref in prefix_map.items():
                        new_val = _deep_replace_in_obj(new_val, {old_pref: new_pref})
                    bucket[tag] = new_val
        except Exception:
            pass

        # Set display tags and rewrite block settings (init/call) in memory
        for block in self.blocks:
            bt = _base_tag(getattr(block, "_original_tag", block.tag))
            if not hasattr(block, "_original_tag"):
                block._original_tag = bt
            block.tag = base_to_display.get(bt, bt)

            if isinstance(getattr(block, "init_settings", None), dict):
                new_init = block.init_settings
                for old_pref, new_pref in prefix_map.items():
                    new_init = _deep_replace_in_obj(new_init, {old_pref: new_pref})
                block.init_settings = new_init

            if isinstance(getattr(block, "call_settings", None), dict):
                new_call = block.call_settings
                for old_pref, new_pref in prefix_map.items():
                    new_call = _deep_replace_in_obj(new_call, {old_pref: new_pref})
                block.call_settings = new_call

        total = len(self.blocks)
        log_dir: Path | None = None
        progress_fp = None
        if self.capture_output == "file":
            log_dir = Path(self.bundle[SAVE_DIR_BUNDLE_KEY]) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            progress_fp = (log_dir / "progress.log").open("a", encoding="utf-8")

            table_lines = self._describe_io_as_lines()
            progress_fp.write("# IO table\n" + "\n".join(table_lines) + "\n\n")
            progress_fp.flush()

        # 3) Run blocks
        for idx, block in enumerate(self.blocks, 1):
            # Override the block’s load_checkpoint flag if requested
            if self.force_checkpoint is not None:
                block.load_checkpoint = self.force_checkpoint

            if self.progress:
                print(f"▶  [{idx}/{total}] {block.tag} …", flush=True)

            t0 = time.perf_counter()

            # Capture output and trace exceptions
            if self.capture_output:
                buf_out, buf_err = io.StringIO(), io.StringIO()
                try:
                    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                        self.bundle = block(self.bundle)
                except Exception:
                    buf_err.write(f"⚠️ Exception in block {block.tag}:\n")
                    traceback.print_exc(file=buf_err)
                    captured = buf_out.getvalue() + buf_err.getvalue()
                    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                    (log_dir / f"{idx:02d}_{block.tag}_{ts}.log").write_text(captured, encoding="utf-8")
                    raise
                else:
                    captured = buf_out.getvalue() + buf_err.getvalue()
                    if self.capture_output == "memory":
                        self.block_logs[block.tag] = captured
                    else:
                        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                        (log_dir / f"{idx:02d}_{block.tag}_{ts}.log").write_text(captured, encoding="utf-8")
            else:
                try:
                    self.bundle = block(self.bundle)
                except Exception:
                    print(f"⚠️ Exception in block {block.tag}:")
                    traceback.print_exc()
                    raise

            # 4) Alias outputs: also store under the base tag (no NN_ prefix)
            disp_tag = block.tag                   # e.g., "06_SemanticHNMFk"
            base_tag = _base_tag(disp_tag)         # e.g., "SemanticHNMFk"
            for base_key in self.bundle.keys_by_tag(disp_tag):
                try:
                    val = self.bundle[f"{disp_tag}.{base_key}"]
                    self.bundle[f"{base_tag}.{base_key}"] = val
                except KeyError:
                    pass

            elapsed = time.perf_counter() - t0
            if self.progress:
                print(f"✓  [{idx}/{total}] {block.tag} finished in {elapsed:,.2f}s")

            if progress_fp:
                ts_iso = _dt.datetime.now().isoformat(timespec="seconds")
                progress_fp.write(f"{ts_iso}\t{idx}/{total}\t{block.tag}\t{elapsed:.2f}s\n")
                progress_fp.flush()

        if progress_fp:
            progress_fp.close()
        return self.bundle


    # ------------------------------------------------------------------ #
    # helper – produce describe-io lines without printing                 #
    # ------------------------------------------------------------------ #
    def _describe_io_as_lines(self) -> List[str]:
        """
        Dynamically compute column widths based on the longest entry in each column
        and return formatted lines describing each block's I/O.
        """
        header_col1 = "Block (tag)"
        header_col2 = "Needs"
        header_col3 = "Provides"

        rows: List[Tuple[str, str, str]] = []
        for blk in self.blocks:
            col1 = f"{blk.__class__.__name__} ({blk.tag})"
            eff_needs = list(blk.needs) + [k for k, _ in blk.conditional_needs]
            col2 = ", ".join(eff_needs)
            col3 = str(list(blk.provides))
            rows.append((col1, col2, col3))

        col1_width = max(len(header_col1), *(len(r[0]) for r in rows))
        col2_width = max(len(header_col2), *(len(r[1]) for r in rows))

        header = f"{header_col1:<{col1_width}} │ {header_col2:<{col2_width}} │ {header_col3}"
        separator = "─" * len(header)

        lines = [header, separator]
        for col1, col2, col3 in rows:
            lines.append(f"{col1:<{col1_width}} │ {col2:<{col2_width}} │ {col3}")

        return lines

    # ------------------------------------------------------------------ #
    # user-friendly descriptions                                         #
    # ------------------------------------------------------------------ #
    def describe_io(self) -> List[Tuple[str, Sequence[str], Sequence[str]]]:
        """
        Print (and return) a table

            Block (tag) │ Needs (✓/✗) │ Provides

        Needs that are NOT currently satisfied are coloured red and suffixed
        with a short reason.  The logic is shared with check_io_consistency()
        so both views stay consistent.
        """
        rows: List[Tuple[str, Sequence[str], Sequence[str]]] = []

        # preload bundle state
        generic_seen: Set[str] = set()
        by_tag: Dict[str, Set[str]] = {}
        for base, bucket in self.bundle._store.items():  # type: ignore[attr-defined]
            generic_seen.add(base)
            for tag in bucket:
                if tag != "_latest":
                    by_tag.setdefault(tag, set()).add(base)

        # collect rows with original and colored needs
        for blk in self.blocks:
            active_cond = [k for k, cond in blk.conditional_needs if cond(self.bundle, blk)]
            eff_needs = list(blk.needs) + active_cond

            display_needs: List[str] = []
            canon = blk._canonical_needs

            for need in eff_needs:
                if "." in need:
                    tag, key = need.split(".", 1)
                    if key not in by_tag.get(tag, set()):
                        display_needs.append(f"\x1b[31m{need} (bad namespace)\x1b[0m")
                    else:
                        display_needs.append(need)
                else:
                    if need not in generic_seen:
                        display_needs.append(f"\x1b[31m{need} (missing)\x1b[0m")
                    else:
                        display_needs.append(need)

            # wrong-order annotation
            suffixes = [n.split(".", 1)[-1] for n in eff_needs if n.split(".", 1)[-1] in canon]
            if tuple(suffixes) != canon:
                display_needs.append("\x1b[31m(order❌)\x1b[0m")

            rows.append((f"{blk.__class__.__name__} ({blk.tag})", display_needs, list(blk.provides)))

            # register provides
            for p in blk.provides:
                generic_seen.add(p)
                by_tag.setdefault(blk.tag, set()).add(p)

        # prepare colored and plain needs strings
        ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
        colored_needs = [", ".join(needs) for _, needs, _ in rows]
        plain_needs = [ANSI_RE.sub("", s) for s in colored_needs]

        # compute dynamic column widths
        col1_label = "Block (tag)"
        col2_label = "Needs (✓/✗)"
        col1_width = max(len(col1_label), *(len(name) for name, _, _ in rows))
        col2_width = max(len(col2_label), *(len(s) for s in plain_needs))

        # print if verbose
        if self.verbose:
            header = f"{col1_label:<{col1_width}} │ {col2_label:<{col2_width}} │ Provides"
            print(header)
            print("─" * len(header))

            def ansi_ljust(s: str, width: int) -> str:
                """
                Left-justify a string containing ANSI codes based on its visible length.
                """
                visible = ANSI_RE.sub("", s)
                padding = width - len(visible)
                return s + " " * max(0, padding)

            for name, needs_list, provides_list in rows:
                colored = ", ".join(needs_list)
                field = ansi_ljust(colored, col2_width)
                print(f"{name:<{col1_width}} │ {field} │ {provides_list}")
            print()

        return rows

    # --------------------------------------------------------------------- #
    # consistency check                                                     #
    # --------------------------------------------------------------------- #
    def check_io_consistency(self) -> None:
        """See README – validates namespace, order, and conditional needs."""
        # --- notebook? ---------------------------------------------------------
        try:
            from IPython import get_ipython
            IN_NOTEBOOK = "IPKernelApp" in get_ipython().config          # type: ignore[attr-defined]
        except Exception:
            IN_NOTEBOOK = False

        # --- gather what we already have in the bundle -------------------------
        generic_seen:  Set[str] = set()
        by_tag:        Dict[str, Set[str]] = {}          # tag → {keys}

        for base, bucket in self.bundle._store.items():                  # type: ignore[attr-defined]
            generic_seen.add(base)
            for tag, _ in bucket.items():
                if tag != "_latest":
                    by_tag.setdefault(tag, set()).add(base)

        # --- walk through the pipeline ----------------------------------------
        lines: list[str] = []
        for blk in self.blocks:
            # evaluate conditional needs on the CURRENT bundle snapshot
            active_conditional: list[str] = []
            for key, cond in blk.conditional_needs:
                try:
                    if cond(self.bundle, blk):
                        active_conditional.append(key)
                except Exception:
                    active_conditional.append(key)

            eff_needs = list(blk.needs) + active_conditional

            # ---------- rule 1 & 2 : missing / bad namespace ------------------
            missing, bad_ns = [], []
            for need in eff_needs:
                if "." in need:
                    tag, key = need.split(".", 1)
                    if tag not in by_tag:
                        missing.append(need)
                    elif key not in by_tag[tag]:
                        bad_ns.append(need)
                else:
                    if need not in generic_seen:
                        missing.append(need)

            # ---------- rule 3 : order of canonical needs ---------------------
            canon = blk._canonical_needs
            user_suffix_seq = [
                (n.split(".", 1)[1] if "." in n else n)
                for n in eff_needs
                if (n.split(".", 1)[-1]) in canon
            ]
            order_error = None
            if tuple(user_suffix_seq) != canon:
                order_error = (
                    "wrong order – expected "
                    f"({', '.join(canon)}) but got "
                    f"({', '.join(user_suffix_seq) or '∅'})"
                )

            # ---------- compose message ---------------------------------------
            problems: list[str] = []
            if missing:
                problems.append(
                    "missing " + ", ".join(f"<code>{m}</code>" for m in missing)
                )
            if bad_ns:
                problems.append(
                    "bad namespace " + ", ".join(f"<code>{b}</code>" for b in bad_ns)
                )
            if order_error:
                problems.append(order_error)

            if problems:
                msg = f"<b style='color:red'>{blk.tag}</b> – " + "; ".join(problems)
            else:
                msg = f"<b style='color:green'>{blk.tag}</b> – all needs met"

            lines.append(msg)

            # ---------- register provides for downstream blocks ---------------
            for p in blk.provides:
                generic_seen.add(p)
                by_tag.setdefault(blk.tag, set()).add(p)

        # --- show report ------------------------------------------------------
        report_html = "<br>".join(lines)
        if IN_NOTEBOOK:
            display(HTML(report_html))
        elif self.verbose:
            for line in lines:
                print(
                    line.replace("<b style='color:green'>", "")
                        .replace("<b style='color:red'>", "")
                        .replace("</b>", "")
                        .replace("<code>", "")
                        .replace("</code>", "")
                )

    # --------------------------------------------------------------------- #
    # persistence helpers                                                   #
    # --------------------------------------------------------------------- #
    def save_settings(self) -> None:
        """
        Serialize every block (via `jsonpickle`) into
        ``<result_path>/saved_settings/{idx}_{ClassName}.json``.
        """
        saved_dir = Path(self.bundle["result_path"]) / "saved_settings"
        saved_dir.mkdir(parents=True, exist_ok=True)

        for idx, blk in enumerate(self.blocks):
            fn = saved_dir / f"{idx}_{blk.__class__.__name__}.json"
            fn.write_text(jsonpickle.encode(blk, keys=True), encoding="utf-8")

    def load_saved_settings(self) -> None:
        """
        Load any JSON files in ``<result_path>/saved_settings`` and restore
        the state of blocks whose *class names* match.
        """
        saved_dir = Path(self.bundle["result_path"]) / "saved_settings"
        if not saved_dir.is_dir():
            raise FileNotFoundError(f"No saved settings found in {saved_dir!r}.")

        for fp in saved_dir.glob("*.json"):
            _, class_name = fp.stem.split("_", 1)
            for blk in self.blocks:
                if blk.__class__.__name__ == class_name:
                    blk.load_settings(fp)
                    break

        self.check_io_consistency()  # re-validate after loading

    # --------------------------------------------------------------------- #
    # internal helpers                                                      #
    # --------------------------------------------------------------------- #
    def _assign_unique_tags(self) -> None:
        """
        Ensure every block has a distinct `tag`.  If duplicates are found,
        they receive a numeric suffix (Block, Block2, Block3, …).
        """
        counts: Dict[str, int] = {}

        for blk in self.blocks:
            tag = getattr(blk, "tag", None) or blk.__class__.__name__
            counts[tag] = counts.get(tag, 0) + 1
            if counts[tag] > 1:
                tag = f"{tag}{counts[tag]}"
            blk.tag = tag  # type: ignore[attr-defined]

    def _ensure_result_path(self) -> None:
        """
        Guarantee the bundle has a writable `result_path`.
        """
        if "result_path" not in self.bundle:
            self.bundle["result_path"] = Path.cwd() / "results"



import json, os, re, uuid, pickle
from pathlib import Path
from typing import Any

INDEXED_DIR_RE = re.compile(r"^\d+_")
TEXT_EXTS = {".json", ".txt", ".yaml", ".yml", ".ini", ".cfg", ".csv", ".tsv"}
PICKLE_EXTS = {".p", ".pkl", ".pickle"}

def _base_tag(name: str) -> str:
    return INDEXED_DIR_RE.sub("", name)

def _display_name(tag: str, idx: int, width: int) -> str:
    return f"{idx:0{width}d}_{tag}"

def _find_existing_dir(base: Path, tag: str) -> Path | None:
    candidates = sorted(
        base.glob(f"[0-9][0-9]*_{tag}"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    plain = base / tag
    return plain if plain.exists() else None

def _collect_ckpt_files(base: Path) -> list[Path]:
    return list(base.rglob("__checkpoints__.json"))

def _rewrite_ckpt_paths(ckpt_file: Path, prefix_map: dict[str, str]) -> bool:
    try:
        data = json.loads(ckpt_file.read_text(encoding="utf-8"))
    except Exception:
        return False
    changed = False
    for k, v in list(data.items()):
        if isinstance(v, str):
            for old_prefix, new_prefix in prefix_map.items():
                if v == old_prefix or v.startswith(old_prefix + os.sep):
                    data[k] = new_prefix + v[len(old_prefix):]
                    changed = True
    if changed:
        ckpt_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return changed

def _deep_replace(obj: Any, old_prefix: str, new_prefix: str, seen: set[int] | None = None) -> tuple[bool, Any]:
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return False, obj
    seen.add(oid)

    if isinstance(obj, str):
        if old_prefix in obj:
            return True, obj.replace(old_prefix, new_prefix)
        return False, obj

    if isinstance(obj, dict):
        changed = False
        out = {}
        for k, v in obj.items():
            ck, nk = _deep_replace(k, old_prefix, new_prefix, seen) if isinstance(k, str) else (False, k)
            cv, nv = _deep_replace(v, old_prefix, new_prefix, seen)
            changed = changed or ck or cv
            out[nk] = nv
        return changed, out

    if isinstance(obj, list):
        changed = False
        out = []
        for v in obj:
            cv, nv = _deep_replace(v, old_prefix, new_prefix, seen)
            changed = changed or cv
            out.append(nv)
        return changed, out

    if isinstance(obj, tuple):
        changed = False
        out_list = []
        for v in obj:
            cv, nv = _deep_replace(v, old_prefix, new_prefix, seen)
            changed = changed or cv
            out_list.append(nv)
        return changed, tuple(out_list)

    try:
        attrs = vars(obj)
    except Exception:
        return False, obj

    changed = False
    for k, v in list(attrs.items()):
        cv, nv = _deep_replace(v, old_prefix, new_prefix, seen)
        if cv:
            try:
                setattr(obj, k, nv)
                changed = True
            except Exception:
                pass
    return changed, obj

def _rewrite_internal_paths_in_tree(root: Path, prefix_map: dict[str, str]) -> None:
    """
    Walk files under `root` and replace occurrences of *any* old→new prefix.
    Handles text and pickle files; best-effort, silent on failures.
    """
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()

        if ext in TEXT_EXTS:
            try:
                s = p.read_text(encoding="utf-8", errors="ignore")
                changed = False
                for old_prefix, new_prefix in prefix_map.items():
                    if old_prefix in s:
                        s = s.replace(old_prefix, new_prefix)
                        changed = True
                if changed:
                    p.write_text(s, encoding="utf-8")
            except Exception:
                pass
            continue

        if ext in PICKLE_EXTS:
            try:
                with p.open("rb") as f:
                    obj = pickle.load(f)
                changed_any = False
                for old_prefix, new_prefix in prefix_map.items():
                    changed, obj = _deep_replace(obj, old_prefix, new_prefix)
                    changed_any = changed_any or changed
                if changed_any:
                    with p.open("wb") as f:
                        pickle.dump(obj, f)
            except Exception:
                pass
            continue

def _renumber_dirs_and_update_ckpts(base: Path, blocks: list) -> tuple[dict[str, str], dict[str, str]]:
    """
    Rename block directories to NN_<tag>, rewrite checkpoint JSONs,
    and migrate internal absolute/relative paths inside files under renamed dirs.
    RETURNS:
      (base_to_display, prefix_map) where prefix_map includes abs+rel variants.
    """
    base.mkdir(parents=True, exist_ok=True)

    width = max(2, len(str(len(blocks))))
    desired: list[tuple[str, str]] = []
    for i, block in enumerate(blocks, start=1):
        bt = _base_tag(getattr(block, "_original_tag", block.tag))
        desired.append((bt, _display_name(bt, i, width)))

    # Plan renames
    rename_plan: list[tuple[Path, Path]] = []
    for (bt, new_disp) in desired:
        src = _find_existing_dir(base, bt)
        if src is None:
            continue
        dst = base / new_disp
        if src.resolve() != dst.resolve():
            rename_plan.append((src, dst))

    # Two-phase rename to avoid collisions
    temp_map: dict[Path, Path] = {}
    for src, dst in rename_plan:
        if not src.exists():
            continue
        tmp = base / (src.name + f".tmp-{uuid.uuid4().hex[:8]}")
        src.rename(tmp)
        temp_map[tmp] = dst

    # Move temps to final
    for tmp, dst in temp_map.items():
        if dst.exists():
            if dst.is_dir() and tmp.is_dir():
                for item in tmp.iterdir():
                    target = dst / item.name
                    if not target.exists():
                        item.rename(target)
                tmp.rmdir()
            else:
                n = 1
                alt = Path(str(dst) + f".old{n}")
                while alt.exists():
                    n += 1
                    alt = Path(str(dst) + f".old{n}")
                tmp.rename(alt)
        else:
            tmp.rename(dst)

    # Build a rich prefix map (absolute + relative variants)
    prefix_map = _build_prefix_map_variants(base, rename_plan)

    # Rewrite checkpoint JSON files using all variants
    for ckpt in _collect_ckpt_files(base):
        _rewrite_ckpt_paths(ckpt, prefix_map)

    # Rewrite internals inside each renamed directory tree (text+pickle) using all variants
    for src, dst in rename_plan:
        try:
            _rewrite_internal_paths_in_tree(dst, prefix_map)
        except Exception:
            pass

    base_to_display = {bt: new_disp for (bt, new_disp) in desired}
    return base_to_display, prefix_map


def _deep_replace_in_obj(obj, prefix_map: dict[str, str]):
    from pathlib import Path as _Path
    if isinstance(obj, (str, _Path)):
        s = str(obj)
        for old, new in prefix_map.items():
            if s == old or s.startswith(old + os.sep):
                s = new + s[len(old):]
        return _Path(s) if isinstance(obj, _Path) else s
    if isinstance(obj, dict):
        return { _deep_replace_in_obj(k, prefix_map) if isinstance(k, (str, _Path)) else k:
                 _deep_replace_in_obj(v, prefix_map) for k, v in obj.items() }
    if isinstance(obj, list):
        return [ _deep_replace_in_obj(v, prefix_map) for v in obj ]
    if isinstance(obj, tuple):
        return tuple(_deep_replace_in_obj(v, prefix_map) for v in obj)
    return obj

def _build_prefix_map_variants(base: Path, rename_plan: list[tuple[Path, Path]]) -> dict[str, str]:
    """
    For each (src,dst) directory rename, return a mapping that includes:
      - absolute: /abs/.../07_Tag  -> /abs/.../06_Tag
      - relative (from CWD):  src  ->  dst   (e.g., 'example_results/.../07_Tag' -> '.../06_Tag')
    This catches both absolute and relative paths embedded in files or memory.
    """
    m: dict[str, str] = {}
    for src, dst in rename_plan:
        # absolute variants
        abs_old = str(src.resolve())
        abs_new = str(dst.resolve())
        m[abs_old] = abs_new

        # relative variants (as written on disk; your code uses Path(...) directly)
        rel_old = str(src)  # typically 'example_results/.../07_Tag'
        rel_new = str(dst)
        m[rel_old] = rel_new

        # Sometimes code stores a trailing slash; add those too
        if not abs_old.endswith(os.sep):
            m[abs_old + os.sep] = abs_new + os.sep
        if not rel_old.endswith(os.sep):
            m[rel_old + os.sep] = rel_new + os.sep
    return m
