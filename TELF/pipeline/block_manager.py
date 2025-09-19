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
                    # Write error log and re-raise
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
