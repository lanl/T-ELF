# blocks/sbatch_block.py
from __future__ import annotations
import subprocess, textwrap
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import jsonpickle
from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

# ───────────────────────── helpers ──────────────────────────

def _nospace(s: str) -> str:
    """Make a safe SLURM job-name / path fragment by replacing spaces."""
    return s.replace(" ", "_")

def clean_path(p: Path | str) -> Path:
    """Recursively apply _nospace to every component of the path."""
    p = Path(p)
    return Path(*(_nospace(part) for part in p.parts))

def shquote(x: str | Path) -> str:  # cheap, POSIX-only quoting
    return f'"{x}"'

# Default SLURM flags; the caller may override any of them via slurm_config
DEFAULT_SLURM: Dict[str, str] = {
    "partition": "production",
    "output":    "slurm-%j.out",
}


class SBatchBlock(AnimalBlock):
    """
    Wrap *any* AnimalBlock and run it via `sbatch`.

    Behaviour
    ---------
    • **Skip submission** if every declared checkpoint already exists.
    • Otherwise:
      1. create a staging dir `<SAVE_DIR>/<tag>` (spaces → `_`);
      2. serialize the current bundle + wrapped block there;
      3. write a `run_block.py` runner (faulthandler + hardened jsonpickle);
      4. write a hardened `run.sbatch` (threads pinned, Arrow off, PYTHONPATH set);
      5. `sbatch run.sbatch` and exit so the pipeline resumes later.
    """
    def __init__(
        self,
        wrapped_block: AnimalBlock,
        *,
        venv_type: str = "conda",            # "conda" | "venv" | "poetry"
        venv_path: str = "TELF",             # env name or path
        slurm_config: Dict[str, str] | None = None,
        needs: Sequence[str] = (),
        provides: Sequence[str] = (),
        # tag: str = "SBatchWrapper",
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        **kw,
    ) -> None:
        tag = wrapped_block.tag
        super().__init__(
            needs=needs or wrapped_block.needs,
            provides=provides or wrapped_block.provides,
            tag=tag,
            conditional_needs=conditional_needs,
            init_settings=init_settings or {},
            call_settings=call_settings or {},
            **kw,
        )
        self.wrapped_block = wrapped_block
        self.venv_type     = venv_type.lower()
        self.venv_path     = venv_path
        self.slurm         = {**DEFAULT_SLURM, **(slurm_config or {})}

    def run(self, bundle: DataBundle) -> None:
        # 1) fast-skip on existing checkpoints
        ck_keys = getattr(self.wrapped_block, "checkpoint_keys", ())
        if ck_keys and all(
            (f"{self.wrapped_block.tag}.{ck}") in bundle and
            Path(bundle[f"{self.wrapped_block.tag}.{ck}"]).exists()
            for ck in ck_keys
        ):
            print(f"⏭ {self.tag}: existing checkpoints – skipping submission.")
            try:
                self.wrapped_block._after_checkpoint_skip(bundle)
            except AttributeError:
                pass
            return

        # 2) staging directory
        base_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]).resolve()
        workdir  = clean_path(base_dir / self.tag)
        workdir.mkdir(parents=True, exist_ok=True)

        # adjust SLURM output path
        orig_out = self.slurm.get("output", DEFAULT_SLURM["output"])
        self.slurm["output"] = str(workdir / orig_out)

        # 2b) resume if runner already completed
        done = workdir / "_complete.json"
        if done.exists():
            # runner wrote out a small dict: { provide_name: value, ... }
            resumed: dict = jsonpickle.decode(done.read_text(), keys=True)
            wrapped_tag = self.wrapped_block.tag
            for p in self.provides:
                ns = f"{wrapped_tag}.{p}"
                if ns not in bundle:
                    bundle[ns] = resumed[p]
            return

        # ensure SAVE_DIR is absolute
        bundle[SAVE_DIR_BUNDLE_KEY] = str(base_dir)

        # 3) serialize bundle + block (keys=True preserves non-string dict keys)
        (workdir / "input_bundle.json").write_text(
            jsonpickle.encode(bundle, keys=True), encoding="utf-8"
        )
        (workdir / "block.json").write_text(
            jsonpickle.encode(self.wrapped_block, keys=True), encoding="utf-8"
        )
        # sidecar with import path for robust re-instantiation
        block_cls_path = f"{self.wrapped_block.__class__.__module__}:{self.wrapped_block.__class__.__qualname__}"
        (workdir / "block_class.txt").write_text(block_cls_path, encoding="utf-8")

        # 4) runner script (Template placeholders; adds sys.path; spaCy Table handler; dict→class fallback)
        from string import Template

        runner = workdir / "run_block.py"
        project_root = Path.cwd()  # adjust if your repo root differs at submit time

        tpl = Template("""
import sys, importlib, types, faulthandler, jsonpickle
faulthandler.enable()

# Make sure we can import your package/modules
sys.path.insert(0, $PROJECT_ROOT)

# ---- jsonpickle handler: spaCy Table needs constructor run before inserts ----
from jsonpickle.handlers import BaseHandler
try:
    from spacy.lookups import Table as _SpaCyTable
    _TABLES = [_SpaCyTable]
except Exception:
    _TABLES = []

class _TableHandler(BaseHandler):
    def flatten(self, obj, data):
        data['py/object'] = obj.__class__.__module__ + '.' + obj.__class__.__name__
        data['items'] = list(obj.items())
        return data

    def restore(self, data):
        cls = self.restore_class(data)
        inst = cls()  # ensure __init__ runs (creates .bloom, etc.)
        if 'items' in data:
            kv = data['items']
        else:
            state = data.get('py/state', {})
            if isinstance(state, dict):
                mapping = state.get('data', state)
            else:
                mapping = {}
            kv = list(getattr(mapping, 'items', lambda: [])())
        if kv:
            inst.update(dict(kv))
        return inst

for _T in _TABLES:
    jsonpickle.handlers.register(_T, _TableHandler)

# ---- decode inputs (keys=True for non-string keys elsewhere) ----
with open($INPUT_JSON, "r", encoding="utf-8") as _f:
    bundle = jsonpickle.decode(_f.read(), keys=True)
with open($BLOCK_JSON, "r", encoding="utf-8") as _f:
    block = jsonpickle.decode(_f.read(), keys=True)

# If import failed and we got a dict, rebuild the class instance using the sidecar
if isinstance(block, dict):
    with open($BLOCK_CLASS_TXT, "r", encoding="utf-8") as _f:
        cls_path = _f.read().strip()
    mod_name, qualname = cls_path.split(":", 1)
    mod = importlib.import_module(mod_name)
    cls = mod
    for part in qualname.split("."):
        cls = getattr(cls, part)
    # instantiate and copy state (skip framework-managed attrs)
    NON_STATE = {"tag", "needs", "provides", "conditional_needs", "load_checkpoint"}
    inst = cls()
    for k, v in block.items():
        if k not in NON_STATE:
            setattr(inst, k, v)
    # keep tag/needs/provides if present in the dict
    for k in ("tag", "needs", "provides", "conditional_needs"):
        if k in block:
            setattr(inst, k, block[k])
    block = inst

# Run the block
out_bundle = block(bundle)

# collect just the wrapped block's provides
result = {}
for p in $PROVIDES:
    result[p] = out_bundle[block.tag + "." + p]

with open($DONE_JSON, "w", encoding="utf-8") as _f:
    _f.write(jsonpickle.encode(result, keys=True))
""")

        script = tpl.substitute(
            PROJECT_ROOT=repr(str(project_root)),
            INPUT_JSON=repr(str(workdir / "input_bundle.json")),
            BLOCK_JSON=repr(str(workdir / "block.json")),
            BLOCK_CLASS_TXT=repr(str(workdir / "block_class.txt")),
            DONE_JSON=repr(str(done)),
            PROVIDES=repr(list(self.provides)),
        )
        runner.write_text(textwrap.dedent(script), encoding="utf-8")

        # 5) sbatch script (export PYTHONPATH to make imports resolvable in the job)
        sbatch = workdir / "run.sbatch"
        lines  = [
            "#!/bin/bash",
            f"#SBATCH --job-name={_nospace(self.tag)}",
            *(f"#SBATCH --{k}={v}" for k, v in self.slurm.items()),
            "",
            "ulimit -c unlimited",
            "export OMP_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export NUMEXPR_NUM_THREADS=1",
            "export MKL_THREADING_LAYER=GNU",
            "export OPENBLAS_DISABLE_THREADS=1",
            "export PANDAS_ARROW_DISABLED=1",
            "",
            f"cd {shquote(workdir)}",
            f'export PYTHONPATH={shquote(project_root)}:"$PYTHONPATH"',
        ]

        match self.venv_type:
            case "venv":
                lines.append(f"source {shquote(Path(self.venv_path).resolve() / 'bin' / 'activate')}")
            case "conda":
                lines += [
                    "source $(conda info --base)/etc/profile.d/conda.sh",
                    f"conda activate {self.venv_path}",
                ]
            case "poetry":
                lines += [
                    f"cd {shquote(Path.cwd())}",
                    "poetry install --no-root --quiet",
                ]
            case _:
                raise ValueError(f"Unknown venv_type {self.venv_type!r}")

        lines.append(f"python -Xfaulthandler {shquote(runner)}")
        sbatch.write_text("\n".join(lines), encoding="utf-8")

        # 6) submit and exit
        print(f"🚀 {self.tag}: submitting via sbatch …")
        subprocess.run(["sbatch", str(sbatch)], check=True)
        raise SystemExit(f"{self.tag} submitted – pipeline will resume after job finishes.")
